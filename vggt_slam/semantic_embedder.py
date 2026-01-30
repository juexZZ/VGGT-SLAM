"""
Offline image semantic embedder for projecting CLIP features onto SAM2 segments.

Goal:
- Input: a folder of RGB images
- Output: one dense semantic embedding per image, saved to disk as NPZ:
    `embeddings/<frame_name>.npz` with `embedding` shaped (H, W, d).

How it works:
- SAM2 generates a set of instance masks for the image.
- Each mask is used to crop an object image (black background, bbox crop).
- CLIP produces one embedding vector per object crop.
- We "paint" a dense embedding image: pixels belonging to a mask get that mask's CLIP vector.
  (If masks overlap, later masks overwrite earlier ones; you can change this policy.)

Usage (run in your SAM2 environment to avoid dependency conflicts):

```bash
cd /local_data/jz4725/sam2
source .venv/bin/activate
python /local_data/jz4725/VGGT-SLAM/vggt_slam/semantic_embedder.py \
  --image_folder /path/to/images \
  --output_folder /path/to/embeddings \
  --ext .jpg .png .jpeg
```
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from transformers import CLIPProcessor, CLIPModel, SiglipModel, SiglipProcessor
from PIL import Image
import os
from pathlib import Path
import argparse
from typing import Iterable, Optional, Sequence, Tuple
from tqdm import tqdm
import multiprocessing as mp
import sys

class SAMCLIPEmbedder:
    def __init__(
        self,
        sam_checkpoint: str = "/local_data/jz4725/sam2/checkpoints/sam2.1_hiera_base_plus.pt",
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml",
        clip_model_name: str = "openai/clip-vit-base-patch32",
        bbox_expand_pct: float = 0.0,
        device: Optional[str] = None,
    ):
        #SAM Initialization
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model_name = clip_model_name

        self.sam = build_sam2(model_cfg, sam_checkpoint)
        self.sam.to(device=self.device)
        self.sam.eval()
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam,
            points_per_side=24,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processing
        )

        # CLIP/SigLIP Initialization
        self.is_siglip = "siglip" in clip_model_name.lower()
        if self.is_siglip:
            self.clip_model = SiglipModel.from_pretrained(clip_model_name)
            self.processor = SiglipProcessor.from_pretrained(clip_model_name, use_fast=True)
        else:
            self.clip_model = CLIPModel.from_pretrained(clip_model_name)
            self.processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
        
        self.clip_model.to(self.device)
        self.clip_model.eval()

        # Initialize other attributes
        self.image: Optional[np.ndarray] = None
        self.extracted_objects = []
        self.masks = []
        self.seperated_masks = []
        self.image_embeddings = []
        self.bbox_expand_pct = self._normalize_bbox_expand_pct(bbox_expand_pct)
        self.embedding_dim = self._infer_embedding_dim()
        print(f"embedding_dim: {self.embedding_dim}")

    def _infer_embedding_dim(self) -> int:
        config = getattr(self.clip_model, "config", None)
        for key in ("projection_dim", "hidden_size", "embed_dim"):
            value = getattr(config, key, None)
            if isinstance(value, int) and value > 0:
                return value
        return 512

    def _normalize_bbox_expand_pct(self, bbox_expand_pct: float) -> float:
        if bbox_expand_pct < 0:
            raise ValueError("bbox_expand_pct must be >= 0.")
        return float(bbox_expand_pct)

    def _expand_bbox(
        self,
        bbox: Sequence[float],
        image_hw: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        x, y, w, h = [float(v) for v in bbox]
        if self.bbox_expand_pct <= 0:
            return int(x), int(y), int(w), int(h)
        expand_w = w * self.bbox_expand_pct
        expand_h = h * self.bbox_expand_pct
        new_x = x - expand_w / 2.0
        new_y = y - expand_h / 2.0
        new_w = w + expand_w
        new_h = h + expand_h
        height, width = image_hw
        x0 = max(0, int(np.floor(new_x)))
        y0 = max(0, int(np.floor(new_y)))
        x1 = min(width, int(np.ceil(new_x + new_w)))
        y1 = min(height, int(np.ceil(new_y + new_h)))
        return x0, y0, max(0, x1 - x0), max(0, y1 - y0)

    def set_image(self, image_rgb: np.ndarray) -> None:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected image shaped (H,W,3), got {image_rgb.shape}")
        self.image = image_rgb
    
    def extract_object_images(self):
        self.extracted_objects = []
        if self.image is None:
            raise ValueError("Image not set. Call set_image() first.")
        for mask_data in self.masks:
            segmentation = mask_data['segmentation']
            bbox = mask_data['bbox']
        
            # 1. Create a 3-channel black background image
            # np.zeros_like(image) creates a black image with the same dimensions and type as the original
            black_background_image = np.zeros_like(self.image)
        
            # 2. Copy the pixels from the original image where the mask is True
            black_background_image[segmentation] = self.image[segmentation] #type: ignore
            
            # 3. Crop the image using the bounding box
            x, y, w, h = self._expand_bbox(bbox, self.image.shape[:2])
            x = max(0, x); y = max(0, y)
            w = max(0, w); h = max(0, h)
            cropped_object = black_background_image[y:y + h, x:x + w]
            if cropped_object.size == 0:
                print(f"[ERR] No object found for mask {mask_data['segmentation']} with bbox {bbox}")
                # remove the mask from the masks list
                continue
        
            self.extracted_objects.append(cropped_object)
        return self.extracted_objects

    
    #Takes in seperated masks using seperated_masks = [mask['segmentation'] for mask in masks]
    def visualize_masks(self):
        print(f"Found {len(self.seperated_masks)} separate masks.")
        
        # 3. Visualize the seperated masks using matplotlib
        if self.seperated_masks:
            # Create a subplot grid based on the number of masks
            num_masks = len(self.seperated_masks)
            # Aim for a grid that is roughly square
            cols = int(np.ceil(np.sqrt(num_masks)))
            rows = int(np.ceil(num_masks / cols))
            
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
            axes = axes.flatten() # Flatten the grid to easily iterate over it
            
            for i, mask in enumerate(self.seperated_masks):
                ax = axes[i]
                ax.imshow(mask) # The mask is a boolean array, imshow handles it well
                ax.set_title(f"Mask #{i + 1}")
                ax.axis('off') # Hide axes ticks
            
            # Hide any unused subplots
            for i in range(num_masks, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.show()
        else:
            print("No masks were generated with the current settings.")

    def visualize_extracted_objects(self):
        # 4. Visualize the extracted objects
        if self.extracted_objects:
            num_objects = len(self.extracted_objects)
            cols = int(np.ceil(np.sqrt(num_objects)))
            rows = int(np.ceil(num_objects / cols))
        
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            axes = axes.flatten()
        
            for i, obj_img in enumerate(self.extracted_objects):
                ax = axes[i]
                ax.imshow(obj_img)
                ax.set_title(f"Object #{i + 1}")
                ax.axis('off')
        
            for i in range(num_objects, len(axes)):
                axes[i].set_visible(False)
        
            plt.tight_layout()
            plt.show()
    

    
    
    
    def get_image_embedding(self, image):
        # 1. Load the image
        # image = Image.open(image_path).convert("RGB")
    
        # 2. Preprocess the image
        # inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device) #type: ignore
    
        # 3. Generate the embedding (feature vector)
        with torch.no_grad():
            if hasattr(self.clip_model, "get_image_features"):
                image_features = self.clip_model.get_image_features(pixel_values=inputs.pixel_values) #type: ignore
            else:
                outputs = self.clip_model(**inputs)
                image_features = outputs.image_embeds  #type: ignore
    
        # 4. Normalize the embedding (crucial for accurate cosine similarity)
        # The image_features will be of shape (1, embedding_dimension), e.g., (1, 768)
        image_embedding = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        # Convert to a NumPy array or Python list for storage, or keep as a torch tensor
        # return image_embedding.cpu().numpy()
        return image_embedding
    
    def get_clip_embeddings(self):
        # Convert object crops to PIL for CLIP processor stability.
        pil_objects = [Image.fromarray(obj.astype(np.uint8)) for obj in self.extracted_objects]
        self.image_embeddings = [self.get_image_embedding(x) for x in pil_objects]
        return self.image_embeddings
    
    # Example usage:
    # image_embedding = get_image_embedding("path/to/my/image.jpg")
    
    def get_text_embedding(self, text_query):
        # 1. Tokenize and preprocess the text
        inputs = self.processor(text=text_query, return_tensors="pt").to(self.device) #type: ignore
    
        # 2. Generate the embedding (feature vector)
        with torch.no_grad():
            if hasattr(self.clip_model, "get_text_features"):
                text_features = self.clip_model.get_text_features(**inputs) #type: ignore
            else:
                outputs = self.clip_model(**inputs)
                text_features = outputs.text_embeds  #type: ignore
    
        # 3. Normalize the embedding (crucial for accurate cosine similarity)
        # The text_features will be of shape (1, embedding_dimension), e.g., (1, 768)
        text_embedding = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        # Convert to a NumPy array or Python list
        return text_embedding

    def generate_masks(self):
        if self.image is None:
            raise ValueError("Image not set. Please set the image before generating masks.")
        
        for mask in self.mask_generator.generate(self.image):
            bbox = mask['bbox']
            if bbox[2] <= 0 or bbox[3] <= 0:
                continue
            self.masks.append(mask)
        self.seperated_masks = [mask['segmentation'] for mask in self.masks]
        return self.masks
    
    def get_image_from_lang_query(self, text_query):
        if not self.image_embeddings:
            raise ValueError("No image embeddings found. Please generate embeddings first.")
        
        query_embedding = self.get_text_embedding(text_query)
        
        # Compute cosine similarities
        # self.image_embeddings are torch tensors (1,d). query_embedding is numpy (1,d).
        qe = query_embedding.to(self.device).float()
        similarities = torch.stack([torch.matmul(e.to(self.device), qe.T).squeeze() for e in self.image_embeddings]).detach().cpu().numpy()
        
        # Find the index of the most similar image embedding
        most_similar_index = np.argmax(similarities)
        
        return self.extracted_objects[most_similar_index], similarities[most_similar_index]
    
    # Dummy get_logger to be replaced by the Node's logger
    def get_logger(self):
        import logging
        return logging.getLogger("ImageEmbedder")

    def get_best_match_from_text(self, text_query):
        """Finds the best object match for a given text query."""
        if not self.image_embeddings:
            self.get_logger().warn("No image embeddings available for matching.")
            return None, -1

        # Get text embedding
        inputs = self.processor(text=text_query, return_tensors="pt").to(self.device) #type: ignore
        with torch.no_grad():
            if hasattr(self.clip_model, "get_text_features"):
                text_features = self.clip_model.get_text_features(**inputs) #type: ignore
            else:
                outputs = self.clip_model(**inputs)
                text_features = outputs.text_embeds  #type: ignore
        text_embedding = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # Compute cosine similarities
        similarities = torch.matmul(torch.cat(self.image_embeddings), text_embedding.T).squeeze()
        
        # Find the index of the most similar object
        best_match_index = torch.argmax(similarities).item()
        
        return best_match_index, similarities[best_match_index].item() #type: ignore
    
    def get_fully_embedded_image(self):
        """
        Creates a dense 2D map where each pixel contains its corresponding CLIP embedding.
        This version is vectorized for performance.
        """
        if self.image is None:
            raise ValueError("Image not set. Call set_image() first.")

        image_shape = self.image.shape[:2]  # (height, width) #type: ignore

        # Squeeze to remove the extra dimension from CLIP's output
        clip_np = torch.cat(self.image_embeddings).cpu().numpy()

        embedding_size = clip_np.shape[1]
        height, width = image_shape[0], image_shape[1]
        
        # Start with a zero-filled image for pixels with no mask
        fully_embedded_image = np.zeros((height, width, embedding_size), dtype=np.float32)
        print(f"number of masks: {len(self.masks)}")
        # This is the fast part: iterate through masks and use boolean indexing
        for i, mask_data in enumerate(self.masks):
            segmentation_mask = mask_data['segmentation'] # This is a (height, width) boolean array
            # Assign the corresponding embedding vector to all True pixels in one go
            fully_embedded_image[segmentation_mask] = clip_np[i]
            
        return fully_embedded_image

    def embed_single_image(self, image_path: str, target_hw: tuple[int, int]) -> np.ndarray:
        """
        Produce dense semantic embedding for a single image.
        Returns: (H, W, d) float32 numpy array.
        """
        img_bgr = cv2.imread(image_path)
        # resize the image to target_hw
        img_bgr = cv2.resize(img_bgr, target_hw, interpolation=cv2.INTER_LINEAR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Reset per-image state
        self.masks = []
        self.seperated_masks = []
        self.extracted_objects = []
        self.image_embeddings = []

        self.set_image(img_rgb)
        self.generate_masks()
        self.extract_object_images()
        self.get_clip_embeddings()

        embedded = self.get_fully_embedded_image()
        if embedded is None:
            # Fall back to zeros (no masks) so downstream code can still load shape-consistent data.
            h, w = img_rgb.shape[:2]
            embedded = np.zeros((h, w, self.embedding_dim), dtype=np.float32)
        return embedded.astype(np.float32)

    def save_embedding_npz(self, embedding: np.ndarray, output_path: str, image_path: Optional[str] = None) -> None:
        """
        Save embedding to a compressed NPZ. Stores:
        - embedding: (H,W,d) float32/float16
        - image_path: optional string
        """
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if image_path is None:
            np.savez_compressed(output_path, embedding=embedding)
        else:
            np.savez_compressed(output_path, embedding=embedding, image_path=str(image_path))

    def save_masks_visualization(self, output_path: str, alpha: float = 0.5) -> None:
        """
        Save a visualization of SAM2 masks overlaid on the current image.
        """
        if self.image is None:
            raise ValueError("Image not set. Call set_image() before saving mask visualization.")
        if not self.masks:
            # No masks: just save the original image for debugging consistency.
            img_bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img_bgr)
            return

        rng = np.random.RandomState(0)
        overlay = self.image.astype(np.float32).copy()
        base = self.image.astype(np.float32)

        for mask_data in self.masks:
            segmentation_mask = mask_data["segmentation"]
            color = rng.randint(0, 256, size=(3,), dtype=np.uint8).astype(np.float32)
            overlay[segmentation_mask] = (1.0 - alpha) * base[segmentation_mask] + alpha * color

        vis_rgb = np.clip(overlay, 0, 255).astype(np.uint8)
        vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, vis_bgr)

    def embed_folder_to_npz(
        self,
        image_folder: str,
        output_folder: str,
        exts: Sequence[str] = (".jpg", ".jpeg", ".png"),
        overwrite: bool = False,
        gpu_ids: Optional[Sequence[int]] = None,
    ) -> None:
        """
        Embed all images in `image_folder` and write one NPZ per image to `output_folder`.
        """
        image_folder_p = Path(image_folder)
        output_folder_p = Path(output_folder)
        output_folder_p.mkdir(parents=True, exist_ok=True)

        exts_l = tuple(e.lower() for e in exts)
        image_paths = sorted([p for p in image_folder_p.glob("*") if p.suffix.lower() in exts_l])
        if len(image_paths) == 0:
            raise RuntimeError(f"No images found in {image_folder} with extensions {exts}")

        if gpu_ids and len(gpu_ids) > 1:
            chunks = _chunk_list(image_paths, len(gpu_ids))
            ctx = mp.get_context("spawn")
            processes = []
            for gpu_id, chunk in zip(gpu_ids, chunks):
                if not chunk:
                    continue
                proc = ctx.Process(
                    target=_embed_worker,
                    args=(
                        gpu_id,
                        [str(p) for p in chunk],
                        str(output_folder_p),
                        tuple(exts),
                        overwrite,
                        self.clip_model_name,
                        self.bbox_expand_pct,
                    ),
                )
                proc.start()
                processes.append(proc)
            for proc in processes:
                proc.join()
            return

        for p in tqdm(image_paths):
            out_path = output_folder_p / f"{p.stem}.npz"
            if out_path.exists() and not overwrite:
                print(f"[skip] {p.name} -> {out_path}")
                continue
            print(f"[embed] {p.name}")
            emb = self.embed_single_image(str(p), target_hw=(518, 518))
            self.save_embedding_npz(emb, str(out_path), image_path=str(p))
            mask_viz_path = output_folder_p / "mask_viz" / f"{p.stem}_masks.png"
            self.save_masks_visualization(str(mask_viz_path))
            print(f"[saved] {out_path} shape={emb.shape} dtype={emb.dtype}")


def _chunk_list(items: Sequence[Path], num_chunks: int) -> list[list[Path]]:
    if num_chunks <= 0:
        return [list(items)]
    chunks: list[list[Path]] = [[] for _ in range(num_chunks)]
    for idx, item in enumerate(items):
        chunks[idx % num_chunks].append(item)
    return chunks


def _embed_worker(
    gpu_id: int,
    image_paths: Sequence[str],
    output_folder: str,
    exts: Sequence[str],
    overwrite: bool,
    clip_model_name: str,
    bbox_expand_pct: float,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    embedder = SAMCLIPEmbedder(
        clip_model_name=clip_model_name,
        bbox_expand_pct=bbox_expand_pct,
        device=device,
    )
    output_folder_p = Path(output_folder)
    output_folder_p.mkdir(parents=True, exist_ok=True)
    exts_lower = [e.lower() for e in exts]
    for i, image_path in enumerate(image_paths):
        p = Path(image_path)
        if p.suffix.lower() not in exts_lower:
            continue
        out_path = output_folder_p / f"{p.stem}.npz"
        if out_path.exists() and not overwrite:
            print(f"[gpu {gpu_id}] [skip] {p.name} -> {out_path}")
            continue
        print(f"[gpu {gpu_id}] [embed] {p.name}")
        emb = embedder.embed_single_image(str(p), target_hw=(518, 518))
        embedder.save_embedding_npz(emb, str(out_path), image_path=str(p))
        mask_viz_path = output_folder_p / "mask_viz" / f"{p.stem}_masks.png"
        embedder.save_masks_visualization(str(mask_viz_path))
        print(f"[gpu {gpu_id}] [saved] {out_path} shape={emb.shape} dtype={emb.dtype}, its progress is {i+1}/{len(image_paths)}")


def _embed_multi_gpu(
    image_folder: str,
    output_folder: str,
    exts: Sequence[str],
    overwrite: bool,
    gpu_ids: Sequence[int],
    clip_model_name: str,
    bbox_expand_pct: float,
) -> None:
    image_folder_p = Path(image_folder)
    output_folder_p = Path(output_folder)
    output_folder_p.mkdir(parents=True, exist_ok=True)

    exts_l = tuple(e.lower() for e in exts)
    image_paths = sorted([p for p in image_folder_p.glob("*") if p.suffix.lower() in exts_l])
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {image_folder} with extensions {exts}")

    chunks = _chunk_list(image_paths, len(gpu_ids))
    ctx = mp.get_context("spawn")
    processes = []
    for gpu_id, chunk in zip(gpu_ids, chunks):
        if not chunk:
            continue
        proc = ctx.Process(
            target=_embed_worker,
            args=(
                gpu_id,
                [str(p) for p in chunk],
                str(output_folder_p),
                tuple(exts),
                overwrite,
                clip_model_name,
                bbox_expand_pct,
            ),
        )
        proc.start()
        processes.append(proc)
    for proc in processes:
        proc.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dense semantic embeddings for a folder of images (SAM2 + CLIP).")
    parser.add_argument("--image_folder", type=str, required=True, help="Folder containing images (flat folder).")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to write per-image .npz embeddings.")
    parser.add_argument("--ext", type=str, nargs="*", default=[".jpg", ".jpeg", ".png"], help="Allowed extensions.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz files.")
    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="HF model name for CLIP/SigLIP (e.g., google/siglip-so400m-patch14-384).",
    )
    parser.add_argument(
        "--bbox_expand_pct",
        type=float,
        default=0.0,
        help="Expand bbox by this pct (0.1=10%, 0.2=20%).",
    )
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="*",
        default=None,
        help="GPU ids for parallel embedding (e.g., 0 1 2).",
    )
    args = parser.parse_args()

    if args.gpu_ids and len(args.gpu_ids) > 1:
        _embed_multi_gpu(
            args.image_folder,
            args.output_folder,
            exts=tuple(args.ext),
            overwrite=args.overwrite,
            gpu_ids=args.gpu_ids,
            clip_model_name=args.clip_model_name,
            bbox_expand_pct=args.bbox_expand_pct,
        )
        sys.exit(0)

    device = None
    if args.gpu_ids and len(args.gpu_ids) == 1 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_ids[0])
        device = f"cuda:{args.gpu_ids[0]}"

    embedder = SAMCLIPEmbedder(
        clip_model_name=args.clip_model_name,
        bbox_expand_pct=args.bbox_expand_pct,
        device=device,
    )
    
    # # demo verify sam and feature quality
    # img_path = "/local_data/jz4725/sam2/notebooks/images/groceries.jpg"
    # text_query = "grocery bags"
    # image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # embed_single_image = embedder.embed_single_image(img_path)
    # print(f"Image embeddings shape: {embed_single_image.shape}")
    # print(f"Image embeddings dtype: {embed_single_image.dtype}")
    # nearest_image, similarity = embedder.get_image_from_lang_query(text_query)
    # print(f"Most similar image has similarity score: {similarity}")
    # plt.imshow(nearest_image)
    # plt.axis('off')
    # plt.savefig('debug_nearest_image.jpg')
    # plt.close()
    
    # embed folder to npz
    embedder.embed_folder_to_npz(
        args.image_folder,
        args.output_folder,
        exts=tuple(args.ext),
        overwrite=args.overwrite,
        gpu_ids=args.gpu_ids,
    )
    