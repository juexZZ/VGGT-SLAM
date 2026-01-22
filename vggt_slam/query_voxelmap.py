# Load semantic voxel map
# Query the voxel map with a clip embedding vector
# get the top k most similar voxels using semantic voxel map's query_with_embedding function
# find the latest (highest submap id and largest frame index) frame filename of the most similar voxel
# visualize the voxel map by highlighting the most similar voxel
# visualize the retrieved frame image
import torch
from vggt_slam.semantic_voxel import SemanticVoxelMap
from transformers import CLIPProcessor, CLIPModel
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query semantic voxel map with a clip embedding vector.")
    parser.add_argument("--port", type=int, default=8081, help="Port for viser server.")
    parser.add_argument("--voxel_map_dir", type=str, required=True, help="Directory containing semantic voxel map (semantic_voxels.npz + frame_names.json).")
    parser.add_argument("--query_prompt", type=str, required=True, help="text prompt to query the semantic voxel map.")
    parser.add_argument("--output_dir", type=str, default="retrieval_results", help="Directory to save the retrieval results.")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory containing images to retrieve.")
    args = parser.parse_args()

    voxel_map = SemanticVoxelMap.load_from_directory(args.voxel_map_dir)
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)
    clip_model.eval()
    inputs = clip_processor(text=args.query_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_embedding = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
    topk_indices, topk_coords, topk_similarities = voxel_map.query_with_embedding(
        text_embedding.cpu().numpy(), 
        top_k=1
    )
    print("topk retrieval results for query prompt: ", args.query_prompt)
    save_dir = os.path.join(args.output_dir, "_".join(args.query_prompt.split(" ")))
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(topk_indices)):
        voxel_index = topk_indices[i]
        voxel_coord = topk_coords[i]
        similarity = topk_similarities[i]
        frame_name, submap_id, frame_id = voxel_map.get_latest_frame_at_voxel(voxel_index)
        print(f"Top {i+1} retrieval result: Voxel index: {voxel_index}, voxel coordinate: {voxel_coord}, similarity: {similarity}")
        print(f"submap id: {submap_id}, frame id: {frame_id}")
        print(f"frame name: {frame_name}")
        title_x = f"Top {i+1} retrieval result: Voxel index: {voxel_index}, voxel coordinate: {voxel_coord}, similarity: {similarity}, frame name: {frame_name}"
        plt.title(title_x)
        image_path = os.path.join(args.image_dir, frame_name)
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
        save_path = os.path.join(save_dir, f"top{i+1}_retrieval_result.png")
        plt.savefig(save_path)
        plt.close()
    
    # visualize the voxel map by highlighting the color of the most similar voxel
    voxel_map.visualize(
        port=args.port,
        render_mode="cubes",
        color_mode="query",
        base_color=(0.75, 0.75, 0.75),
        highlight_color=(1.0, 0.0, 0.0),
        wireframe=False,
        opacity=0.5,
        keep_alive=True,
        query_voxel_indices=topk_indices,
    )
    
    