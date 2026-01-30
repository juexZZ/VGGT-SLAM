import torch
import numpy as np
import json
import re
import os
from transformers import CLIPProcessor, CLIPModel

def get_ts(f):
    match = re.search(r'_(\d+)\.', str(f))
    return int(match.group(1)) if match else None

class BaseEvaluator:
    def ingest_chapter(self, chapter_metadata):
        pass
    def evaluate(self, voxel_map, step_info):
        raise NotImplementedError


# --- SEARCH VALIDITY EVALUATOR ---
class SearchValidityEvaluator(BaseEvaluator):
    def __init__(self, annotation_path, time_tolerance_ns=5e7):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.time_tolerance = time_tolerance_ns

        # 1. Load Ground Truth & Sort by Timestamp
        with open(annotation_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, dict) and "images" in data:
            self.annotations = data["images"]
        elif isinstance(data, list):
            self.annotations = data
        else:
            self.annotations = []
        
        # Sort annotations by timestamp to allow binary search
        def get_annotation_ts(item):
            if "timestamp" in item:
                return item["timestamp"]
            fname = item.get("file", item.get("file_name", ""))
            return get_ts(fname) if fname else None

        self.annotations.sort(key=lambda x: get_annotation_ts(x) if get_annotation_ts(x) is not None else 0)
        self.timestamps = [get_annotation_ts(x) for x in self.annotations]

    def evaluate(self, voxel_map, step_info):
        queries = step_info.get("queries") or step_info.get("query")
        if not queries:
            return None
        if isinstance(queries, str):
            queries = [queries]
        top_k = int(step_info.get("top_k", 1))

        results = []
        feats = voxel_map.get_features()
        if feats is None or len(feats) == 0:
            return [{"query": q, "found": False, "reason": "Voxel map empty"} for q in queries]

        for user_query in queries:
            # 1. Embed Query
            inputs = self.clip_processor(text=[user_query], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                query_embedding = self.clip_model.get_text_features(**inputs).cpu().numpy()[0]
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm

            # 2. Perform Search on Voxel Map
            topk_indices, topk_coords, topk_similarities = voxel_map.query_with_embedding(
                query_embedding, top_k=top_k
            )
            if not topk_indices:
                results.append({"query": user_query, "found": False, "reason": "No voxels returned"})
                continue

            voxel_index = topk_indices[0]
            voxel_coord = topk_coords[0]
            score = topk_similarities[0]
            frame_name, submap_id, frame_id = voxel_map.get_latest_frame_at_voxel(voxel_index)
            retrieved_ts = get_ts(frame_name) if frame_name else None

            # 3. Check Validity against Ground Truth
            relevant_gt_indices = [
                i for i, ann in enumerate(self.annotations)
                if user_query.lower() in str(ann.get('label', '')).lower()
            ]

            is_valid = False
            closest_gt = None
            min_dt = float('inf')
            for idx in relevant_gt_indices:
                gt_ts = self.timestamps[idx]
                if gt_ts is None or retrieved_ts is None:
                    continue
                dt = abs(gt_ts - retrieved_ts)
                if dt < min_dt:
                    min_dt = dt
                    closest_gt = self.annotations[idx]
                if dt <= self.time_tolerance:
                    is_valid = True

            results.append({
                "query": user_query,
                "found": True,
                "valid": is_valid,
                "score": float(score),
                "retrieved_ts": retrieved_ts,
                "retrieved_voxel_index": int(voxel_index),
                "retrieved_voxel_coord": [int(x) for x in voxel_coord],
                "closest_gt_label": closest_gt['label'] if closest_gt and 'label' in closest_gt else "None",
                "time_diff_ns": float(min_dt) if min_dt != float('inf') else None,
                "retrieved_img": os.path.basename(frame_name) if frame_name else None,
                "retrieved_submap_id": int(submap_id) if submap_id is not None else None,
                "retrieved_frame_id": str(frame_id) if frame_id is not None else None,
            })

        return results


# --- VOXEL COUNT EVALUATOR ---
class VoxelCountEvaluator(BaseEvaluator):
    def evaluate(self, voxel_map, step_info):
        centers = voxel_map.get_centers_world()
        feats = voxel_map.get_features()
        feature_dim = int(feats.shape[1]) if feats is not None and feats.ndim == 2 else 0
        return {
            "num_voxels": int(centers.shape[0]) if centers is not None else 0,
            "feature_dim": feature_dim,
            "voxel_size": float(voxel_map.get_voxel_size()),
        }


# --- PERFORMANCE EVALUATOR ---
class PerformanceEvaluator(BaseEvaluator):
    """
    Placeholder for performance stats during voxel map evaluation.
    """
    def __init__(self):
        pass
    
    def evaluate(self, voxel_map, step_info):
        """
        Semantic voxel maps do not track timing stats by default.
        """
        return {
            "status": "not_available",
            "reason": "Semantic voxel map evaluation does not collect timing stats.",
        }

def get_evaluator(name, config):
    if name == "search_validity_metric":
        return SearchValidityEvaluator(
            annotation_path=config['annotation_path']
        )
    elif name in {"node_count_metric", "voxel_count_metric"}:
        return VoxelCountEvaluator()
    elif name in {"navigability_metric", "localization_metric"}:
        print(f"Warning: Evaluator '{name}' is not supported for voxel maps.")
        return BaseEvaluator()
    elif name == "performance_metric":
        return PerformanceEvaluator()
    else:
        print(f"Warning: Unknown evaluator '{name}'")
        return BaseEvaluator()