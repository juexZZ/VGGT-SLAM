# voxel map evaluation manager
import json
import os
import argparse
import multiprocessing
import time
from itertools import product
from tqdm import tqdm

from semantic_voxel import SemanticVoxelMap
import voxel_evaluators as evaluators

def run_experiment(config_packet):
    dataset_path = config_packet['dataset_path']
    annotation_path = config_packet['annotation_path']
    voxel_dir = config_packet.get('voxel_dir')
    params = config_packet['params']
    requested_evals = config_packet['eval_funcs']
    if not voxel_dir:
        return {"status": "failed", "error": "Missing voxel_dir in config packet."}
    if not os.path.exists(voxel_dir):
        return {"status": "failed", "error": f"voxel_dir not found: {voxel_dir}"}

    voxel_map = SemanticVoxelMap.load_from_directory(voxel_dir)

    # --- SETUP EVALUATORS ---
    active_evaluators = []
    for eval_name in requested_evals:
        try:
            ev_obj = evaluators.get_evaluator(eval_name, {**config_packet})
            if hasattr(ev_obj, "ingest_chapter"):
                ev_obj.ingest_chapter([])
            active_evaluators.append((eval_name, ev_obj))
        except Exception as e:
            print(f"Warning: Failed to initialize evaluator {eval_name}: {e}")

    # --- EVALUATION ---
    queries = params.get("queries", None)
    if isinstance(queries, str):
        queries = [queries]
    top_k = params.get("top_k", 1)

    step_info = {
        "timestamp": time.time(),
        "experiment_name": config_packet.get('experiment_name', 'default_eval'),
        "dataset_path": dataset_path,
        "annotation_path": annotation_path,
        "voxel_dir": voxel_dir,
        "top_k": top_k,
    }
    if queries:
        step_info["queries"] = queries
        if len(queries) == 1:
            step_info["query"] = queries[0]

    run_history = []
    voxel_metrics = {}
    for name, evaluator in active_evaluators:
        try:
            res = evaluator.evaluate(voxel_map, step_info)
            voxel_metrics[name] = res
        except Exception as e:
            voxel_metrics[name] = {"error": str(e)}

    run_history.append({"voxel_map": os.path.basename(voxel_dir), "metrics": voxel_metrics})

    return {
        "status": "success",
        "config": config_packet,
        "history": run_history
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="eval_config.json")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    print(f"--- Starting Evaluation Manager: {config.get('experiment_name', 'Unnamed')} ---")

    jobs = []
    datasets = config['datasets']
    generators = config.get('generators', [None])
    detectors = config.get('detectors', [None])
    hyperparameters = config.get('hyperparameters', [{}])

    combos = product(datasets, generators, detectors, hyperparameters)

    for ds, gen, det, params in combos:
        voxel_dir = ds.get('voxel_dir', ds.get('voxel_map_dir', '')).strip()
        jobs.append({
            "dataset_path": ds['path'], 
            "annotation_path": ds.get('annotation_file', os.path.join(ds['path'], 'annotations.json')),
            "pcd_path": ds.get('pcd_file', '').strip(),
            "colmap_path": ds.get('colmap_file', '').strip(),
            "voxel_dir": voxel_dir,
            "generator": gen, # should all be none for voxel maps
            "detector": det,
            "params": params,
            "eval_funcs": config['eval_functions'],
            "experiment_name": config.get('experiment_name', 'default_eval')
        })

    print(f"Generated {len(jobs)} jobs. Running with {args.workers} workers...")

    results = []
    if args.workers == 1:
        for job in tqdm(jobs):
            results.append(run_experiment(job))
    else:
        with multiprocessing.Pool(processes=args.workers) as pool:
            results = list(tqdm(pool.imap(run_experiment, jobs), total=len(jobs)))

    out_dir = "results/evals"
    os.makedirs(out_dir, exist_ok=True)
    exp_name = config.get('experiment_name', 'default_eval')
    safe_exp_name = "".join([c for c in exp_name if c.isalpha() or c.isdigit() or c=='_' or c=='-']).rstrip()
    out_filename = os.path.join(out_dir, f"{safe_exp_name}.json")
    
    with open(out_filename, 'w') as f:
        json.dump({"experiment_meta": config, "timestamp": time.time(), "runs": results}, f, indent=4)

    print(f"\n--- Complete. Saved to {out_filename} ---")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()