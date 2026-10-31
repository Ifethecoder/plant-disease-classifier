"""Run the matched comparison, preparing one dataset for both initializations."""
from dataclasses import replace
from pathlib import Path

from .dataset import load_metadata, make_loaders, make_splits, materialize_development, seed_everything
from .evaluate import comparison_table, evaluate_checkpoint
from .models import build_model
from .train import TrainConfig, fit, parse_args, save_json


def run_comparison(config: TrainConfig, output_dir: str | Path, cache_dir: str | Path):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    metadata = load_metadata(Path(cache_dir) / "hub")
    splits = make_splits(metadata, config.fraction, config.seed, config.val_fraction)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "splits.json", splits)
    save_json(output / "dataset_schema.json", {"features": metadata["upstream_schema"],
              "published_counts": metadata["published_counts"], "revision": metadata["revision"]})
    print(f"Dataset counts: {splits['counts']}; image fraction: {splits['actual_image_fraction']:.4f}", flush=True)
    development = materialize_development(metadata, splits, cache_dir)
    runs = {}
    for name, pretrained in (("random", False), ("pretrained", True)):
        current = replace(config, pretrained=pretrained)
        seed_everything(current.seed)
        # Recreate the shuffle generator to match mini-batch order in both runs.
        loaders = make_loaders(development, splits, current.batch_size, current.seed,
                               current.num_workers, current.image_size)
        model = build_model(len(metadata["classes"]), pretrained, current.freeze_backbone)
        run_dir = output / name
        fit(model, *loaders, current, run_dir, metadata["classes"], splits["split_sha256"], split_manifest=splits)
        # Release training model/optimizer memory before checkpoint evaluation.
        del model
        evaluate_checkpoint(run_dir / "best.pt", loaders[1])
        runs[name] = run_dir
    return comparison_table(runs, output)


if __name__ == "__main__":
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args = vars(parse_args(comparison=True))
    output, cache = args.pop("output"), args.pop("cache_dir")
    run_comparison(TrainConfig(**args), output, cache)
