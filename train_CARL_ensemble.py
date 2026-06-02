from __future__ import annotations

import argparse
from pathlib import Path
import torch

from carl_preprocessing import CARLPreprocessor
from carl_ensemble_training import CARLEnsembleTrainer, ModelConfig, TrainingConfig
from carl_diagnostics import CARLDiagnostics, EnsemblePredictor


def parse_gpus(gpu_args: list[int] | None) -> list[int] | None:
    if gpu_args is None or len(gpu_args) == 0 or gpu_args == [-1]:
        return None
    return gpu_args


def main():
    parser = argparse.ArgumentParser(description="Train a bootstrap ensemble of CARL likelihood-ratio estimators.")
    parser.add_argument("--name", type=str, required=True, help="Common suffix/name used for all output files.")
    parser.add_argument("--signal", type=str, required=True, help="Path to the signal/target h5 file.")
    parser.add_argument("--backgrounds", type=str, nargs="+", required=True, help="One or more background/reference h5 files.")
    parser.add_argument("--gpus", type=int, nargs="*", default=None, help="GPU IDs to use round-robin, e.g. --gpus 0 1 2 3. Use --gpus -1 for CPU.")
    parser.add_argument("--n-ensemble", type=int, default=10, help="Number of bootstrap ensemble members.")
    parser.add_argument("--seed", type=int, default=52, help="Base random seed. Member i uses seed+i.")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--bootstrap-fraction", type=float, default=1.0, help="Bootstrap sample size as a fraction of the nominal training set.")
    parser.add_argument("--output-dir", type=str, default="carl_ensemble_outputs")
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-batch-size", type=int, default=2028)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=10)
    parser.add_argument("--n-nodes", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--init-variance", type=float, default=0.01)
    parser.add_argument("--xavier", action="store_true")
    parser.add_argument("--diagnostic-feature", type=str, default=None, help="Feature name for the reweighting closure plot. Default reproduces old feature index 6 when available.")
    parser.add_argument("--predict-gpu", type=int, default=None, help="GPU ID used for final ensemble diagnostics. Defaults to first training GPU if available.")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("medium")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpus = parse_gpus(args.gpus)
    preprocessor = CARLPreprocessor(args.signal, tuple(args.backgrounds), args.name)
    base_dataset = preprocessor.load_dataset(output_dir=output_dir)
    split = preprocessor.make_train_val_split(base_dataset, train_fraction=args.train_fraction, split_seed=args.seed)

    model_config = ModelConfig(
        n_layers=args.n_layers,
        n_nodes=args.n_nodes,
        learning_rate=args.learning_rate,
        init_variance=args.init_variance,
        xavier=args.xavier,
    )
    training_config = TrainingConfig(
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
    )
    ensemble_trainer = CARLEnsembleTrainer(
        base_dataset=base_dataset,
        base_train_idx=split.train_idx,
        val_idx=split.val_idx,
        run_name=args.name,
        output_dir=output_dir,
        model_config=model_config,
        training_config=training_config,
    )
    histories = ensemble_trainer.train_parallel(
        n_members=args.n_ensemble,
        seed=args.seed,
        gpu_ids=gpus,
        bootstrap_fraction=args.bootstrap_fraction,
    )

    # Run diagnostics once
    
    manifest_path = output_dir / f"ensemble_manifest_{args.name}.json"
    pred_gpu = args.predict_gpu if args.predict_gpu is not None else (gpus[0] if gpus else None)
    pred_device = f"cuda:{pred_gpu}" if torch.cuda.is_available() and pred_gpu is not None else "cpu"
    models, _ = CARLEnsembleTrainer.load_ensemble_from_manifest(manifest_path, map_location=pred_device)

    # Validation weights are untouched holdout weights, matching the original script's validation semantics.
    _, val_loader = CARLPreprocessor.make_loaders(
        base_dataset,
        split.train_idx,
        split.val_idx,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    predictions = EnsemblePredictor(models, device=pred_device).predict_mean_and_members(val_loader)
    diagnostics = CARLDiagnostics(args.name, output_dir=output_dir)
    diagnostics.plot_training_diagnostics(histories)
    diagnostics.plot_roc(predictions["scores_mean"], predictions["labels"], predictions["weights"])
    diagnostics.plot_calibration(predictions["scores_mean"], predictions["labels"], predictions["weights"])
    diagnostics.plot_reweighting_closure(
        scores=predictions["scores_mean"],
        labels=predictions["labels"],
        weights=predictions["weights"],
        features_scaled=predictions["features"],
        feature_names=base_dataset.feature_names,
        mean=base_dataset.mean,
        std=base_dataset.std,
        feature_name=args.diagnostic_feature,
    )
    diagnostics.write_member_spread(predictions["scores_members"])


if __name__ == "__main__":
    main()
