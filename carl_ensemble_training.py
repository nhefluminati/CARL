from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import lightning as light

from CARL import CARL, LossHistory
from carl_preprocessing import CARLPreprocessor, NSBIDataset



def _train_member_worker(trainer: "CARLEnsembleTrainer", member_id: int, seed: int, gpu_id: int | None, bootstrap_fraction: float) -> dict:
    """Top-level worker so ProcessPoolExecutor can pickle it reliably."""
    return trainer.train_member(member_id, seed, gpu_id, bootstrap_fraction)

@dataclass
class ModelConfig:
    n_layers: int = 10
    n_nodes: int = 128
    learning_rate: float = 1e-6
    init_variance: float = 0.01
    xavier: bool = False


@dataclass
class TrainingConfig:
    max_epochs: int = 400
    batch_size: int = 512
    val_batch_size: int = 2028
    num_workers: int = 4
    log_every_n_steps: int = 3
    precision: str = "32-true"


class CARLEnsembleTrainer:
    def __init__(
        self,
        base_dataset: NSBIDataset,
        base_train_idx: np.ndarray,
        val_idx: np.ndarray,
        run_name: str,
        output_dir: str | Path,
        model_config: ModelConfig,
        training_config: TrainingConfig,
    ):
        self.base_dataset = base_dataset
        self.base_train_idx = np.asarray(base_train_idx)
        self.val_idx = np.asarray(val_idx)
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_config = model_config
        self.training_config = training_config

    def train_member(self, member_id: int, seed: int, gpu_id: int | None, bootstrap_fraction: float = 1.0) -> dict:
        light.seed_everything(seed, workers=True)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        member_name = f"{self.run_name}_member{member_id:03d}"
        member_dir = self.output_dir / f"member_{member_id:03d}"
        member_dir.mkdir(parents=True, exist_ok=True)

        dataset = self.base_dataset.clone_with_fresh_weights()
        bootstrap_idx = CARLPreprocessor.make_bootstrap_indices(self.base_train_idx, seed=seed, bootstrap_fraction=bootstrap_fraction)
        CARLPreprocessor.rebalance_training_weights(dataset, bootstrap_idx, verbose=True)
        train_loader, val_loader = CARLPreprocessor.make_loaders(
            dataset,
            bootstrap_idx,
            self.val_idx,
            batch_size=self.training_config.batch_size,
            val_batch_size=self.training_config.val_batch_size,
            num_workers=self.training_config.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        model = CARL(
            dataset.n_features,
            n_layers=self.model_config.n_layers,
            n_nodes=self.model_config.n_nodes,
            learning_rate=self.model_config.learning_rate,
            init_variance=self.model_config.init_variance,
            xavier=self.model_config.xavier,
            name=member_name,
            checkpoint_dir=str(member_dir / "checkpoints"),
            enable_checkpoint=True,
        )
        loss_history = LossHistory()
        use_gpu = torch.cuda.is_available() and gpu_id is not None
        trainer = light.Trainer(
            max_epochs=self.training_config.max_epochs,
            log_every_n_steps=self.training_config.log_every_n_steps,
            accelerator="gpu" if use_gpu else "cpu",
            devices=[gpu_id] if use_gpu else 1,
            strategy="auto",
            callbacks=[loss_history],
            logger=False,
            precision=self.training_config.precision,
            enable_progress_bar=True,
        )
        trainer.fit(model, train_loader, val_loader)

        ckpt_path = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else ""
        if not ckpt_path:
            ckpt_path = str(member_dir / f"{member_name}_last.ckpt")
            trainer.save_checkpoint(ckpt_path)

        summary = {
            "member": member_id,
            "seed": seed,
            "gpu_id": gpu_id,
            "checkpoint": ckpt_path,
            "train_loss": loss_history.train_loss,
            "val_loss": loss_history.val_loss,
            "val_norm": loss_history.val_norm,
            "n_bootstrap_events": int(len(bootstrap_idx)),
        }
        with open(member_dir / "history.json", "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    def train_sequential(self, n_members: int, seed: int, gpu_ids: list[int] | None, bootstrap_fraction: float = 1.0) -> list[dict]:
        summaries = []
        gpu_ids = gpu_ids or [None]
        for member_id in range(n_members):
            gpu_id = gpu_ids[member_id % len(gpu_ids)]
            summaries.append(self.train_member(member_id, seed + member_id, gpu_id, bootstrap_fraction))
        self._write_manifest(summaries)
        return summaries

    def train_parallel(self, n_members: int, seed: int, gpu_ids: list[int] | None, bootstrap_fraction: float = 1.0) -> list[dict]:

        if gpu_ids is None or len(gpu_ids) == 0:
            # CPU training is intentionally kept serial to avoid oversubscribing cores/RAM.
            return self.train_sequential(n_members, seed, gpu_ids=None, bootstrap_fraction=bootstrap_fraction)

        max_workers = min(n_members, len(gpu_ids))
        assignments = [
            (member_id, seed + member_id, gpu_ids[member_id % len(gpu_ids)])
            for member_id in range(n_members)
        ]

        # Use spawn rather than fork because CUDA + fork is unsafe once CUDA has
        # been initialized anywhere in the parent process.
        ctx = mp.get_context("spawn")
        summaries = []

        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {
                executor.submit(
                    _train_member_worker,
                    self,
                    member_id,
                    member_seed,
                    gpu_id,
                    bootstrap_fraction,
                ): member_id
                for member_id, member_seed, gpu_id in assignments
            }

            for future in as_completed(futures):
                member_id = futures[future]
                try:
                    summaries.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"Ensemble member {member_id} failed") from exc

        summaries = sorted(summaries, key=lambda item: item["member"])
        self._write_manifest(summaries)
        return summaries

    def _write_manifest(self, summaries: list[dict]) -> None:
        manifest = {
            "run_name": self.run_name,
            "members": summaries,
            "feature_names": self.base_dataset.feature_names,
            "mean": self.base_dataset.mean.tolist(),
            "std": self.base_dataset.std.tolist(),
            "n_features": self.base_dataset.n_features,
        }
        with open(self.output_dir / f"ensemble_manifest_{self.run_name}.json", "w") as f:
            json.dump(manifest, f, indent=2)

    @staticmethod
    def load_ensemble_from_manifest(manifest_path: str | Path, map_location: str | torch.device = "cpu") -> tuple[list[CARL], dict]:
        manifest_path = Path(manifest_path)
        with open(manifest_path) as f:
            manifest = json.load(f)
        models = []
        for member in manifest["members"]:
            ckpt = member["checkpoint"]
            model = CARL.load_from_checkpoint(ckpt, map_location=map_location)
            model.eval()
            models.append(model)
        return models, manifest
