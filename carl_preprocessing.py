from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py as h5
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class NSBIDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        w: np.ndarray | torch.Tensor,
        feature_names: list[str],
        mean: np.ndarray,
        std: np.ndarray,
        n_target: int,
        n_reference: int,
    ):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
        self.w = torch.as_tensor(w, dtype=torch.float32).reshape(-1, 1)
        self.feature_names = list(feature_names)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.n_target = int(n_target)
        self.n_reference = int(n_reference)
        self.n_features = self.x.shape[1]

    def clone_with_fresh_weights(self) -> "NSBIDataset":
        return NSBIDataset(
            self.x,
            self.y,
            self.w.clone(),
            self.feature_names,
            self.mean,
            self.std,
            self.n_target,
            self.n_reference,
        )

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]


@dataclass(frozen=True)
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray


class CARLPreprocessor:
    """Loads HDF5 templates and reproduces preprocessing.

    """

    def __init__(self, signal_path: str, background_paths: list[str] | tuple[str, ...], run_name: str):
        self.signal_path = signal_path
        self.background_paths = tuple(background_paths)
        self.run_name = run_name

    def load_dataset(self, output_dir: str | Path = ".") -> NSBIDataset:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with h5.File(self.signal_path, "r") as f_target:
            keys_target = sorted(k for k in f_target.keys() if k != "weight")
            features_target = np.stack([f_target[k][:] for k in keys_target], axis=1)
            weights_target = f_target["weight"][:].astype(np.float64)

        weights_target = weights_target / weights_target.sum()
        feature_names = keys_target

        features_ref_list = []
        weights_ref_list = []
        for path in self.background_paths:
            with h5.File(path, "r") as f_ref:
                keys_ref = sorted(k for k in f_ref.keys() if k != "weight")
                if keys_ref != keys_target:
                    raise ValueError(f"Feature mismatch in {path}: {keys_ref} != {keys_target}")
                features_ref = np.stack([f_ref[k][:] for k in keys_ref], axis=1)
                weights_ref = f_ref["weight"][:].astype(np.float64)
            weights_ref = weights_ref / weights_ref.sum()
            features_ref_list.append(features_ref)
            weights_ref_list.append(weights_ref)

        features_ref = np.concatenate(features_ref_list, axis=0)
        weights_ref = np.concatenate(weights_ref_list, axis=0)

        x = np.concatenate([features_target, features_ref], axis=0)
        y = np.concatenate([np.ones(len(weights_target)), np.zeros(len(weights_ref))]).reshape(-1, 1)
        w = np.concatenate([weights_target, weights_ref]).reshape(-1, 1)

        # Do not fit/apply the feature scaler here. To match the old toolkit,
        # the train/validation split is created first and the scaler is fitted
        # on the training subset only. Placeholder values are overwritten by
        # fit_scaler_on_train_and_transform(...).
        mean = np.zeros(x.shape[1], dtype=np.float64)
        std = np.ones(x.shape[1], dtype=np.float64)

        print("Signal weight sum before train-only rebalancing:", weights_target.sum())
        print("Total background weight sum before train-only rebalancing:", weights_ref.sum())

        return NSBIDataset(
            x=x,
            y=y,
            w=w,
            feature_names=feature_names,
            mean=mean,
            std=std,
            n_target=len(weights_target),
            n_reference=len(weights_ref),
        )

    @staticmethod
    def make_train_val_split(dataset: NSBIDataset, train_fraction: float = 0.8, split_seed: int = 52) -> SplitIndices:
        n_target = dataset.n_target
        n_ref = dataset.n_reference
        target_indices = np.arange(n_target)
        ref_indices = np.arange(n_target, n_target + n_ref)

        target_perm = np.random.default_rng(split_seed).permutation(target_indices)
        ref_perm = np.random.default_rng(split_seed).permutation(ref_indices)
        target_split = int(train_fraction * n_target)
        ref_split = int(train_fraction * n_ref)

        train_idx = np.concatenate([target_perm[:target_split], ref_perm[:ref_split]])
        val_idx = np.concatenate([target_perm[target_split:], ref_perm[ref_split:]])

        rng_final = np.random.default_rng(split_seed)
        return SplitIndices(train_idx=rng_final.permutation(train_idx), val_idx=rng_final.permutation(val_idx))

    @staticmethod
    def make_bootstrap_indices(train_idx: np.ndarray, seed: int, bootstrap_fraction: float = 1.0) -> np.ndarray:
        n = int(round(len(train_idx) * bootstrap_fraction))
        if n <= 0:
            raise ValueError("bootstrap_fraction gives zero sampled events.")
        rng = np.random.default_rng(seed)
        return rng.choice(train_idx, size=n, replace=True)

    @staticmethod
    def fit_scaler_on_train_and_transform(
        dataset: NSBIDataset,
        train_idx: np.ndarray,
        output_dir: str | Path = ".",
        run_name: str = "default",
    ) -> NSBIDataset:
        """Fit mean/std on the nominal training subset and scale all events.

        This matches the old toolkit logic: split first, fit the scaler only on
        training events, then transform both training and validation events with
        that scaler. The scaler is shared by all bootstrap ensemble members.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_idx_t = torch.as_tensor(train_idx, dtype=torch.long)
        x_train = dataset.x[train_idx_t]

        mean_t = x_train.mean(dim=0)
        std_t = x_train.std(dim=0, unbiased=False)
        std_t = torch.where(std_t < 1e-8, torch.ones_like(std_t), std_t)

        dataset.x = (dataset.x - mean_t) / std_t
        dataset.mean = mean_t.detach().cpu().numpy().astype(np.float64)
        dataset.std = std_t.detach().cpu().numpy().astype(np.float64)

        np.savetxt(output_dir / f"mean_{run_name}.csv", dataset.mean, delimiter=";")
        np.savetxt(output_dir / f"std_{run_name}.csv", dataset.std, delimiter=";")

        print("Mean fitted on training subset:")
        print(dataset.mean)
        print("STD fitted on training subset:")
        print(dataset.std)

        return dataset

    @staticmethod
    def rebalance_training_weights(dataset: NSBIDataset, train_indices: np.ndarray, verbose: bool = True) -> float:
        # Uses duplicate entries in train_indices, so bootstrap multiplicities enter the yield estimate.
        train_indices_t = torch.as_tensor(train_indices, dtype=torch.long)
        y_train = dataset.y[train_indices_t].flatten()
        w_train = dataset.w[train_indices_t].flatten()
        target_mask = y_train == 1.0
        reference_mask = y_train == 0.0
        target_yield = w_train[target_mask].sum()
        reference_yield = w_train[reference_mask].sum()
        if target_yield <= 0 or reference_yield <= 0:
            raise ValueError("Cannot rebalance training weights: target and reference yields must both be positive.")

        reference_scale = target_yield / reference_yield
        unique_ref_indices = torch.unique(train_indices_t[reference_mask])
        dataset.w[unique_ref_indices] *= reference_scale

        if verbose:
            y_after = dataset.y[train_indices_t].flatten()
            w_after = dataset.w[train_indices_t].flatten()
            print("Training target yield before rebalancing:", target_yield.item())
            print("Training reference yield before rebalancing:", reference_yield.item())
            print("Applied train-only reference weight scale:", reference_scale.item())
            print("Training target yield after rebalancing:", w_after[y_after == 1.0].sum().item())
            print("Training reference yield after rebalancing:", w_after[y_after == 0.0].sum().item())

        return float(reference_scale.detach().cpu().item())

    @staticmethod
    def make_loaders(
        dataset: NSBIDataset,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        batch_size: int = 512,
        val_batch_size: int = 2028,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> tuple[DataLoader, DataLoader]:
        train_data = Subset(dataset, train_idx.tolist())
        val_data = Subset(dataset, val_idx.tolist())
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=False,
        )
        return train_loader, val_loader
