from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import lightning as light
from lightning.pytorch.callbacks import ModelCheckpoint


class LossHistory(light.Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.val_norm = []
        self.val_norm_loss = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            self.train_loss.append(
                trainer.callback_metrics["train_loss"].item()
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            self.val_loss.append(
                trainer.callback_metrics["val_loss"].item()
            )

        if "val_norm" in trainer.callback_metrics:
            self.val_norm.append(
                trainer.callback_metrics["val_norm"].item()
            )

        if "val_norm_loss" in trainer.callback_metrics:
            self.val_norm_loss.append(
                trainer.callback_metrics["val_norm_loss"].item()
            )


class CARL(light.LightningModule):
    def __init__(
        self,
        n_features,
        n_layers,
        n_nodes,
        learning_rate,
        init_variance=0.01,
        name="default",
        xavier=False,
        checkpoint_dir=None,
        enable_checkpoint=True,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.lr = learning_rate
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.enable_checkpoint = enable_checkpoint

        layers = [
            nn.Linear(n_features, n_nodes),
            nn.SiLU(),
        ]

        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Linear(n_nodes, n_nodes),
                    nn.SiLU(),
                    nn.Dropout(0.08),
                ]
            )

        layers.extend(
            [
                nn.Linear(n_nodes, 1),
                nn.Sigmoid(),
            ]
        )

        self.model = nn.Sequential(*layers)

        self.loss_fn = nn.BCELoss(reduction="none")

        self._val_preds = []
        self._val_targets = []
        self._val_weights = []

        def gaussian_init(var: float):
            std = var ** 0.5

            def _init(m):
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=std)
                    if m.bias is not None:
                        m.bias.data.zero_()

            return _init

        def xavier_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

        if xavier:
            self.model.apply(xavier_init)
        else:
            self.model.apply(gaussian_init(init_variance))

    def configure_callbacks(self):
        callbacks = super().configure_callbacks()

        if self.enable_checkpoint:
            checkpoint_kwargs = dict(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="{epoch:02d}-{val_loss:.6f}_" + f"{self.name}",
            )

            if self.checkpoint_dir is not None:
                Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                checkpoint_kwargs["dirpath"] = self.checkpoint_dir

            callbacks.append(ModelCheckpoint(**checkpoint_kwargs))

        return callbacks

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, w = batch

        y_hat = self.model(x).flatten()
        y = y.flatten()
        w = w.flatten()

        loss = (self.loss_fn(y_hat, y) * w).sum() / w.sum()

        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, w = batch

        y_hat = self.model(x).flatten()
        y = y.flatten()
        w = w.flatten()

        # Validation BCE. This is now the actual val_loss used for checkpointing.
        val_loss = (self.loss_fn(y_hat, y) * w).sum() / w.sum()

        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Keep buffers for the NSBI normalization diagnostic.
        self._val_preds.append(y_hat.detach())
        self._val_targets.append(y.detach())
        self._val_weights.append(w.detach())

        return val_loss

    def on_validation_epoch_end(self):
        if len(self._val_preds) == 0:
            return

        preds = torch.cat(self._val_preds, dim=0)
        targets = torch.cat(self._val_targets, dim=0)
        weights = torch.cat(self._val_weights, dim=0)

        if self.trainer.world_size > 1:
            preds = self.all_gather(preds).reshape(-1)
            targets = self.all_gather(targets).reshape(-1)
            weights = self.all_gather(weights).reshape(-1)

        preds = preds.float()
        targets = targets.float()
        weights = weights.float()

        ref_mask = targets == 0.0

        preds_ref = preds[ref_mask]
        weights_ref = weights[ref_mask]

        eps = 1e-8
        preds_ref = torch.clamp(preds_ref, eps, 1.0 - eps)

        r_ref = preds_ref / (1.0 - preds_ref)

        val_norm = torch.sum(r_ref * weights_ref) / torch.sum(weights_ref)
        val_norm_loss = torch.abs(1.0 - val_norm)

        self.log(
            "val_norm",
            val_norm,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val_norm_loss",
            val_norm_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self._val_preds.clear()
        self._val_targets.clear()
        self._val_weights.clear()

    def predict_step(self, batch, batch_idx):
        x = batch if not isinstance(batch, (tuple, list)) else batch[0]
        return self.model(x).flatten()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=1,
            eta_min=1e-9,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "cosine_annealing_warm_restarts",
            },
        }