from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


class EnsemblePredictor:
    
    def __init__(
        self,
        models: list[torch.nn.Module],
        device: str | torch.device = "cpu",
        reference_scales: list[float] | np.ndarray | None = None,
        eps: float = 1e-6,
    ):
        self.models = models
        self.device = torch.device(device)
        self.eps = float(eps)

        if reference_scales is None:
            reference_scales = np.ones(len(models), dtype=np.float64)
        reference_scales = np.asarray(reference_scales, dtype=np.float64).reshape(-1)
        if len(reference_scales) != len(models):
            raise ValueError(
                f"Need one reference scale per model: got {len(reference_scales)} scales "
                f"for {len(models)} models."
            )
        self.reference_scales = reference_scales

        for model in self.models:
            model.to(self.device)
            model.eval()

    @torch.no_grad()
    def predict_mean_and_members(self, loader):
        member_scores = [[] for _ in self.models]
        labels, weights, features = [], [], []

        for x, y, w in loader:
            x_dev = x.to(self.device, non_blocking=True)
            for i, model in enumerate(self.models):
                member_scores[i].append(
                    model(x_dev).flatten().detach().cpu().numpy()
                )
            labels.append(y.numpy().flatten())
            weights.append(w.numpy().flatten())
            features.append(x.numpy())

        scores_members = np.stack(
            [np.concatenate(s) for s in member_scores],
            axis=0,
        )

        scores_clipped = np.clip(scores_members, self.eps, 1.0 - self.eps)
        r_members = (
            self.reference_scales[:, None]
            * scores_clipped
            / (1.0 - scores_clipped)
        )
        r_mean = r_members.mean(axis=0)
        r_std = (
            r_members.std(axis=0, ddof=1)
            if len(self.models) > 1
            else np.zeros(r_members.shape[1])
        )

        # Equivalent score corresponding to the ratio-space ensemble mean.
        scores_equiv = r_mean / (1.0 + r_mean)

        return {
            "scores_members": scores_members,
            "scores_mean_raw": scores_members.mean(axis=0),
            "scores_std_raw": (
                scores_members.std(axis=0, ddof=1)
                if len(self.models) > 1
                else np.zeros(scores_members.shape[1])
            ),
            "scores_mean": scores_equiv,
            "scores_std": np.zeros_like(scores_equiv),
            "r_members": r_members,
            "r_mean": r_mean,
            "r_std": r_std,
            "reference_scales": self.reference_scales.copy(),
            "labels": np.concatenate(labels),
            "weights": np.concatenate(weights),
            "features": np.concatenate(features, axis=0),
        }


class CARLDiagnostics:
    def __init__(self, run_name: str, output_dir: str | Path = "."):
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_diagnostics(self, histories: list[dict]) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
        for hist in histories:
            member = hist["member"]
            ax1.plot(
                np.arange(len(hist["train_loss"])),
                hist["train_loss"],
                alpha=0.35,
                label=f"train m{member}",
            )
            ax1.plot(
                np.arange(len(hist["val_loss"])),
                hist["val_loss"],
                alpha=0.35,
                linestyle="--",
                label=f"val m{member}",
            )
            if "val_norm" in hist:
                ax2.plot(
                    np.arange(len(hist["val_norm"])),
                    hist["val_norm"],
                    alpha=0.45,
                    label=f"norm m{member}",
                )
        ax1.set_ylabel("BCE loss", fontsize=12)
        ax1.set_title("Training / Validation BCE per Ensemble Member")
        ax1.grid(alpha=0.3)
        ax2.axhline(1.0, linestyle="--")
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Norm", fontsize=12)
        ax2.set_title("NSBI normalization test")
        ax2.grid(alpha=0.3)
        if len(histories) <= 10:
            ax1.legend(fontsize=8, ncol=2)
            ax2.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"training_diagnostics_ensemble_{self.run_name}.pdf")
        plt.close()

    def plot_roc(self, scores: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None) -> float:
        labels = np.asarray(labels).reshape(-1)
        scores = np.asarray(scores).reshape(-1)
        finite = np.isfinite(scores) & np.isfinite(labels)
        labels_for_roc = labels[finite]
        scores_for_roc = scores[finite]

        fpr, tpr, _ = roc_curve(labels_for_roc, scores_for_roc)
        roc_auc = roc_auc_score(labels_for_roc, scores_for_roc)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ensemble AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve: Ratio-space Ensemble Mean")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"roc_curve_ensemble_{self.run_name}.pdf")
        plt.close()
        print(f"Ensemble ROC AUC: {roc_auc:.4f}")
        return roc_auc

    def plot_calibration(self, p: np.ndarray, t: np.ndarray, w: np.ndarray, s_nbins: int = 30) -> None:
        p = np.asarray(p).reshape(-1)
        t = np.asarray(t).reshape(-1)
        w = np.asarray(w).reshape(-1)
        finite = np.isfinite(p) & np.isfinite(t) & np.isfinite(w)
        p = p[finite]
        t = t[finite]
        w = w[finite]

        s_bins = np.linspace(0.0, 1.0, s_nbins + 1)
        s_bin_centers = 0.5 * (s_bins[:-1] + s_bins[1:])
        sig_per_bin = np.zeros(s_nbins)
        bkg_per_bin = np.zeros(s_nbins)
        sig_err = np.zeros(s_nbins)
        bkg_err = np.zeros(s_nbins)
        for i in range(s_nbins):
            if i < s_nbins - 1:
                mask = (p >= s_bins[i]) & (p < s_bins[i + 1])
            else:
                mask = (p >= s_bins[i]) & (p <= s_bins[i + 1])
            t_bin = t[mask]
            w_bin = w[mask]
            sig_mask = t_bin == 1.0
            bkg_mask = t_bin == 0.0
            sig_per_bin[i] = np.sum(w_bin[sig_mask])
            bkg_per_bin[i] = np.sum(w_bin[bkg_mask])
            sig_err[i] = np.sqrt(np.sum(w_bin[sig_mask] ** 2))
            bkg_err[i] = np.sqrt(np.sum(w_bin[bkg_mask] ** 2))

        denom = sig_per_bin + bkg_per_bin
        valid = denom > 0.0
        s_true = np.full(s_nbins, np.nan)
        s_err = np.full(s_nbins, np.nan)
        s_true[valid] = sig_per_bin[valid] / denom[valid]
        s_err[valid] = np.sqrt(
            (sig_err[valid] * bkg_per_bin[valid] / denom[valid] ** 2) ** 2
            + (bkg_err[valid] * sig_per_bin[valid] / denom[valid] ** 2) ** 2
        )
        plt.figure(figsize=(7, 6))
        plt.errorbar(
            s_bin_centers[valid],
            s_true[valid],
            yerr=s_err[valid],
            fmt="o",
            capsize=3,
            label="True weighted signal fraction",
        )
        plt.plot([0, 1], [0, 1], "--", label="Ideal: truth = bin center")
        plt.xlabel("ratio-space ensemble equivalent NN output")
        plt.ylabel("True signal probability")
        plt.title("Diagnostic: true signal fraction vs ensemble output")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"diagnostic_calibration_ensemble_{self.run_name}.pdf")
        plt.close()

    def plot_reweighting_closure(
        self,
        r_mean: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray,
        features_scaled: np.ndarray,
        feature_names: list[str],
        mean: np.ndarray,
        std: np.ndarray,
        feature_name: str | None = None,
        feature_index: int | None = None,
        n_xbins: int = 40,
    ) -> None:
        if feature_name is None and feature_index is None:
            feature_index = min(6, len(feature_names) - 1)
            feature_name = feature_names[feature_index]
        elif feature_index is None:
            feature_index = feature_names.index(feature_name)
        elif feature_name is None:
            feature_name = feature_names[feature_index]

        r_mean = np.asarray(r_mean).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        weights = np.asarray(weights).reshape(-1)
        features_scaled = np.asarray(features_scaled)

        finite = (
            np.isfinite(r_mean)
            & np.isfinite(labels)
            & np.isfinite(weights)
            & np.isfinite(features_scaled[:, feature_index])
        )
        r_mean = r_mean[finite]
        labels = labels[finite]
        weights = weights[finite]
        features_scaled = features_scaled[finite]

        xobs = features_scaled[:, feature_index] * std[feature_index] + mean[feature_index]
        num_mask = labels == 1.0
        den_mask = labels == 0.0
        xobs_numerator = xobs[num_mask]
        xobs_denominator = xobs[den_mask]
        w_numerator = weights[num_mask]
        w_denominator = weights[den_mask]
        r_denominator = r_mean[den_mask]

        xmin, xmax = np.min(xobs), np.max(xobs)
        xbins = np.linspace(xmin, xmax, n_xbins + 1)
        xcenters = 0.5 * (xbins[:-1] + xbins[1:])
        xwidths = np.diff(xbins)

        h_num_mc, _ = np.histogram(xobs_numerator, bins=xbins, weights=w_numerator)
        h_denom, _ = np.histogram(xobs_denominator, bins=xbins, weights=w_denominator)
        h_num_carl, _ = np.histogram(
            xobs_denominator,
            bins=xbins,
            weights=w_denominator * r_denominator,
        )
        h_num_mc_var, _ = np.histogram(xobs_numerator, bins=xbins, weights=w_numerator ** 2)
        h_denom_var, _ = np.histogram(xobs_denominator, bins=xbins, weights=w_denominator ** 2)
        h_num_carl_var, _ = np.histogram(
            xobs_denominator,
            bins=xbins,
            weights=(w_denominator * r_denominator) ** 2,
        )

        h_num_mc_density = h_num_mc / xwidths
        h_denom_density = h_denom / xwidths
        h_num_carl_density = h_num_carl / xwidths

        ratio_num_mc_over_denom = np.divide(
            h_num_mc,
            h_denom,
            out=np.full_like(h_num_mc, np.nan, dtype=float),
            where=h_denom > 0,
        )
        ratio_num_carl_over_denom = np.divide(
            h_num_carl,
            h_denom,
            out=np.full_like(h_num_carl, np.nan, dtype=float),
            where=h_denom > 0,
        )
        ratio_carl_over_mc = np.divide(
            h_num_carl,
            h_num_mc,
            out=np.full_like(h_num_carl, np.nan, dtype=float),
            where=h_num_mc > 0,
        )
        ratio_carl_over_mc_err = np.divide(
            np.sqrt(h_num_carl_var),
            h_num_mc,
            out=np.full_like(h_num_carl, np.nan, dtype=float),
            where=h_num_mc > 0,
        )

        fig, (ax1, ax2, ax3) = plt.subplots(
            3,
            1,
            gridspec_kw={"height_ratios": [2, 1, 1]},
            figsize=(5, 6),
            sharex=True,
        )
        ax1.stairs(
            h_num_carl_density,
            xbins,
            color="cornflowerblue",
            linewidth=1.5,
            label="Target from CARL-reweighted reference",
        )
        ax1.stairs(
            h_num_mc_density,
            xbins,
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label="Target MC",
        )
        ax1.stairs(
            h_denom_density,
            xbins,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Reference MC",
        )
        ax2.stairs(ratio_num_carl_over_denom, xbins, color="cornflowerblue", linewidth=1.5)
        ax2.stairs(ratio_num_mc_over_denom, xbins, color="blue", linestyle="--", linewidth=1.5)
        ax2.stairs(np.ones_like(h_denom), xbins, color="black", linestyle="--", linewidth=1.5)
        ax3.stairs(np.ones_like(h_num_mc), xbins, color="blue", linestyle="--", linewidth=1.5)
        valid = h_num_mc > 0
        ax3.errorbar(
            xcenters[valid],
            ratio_carl_over_mc[valid],
            yerr=ratio_carl_over_mc_err[valid],
            xerr=xwidths[valid] / 2.0,
            fmt="none",
            color="cornflowerblue",
            linewidth=1.5,
        )
        ax1.legend(frameon=False, fontsize=10)
        ax1.set_yscale("log")
        ax1.set_ylabel("Density of events", fontsize=12)
        ax2.set_ylabel("target / ref", fontsize=12)
        ax3.set_ylabel("CARL / MC", fontsize=12)
        ax3.set_xlabel(feature_name, fontsize=12)
        ax3.set_xlim(xmin, xmax)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.savefig(
            self.output_dir / f"carl_reweight_ensemble_{feature_name}_{self.run_name}.pdf",
            bbox_inches="tight",
        )
        plt.close()

    def write_member_spread(self, scores_members: np.ndarray, r_members: np.ndarray | None = None) -> None:
        plt.figure(figsize=(7, 5))
        if r_members is None:
            spread = (
                scores_members.std(axis=0, ddof=1)
                if scores_members.shape[0] > 1
                else np.zeros(scores_members.shape[1])
            )
            xlabel = "per-event ensemble std of NN output"
        else:
            spread = (
                r_members.std(axis=0, ddof=1)
                if r_members.shape[0] > 1
                else np.zeros(r_members.shape[1])
            )
            xlabel = "per-event ensemble std of physical ratio"
        plt.hist(spread, bins=50)
        plt.xlabel(xlabel)
        plt.ylabel("Events")
        plt.title("Ensemble predictive spread")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"ensemble_score_spread_{self.run_name}.pdf")
        plt.close()
