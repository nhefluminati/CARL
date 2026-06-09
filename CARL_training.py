import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import lightning as light
import h5py as h5

torch.set_float32_matmul_precision("medium")
light.seed_everything(52)


# ==========================
# Dataset
# ==========================

class NSBIDataset(Dataset):
    def __init__(self, targetPath: str, referencePaths, run_name: str):
        self.run_name = run_name

        if isinstance(referencePaths, str):
            referencePaths = (referencePaths,)

        # --- Load target / signal ---
        with h5.File(targetPath, "r") as fileTarget:
            keys_target = sorted([k for k in fileTarget.keys() if k != "weight"])
            featuresTarget = np.stack([fileTarget[k][:] for k in keys_target], axis=1)
            weightTarget = fileTarget["weight"][:].astype(np.float64)

        self.feature_names = keys_target

        # Normalize target to define a probability-density training measure.
        # The target/reference yield matching is intentionally NOT done here;
        # it is applied only to the training subset after the train/validation split.
        weightTarget = weightTarget / weightTarget.sum()

        # --- Load references / backgrounds ---
        featuresReference_list = []
        weightReference_list = []

        for referencePath in referencePaths:
            with h5.File(referencePath, "r") as fileReference:
                keys_ref = sorted([k for k in fileReference.keys() if k != "weight"])
                assert keys_target == keys_ref, f"Feature mismatch in {referencePath}!"

                featuresRef = np.stack([fileReference[k][:] for k in keys_ref], axis=1)
                weightsRef = fileReference["weight"][:].astype(np.float64)

            # Step 1: each background sample gets identical total weight
            weightsRef = weightsRef / weightsRef.sum()

            featuresReference_list.append(featuresRef)
            weightReference_list.append(weightsRef)

        featuresReference = np.concatenate(featuresReference_list, axis=0)
        weightReference = np.concatenate(weightReference_list, axis=0)

        # Do not force the total reference yield to match the target yield here.
        # That would happen before the train/validation split and can leave the
        # actual sampled training subset imbalanced.

        # --- Combine datasets ---
        x = np.concatenate([featuresTarget, featuresReference], axis=0)
        w = np.concatenate([weightTarget, weightReference]).reshape(-1, 1)

        y = np.concatenate([
            np.ones(len(weightTarget)),        # signal / target
            np.zeros(len(weightReference))     # background / reference
        ]).reshape(-1, 1)

        self.n_target = len(weightTarget)
        self.n_reference = len(weightReference)

        # --- Feature normalization ---
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1.0
        x = (x - self.mean) / self.std

        np.savetxt(f"mean_{self.run_name}.csv", self.mean, delimiter=";")
        np.savetxt(f"std_{self.run_name}.csv", self.std, delimiter=";")

        print("Mean:")
        print(self.mean)
        print("STD:")
        print(self.std)

        print("Signal weight sum before train-only rebalancing:", weightTarget.sum())
        print("Total background weight sum before train-only rebalancing:", weightReference.sum())

        # --- Convert to tensors ---
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.w = torch.tensor(w, dtype=torch.float32)

        self.n_features = self.x.shape[1]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.w[idx]


def rebalance_training_weights(dataset: NSBIDataset, train_indices: np.ndarray) -> None:
    """Rescale only training-set reference weights to match target yield.

    This keeps validation/test weights unchanged, while ensuring that the actual
    sampled training subset passed to the classifier has equal weighted target
    and reference yields
    """
    train_indices_t = torch.as_tensor(train_indices, dtype=torch.long)

    y_train = dataset.y[train_indices_t].flatten()
    w_train = dataset.w[train_indices_t].flatten()

    target_mask = y_train == 1.0
    reference_mask = y_train == 0.0

    target_yield = w_train[target_mask].sum()
    reference_yield = w_train[reference_mask].sum()

    if target_yield <= 0 or reference_yield <= 0:
        raise ValueError(
            "Cannot rebalance training weights: target and reference yields "
            "must both be positive."
        )

    reference_scale = target_yield / reference_yield
    reference_train_indices = train_indices_t[reference_mask]
    dataset.w[reference_train_indices] *= reference_scale

    # Diagnostics after the in-place update
    y_train_after = dataset.y[train_indices_t].flatten()
    w_train_after = dataset.w[train_indices_t].flatten()

    target_yield_after = w_train_after[y_train_after == 1.0].sum().item()
    reference_yield_after = w_train_after[y_train_after == 0.0].sum().item()

    print("Training target yield before rebalancing:", target_yield.item())
    print("Training reference yield before rebalancing:", reference_yield.item())
    print("Applied train-only reference weight scale:", reference_scale.item())
    print("Training target yield after rebalancing:", target_yield_after)
    print("Training reference yield after rebalancing:", reference_yield_after)


# ==========================
# Arguments / Load data
# ==========================

parser = argparse.ArgumentParser(description="Train CARL model with consistent output names.")

parser.add_argument(
    "--name",
    type=str,
    required=True,
    help="Common suffix/name used for all output files, e.g. SBI.",
)

parser.add_argument(
    "--signal",
    type=str,
    required=True,
    help="Path to the signal/target h5 file.",
)

parser.add_argument(
    "--backgrounds",
    type=str,
    nargs="+",
    required=True,
    help="One or more background/reference h5 files.",
)

parser.add_argument(
    "--gpu",
    type = int,
    required = True,
    help = "GPU Id to run on"
)

args = parser.parse_args()

RUN_NAME = args.name
signal_file = args.signal
background_files = tuple(args.backgrounds)

full_dataset = NSBIDataset(signal_file, background_files, RUN_NAME)


# ==========================
# Train/val split
# ==========================

SPLIT_SEED = 52
TRAIN_FRACTION = 0.8

n_target = full_dataset.n_target
n_ref = full_dataset.n_reference

rng_target = np.random.default_rng(SPLIT_SEED)
rng_ref = np.random.default_rng(SPLIT_SEED)

target_indices = np.arange(n_target)
ref_indices = np.arange(n_target, n_target + n_ref)

target_perm = rng_target.permutation(target_indices)
ref_perm = rng_ref.permutation(ref_indices)

target_split = int(TRAIN_FRACTION * n_target)
ref_split = int(TRAIN_FRACTION * n_ref)

train_idx = np.concatenate([
    target_perm[:target_split],
    ref_perm[:ref_split],
])

val_idx = np.concatenate([
    target_perm[target_split:],
    ref_perm[ref_split:],
])

# Shuffle order without changing train/val membership
rng_final = np.random.default_rng(SPLIT_SEED)
train_idx = rng_final.permutation(train_idx)
val_idx = rng_final.permutation(val_idx)

# NSBI/CARL needs equal weighted target/reference yield in the training sample.
# Apply this only after the split, so the actually sampled training subset is
# balanced and the validation subset remains an untouched holdout.
rebalance_training_weights(full_dataset, train_idx)

train_data = torch.utils.data.Subset(full_dataset, train_idx)
val_data = torch.utils.data.Subset(full_dataset, val_idx)

train_loader = DataLoader(
    train_data,
    batch_size=512,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    val_data,
    batch_size=2028,
    num_workers=4,
    pin_memory=True,
)


# ==========================
# Set model
# ==========================

from CARL import *

model = CARL(
    full_dataset.n_features,
    n_layers=10,
    n_nodes=128,
    learning_rate=1e-6,
    name = RUN_NAME
)


# ==========================
# Set GPUs
# ==========================

GPU_IDS = [args.gpu]
use_gpu = torch.cuda.is_available() and len(GPU_IDS) > 0


# ==========================
# Train
# ==========================

loss_callback = LossHistory()

trainer = light.Trainer(
    max_epochs=400,
    log_every_n_steps=3,
    accelerator="gpu" if use_gpu else "cpu",
    devices=GPU_IDS if use_gpu else 1,
    strategy="ddp" if use_gpu and len(GPU_IDS) > 1 else "auto",
    callbacks=[loss_callback],
    logger = False
)

trainer.fit(model, train_loader, val_loader)


# ==========================
# Collect losses
# ==========================

# ==========================
# Collect losses + norm
# ==========================

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(7, 8),
    sharex=True,
)

train_epochs = np.arange(len(loss_callback.train_loss))
val_epochs = np.arange(len(loss_callback.val_loss))
norm_epochs = np.arange(len(loss_callback.val_norm))

# ==========================
# Loss curves
# ==========================

ax1.plot(
    train_epochs,
    loss_callback.train_loss,
    label="train loss",
)

ax1.plot(
    val_epochs,
    loss_callback.val_loss,
    label="validation objective",
)

ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Training / Validation Objective")
ax1.legend()
ax1.grid(alpha=0.3)

# ==========================
# Normalization metric
# ==========================

ax2.plot(
    norm_epochs,
    loss_callback.val_norm,
    label="validation norm",
)

ax2.axhline(
    1.0,
    linestyle="--",
)

ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Norm", fontsize=12)
ax2.set_title("NSBI normalization test")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"training_diagnostics_{RUN_NAME}.pdf"
)

plt.close()


# ==========================
# ROC Curve
# ==========================

model.eval()
all_scores = []
all_labels = []
all_weights = []

with torch.no_grad():
    for x, y, w in val_loader:
        x = x.to(model.device)

        scores = model(x).flatten().cpu().numpy()
        all_scores.append(scores)
        all_labels.append(y.numpy().flatten())
        all_weights.append(w.numpy().flatten())

all_scores = np.concatenate(all_scores)
all_labels = np.concatenate(all_labels)
all_weights = np.concatenate(all_weights)

fpr, tpr, _ = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(f"roc_curve_{RUN_NAME}.pdf")
plt.close()

print(f"ROC AUC: {roc_auc:.4f}")


# ==========================
# Diagnostic: Calibration / Likelihood-ratio sanity check
# ==========================

p = all_scores
t = all_labels
w = all_weights


def makeCalibrationCurve(p, t, w, s_nbins=30):
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

    s_ref = s_bin_centers

    plt.figure(figsize=(7, 6))
    plt.errorbar(
        s_ref[valid],
        s_true[valid],
        yerr=s_err[valid],
        fmt="o",
        capsize=3,
        label="True weighted signal fraction",
    )

    plt.plot([0, 1], [0, 1], "--", label="Ideal: truth = bin center")
    plt.xlabel("NN output bin center")
    plt.ylabel("True signal probability")
    plt.title("Diagnostic: true signal fraction vs NN output bin center")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(f"diagnostic_calibration_{RUN_NAME}.pdf")
    plt.close()


makeCalibrationCurve(all_scores, all_labels, all_weights)


# ==========================
# Diagnostic: CARL reweighting closure
# ==========================

feature_name = full_dataset.feature_names[6]
feature_index = full_dataset.feature_names.index(feature_name)

# Collect validation features
all_features = []

with torch.no_grad():
    for x, y, w in val_loader:
        all_features.append(x.cpu().numpy())

all_features = np.concatenate(all_features, axis=0)

# Undo feature normalization for plotting
xobs = (
    all_features[:, feature_index]
    * full_dataset.std[feature_index]
    + full_dataset.mean[feature_index]
)

# Split into numerator / denominator samples
num_mask = all_labels == 1.0   # target / signal / numerator
den_mask = all_labels == 0.0   # reference / background / denominator

xobs_numerator = xobs[num_mask]
xobs_denominator = xobs[den_mask]

w_numerator = all_weights[num_mask]
w_denominator = all_weights[den_mask]

# Likelihood ratio from denominator predictions
eps = 1e-8
predictions_denominator = np.clip(all_scores[den_mask], eps, 1.0 - eps)
r_denominator = predictions_denominator / (1.0 - predictions_denominator)

# Histogram binning
n_xbins = 40
xmin = np.min(xobs)
xmax = np.max(xobs)
xbins = np.linspace(xmin, xmax, n_xbins + 1)
xcenters = 0.5 * (xbins[:-1] + xbins[1:])
xwidths = np.diff(xbins)

# Histograms
h_num_mc, _ = np.histogram(xobs_numerator, bins=xbins, weights=w_numerator)
h_denom, _ = np.histogram(xobs_denominator, bins=xbins, weights=w_denominator)
h_num_carl, _ = np.histogram(
    xobs_denominator,
    bins=xbins,
    weights=w_denominator * r_denominator,
)

# Variances from sum of squared weights
h_num_mc_var, _ = np.histogram(xobs_numerator, bins=xbins, weights=w_numerator ** 2)
h_denom_var, _ = np.histogram(xobs_denominator, bins=xbins, weights=w_denominator ** 2)
h_num_carl_var, _ = np.histogram(
    xobs_denominator,
    bins=xbins,
    weights=(w_denominator * r_denominator) ** 2,
)

# Convert to densities
h_num_mc_density = h_num_mc / xwidths
h_denom_density = h_denom / xwidths
h_num_carl_density = h_num_carl / xwidths

h_num_mc_density_err = np.sqrt(h_num_mc_var) / xwidths
h_denom_density_err = np.sqrt(h_denom_var) / xwidths
h_num_carl_density_err = np.sqrt(h_num_carl_var) / xwidths

# Ratios
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

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(
    3,
    1,
    gridspec_kw={"height_ratios": [2, 1, 1]},
    figsize=(5, 6),
    sharex=True,
)

# Top: densities
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

# Middle: ratio to denominator
ax2.stairs(
    ratio_num_carl_over_denom,
    xbins,
    color="cornflowerblue",
    linewidth=1.5,
)

ax2.stairs(
    ratio_num_mc_over_denom,
    xbins,
    color="blue",
    linestyle="--",
    linewidth=1.5,
)

ax2.stairs(
    np.ones_like(h_denom),
    xbins,
    color="black",
    linestyle="--",
    linewidth=1.5,
)

# Bottom: CARL / MC closure
ax3.stairs(
    np.ones_like(h_num_mc),
    xbins,
    color="blue",
    linestyle="--",
    linewidth=1.5,
)

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
ax3.set_ylabel("CARL / MC", fontsize=12)

ax3.set_xlabel(feature_name, fontsize=12)
ax3.set_xlim(xmin, xmax)

ax1.tick_params(labelsize=10)
ax2.tick_params(labelsize=10)
ax3.tick_params(labelsize=10)

plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.savefig(f"carl_reweight_{feature_name}_{RUN_NAME}.pdf", bbox_inches="tight")
plt.close()
