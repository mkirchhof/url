import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchmetrics.functional.classification import binary_auroc as auroc
import torch
from scipy.stats import spearmanr
from tueplots import bundles
bundles.icml2022(family="sans-serif", usetex=False, nrows=1)
plt.rcParams.update(bundles.icml2022())

BLUE = "#4878d0"
RED = "#d65f5f"
GREEN = "#6acc64"
ORANGE = "#ee854a"

unc = np.loadtxt("uncertainties_pretrained.csv", delimiter=",")
soft = np.loadtxt("all_soft.csv", delimiter=",")
target = np.loadtxt("all_targets.csv", delimiter=",")
target = target.astype("int")
entropy = np.loadtxt("gt_entropies.csv", delimiter=",")

cub = np.loadtxt(f"./deterioration_plots/baseline/uncertainties_repr_cub.csv")
cars = np.loadtxt(f"./deterioration_plots/baseline/uncertainties_repr_cars.csv")
sop = np.loadtxt(f"./deterioration_plots/baseline/uncertainties_repr_sop.csv")
unc_ood = np.concatenate([cub, cars, sop], axis=0)

unc_21k = torch.load("uncertainties_21k.pt").numpy()

# Compute different metrics of "original label is correct"
has_nonzero_mass = np.array([soft[idx, target] > 0 for idx, target in enumerate(target)])
is_maximum_label = np.argmax(soft, axis=1) == target
is_onehot_gt = entropy == 0
is_nan_entropy = np.isnan(entropy)

# Visualize
sns.kdeplot(unc[has_nonzero_mass], label='Correct labels', fill=True)
sns.kdeplot(unc[np.logical_not(has_nonzero_mass)], label='Hard wrong', fill=True)
plt.xlabel('Pretrained Uncertainty Value')
plt.ylabel('Density')
plt.legend()
plt.show()
plt.close()
print(f"AUROC hard wrong: {auroc(-torch.from_numpy(unc), torch.from_numpy(has_nonzero_mass).int()).item():.3f}")

sns.kdeplot(unc[is_maximum_label], label='Correct labels', fill=True)
sns.kdeplot(unc[np.logical_not(is_maximum_label)], label='Slightly wrong', fill=True)
plt.xlabel('Pretrained Uncertainty Value')
plt.ylabel('Density')
plt.legend()
plt.show()
plt.close()
print(f"AUROC slightly wrong: {auroc(-torch.from_numpy(unc), torch.from_numpy(is_maximum_label).int()).item():.3f}")

with plt.rc_context(bundles.icml2022(column="half")):
    fig, ax = plt.subplots()
    fig.set_figheight(1.8)
    sns.kdeplot(unc[is_onehot_gt], color=GREEN, label='Dirac label', fill=True)
    sns.kdeplot(unc[np.logical_not(is_onehot_gt)], color=ORANGE, label='Ambiguous label', fill=True)
    plt.xlabel('Pretrained Uncertainty Value')
    plt.ylabel('Density')
    plt.legend(framealpha=0)
    plt.savefig("dirac_vs_ambiguous_densities.pdf")
    plt.close()
    print(f"AUROC ambiguous label: {auroc(-torch.from_numpy(unc), torch.from_numpy(is_onehot_gt).int()).item():.3f}")

with plt.rc_context(bundles.icml2022(column="half")):
    fig, ax = plt.subplots()
    fig.set_figheight(1.8)
    sns.kdeplot(unc_21k, color=BLUE, label='ImageNet-21k-W', fill=True)
    sns.kdeplot(unc_ood, color=RED, label='CUB+CARS+SOP', fill=True)
    plt.xlabel('Pretrained Uncertainty Value')
    plt.ylabel('Density')
    plt.legend(framealpha=0)
    plt.savefig("id_vs_ood_densities.pdf")
    plt.close()
    auc = auroc(-torch.from_numpy(np.concatenate((unc_21k, unc_ood))),
                torch.from_numpy(np.concatenate((np.zeros_like(unc_21k, dtype=int), np.ones_like(unc_ood, dtype=int))))).item()
    print(f"AUROC ID-OOD: {auc:.3f}")

# Correlation with human uncertainties
is_not_nan = np.logical_not(np.isnan(entropy))
print(f'Correlation with humans: {spearmanr(unc[is_not_nan], entropy[is_not_nan])[0]:.3f}')
with plt.rc_context(bundles.neurips2023()):
    plt.scatter(unc[is_not_nan], entropy[is_not_nan])
    plt.savefig("human_vs_pretrained_scatter.png", dpi=300)
    plt.close()

print(1+1)