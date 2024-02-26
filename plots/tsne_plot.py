from utils import *
from sklearn.manifold import TSNE
import mplcursors

# Load data
repr = np.loadtxt("vtab_oxford_iiit_pet_features.csv", delimiter=",")
targets = np.loadtxt("vtab_oxford_iiit_pet_targets.csv", delimiter=",").astype(np.int64)
unc = np.loadtxt("vtab_oxford_iiit_pet_unc.csv", delimiter=",")
preds = np.loadtxt("vtab_oxford_iiit_pet_pred.csv", delimiter=",")
point_index = np.arange(len(targets))

# Reduce classes
if targets.max() > 9:
    classlist = {16: 0, 35: 1, 36: 2, 15: 3, 29: 4, 19: 5}
    keep_idxes = [idx for idx, target in enumerate(targets) if target in classlist]
    repr = repr[keep_idxes]
    targets = targets[keep_idxes]
    targets = np.array([classlist[target] for target in targets])
    unc = unc[keep_idxes]
    point_index = point_index[keep_idxes]

unc_normalized = (unc - np.min(unc)) / (np.max(unc) - np.min(unc))

# Make 2D scatterplot
with plt.rc_context(bundles.icml2022(column="half")):
    # Reduce dimensionality
    tsne = TSNE(n_components=2, random_state=1)
    repr_2d = tsne.fit_transform(repr)

    fig, ax = plt.subplots()
    fig.set_figheight(3.2)
    scatter = plt.scatter(repr_2d[:, 0], repr_2d[:, 1], c=targets, s=((unc_normalized + 0.2) * 3.5)**4.2,
                          alpha=np.maximum((1-unc_normalized)**1.4, 0.07), cmap="tab10", edgecolor='none')
    plt.axis('off')

    mplcursors.cursor(hover=True).connect(
        "add", lambda sel: sel.annotation.set_text(f'Index: {unc[sel.index]}')
    )
    plt.show()

    plt.savefig(f"tSNE.pdf")
