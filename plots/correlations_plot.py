from plots_constants import *
from scipy.stats import spearmanr
import pandas
import seaborn

corr = res[["best_eval_top1", "best_eval_r1", "best_eval_auroc_correct", "best_eval_croppedHasBiggerUnc",
            "best_test_avg_downstream_r1", "best_test_avg_downstream_auroc_correct", "best_test_avg_downstream_croppedHasBiggerUnc",
            "Sweep"]]
corr.rename(columns={"best_eval_top1": "Upstream Accuracy",
                     "best_eval_r1": "Upstream R@1",
                     "best_eval_auroc_correct": "Upstream R-AUROC",
                     "best_eval_croppedHasBiggerUnc": "Upstream Pct. Cropped Has Bigger Unc.",
                     "best_test_avg_downstream_r1": "Downstream R@1",
                     "best_test_avg_downstream_auroc_correct": "Downstream R-AUROC",
                     "best_test_avg_downstream_croppedHasBiggerUnc": "Downstream Pct. Cropped Has Bigger Unc."},
            inplace=True)

with plt.rc_context(bundles.neurips2023()):
    fig, ax = plt.subplots()
    fig.set_figheight(4.0)

    g = seaborn.pairplot(corr, diag_kind=None, hue="Sweep", palette=id_to_col, markers=id_to_marker,
                         plot_kws={"edgecolor":"none", "size":res['size']})
    g._legend.remove()

    plt.savefig("correlations.pdf")
    plt.close()
