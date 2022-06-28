from absl import app, flags
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import pandas as pd
from cellot.utils import viz
from cellot.utils import load_config
from cellot.data.cell import read_single_anndata
from cellot.utils.evaluate import load_conditions

FLAGS = flags.FLAGS
flags.DEFINE_string("evaldir", "", "Path to eval directory.")
flags.DEFINE_integer("n_markers", None, "Number of marker genes.")
flags.DEFINE_string("subset", None, "Name of obs entry to use as subset.")
flags.DEFINE_string("subset_name", None, "Name of subset.")
flags.DEFINE_bool("comparison_only", False, "Whether to only plot comparison.")

flags.DEFINE_enum(
    "setting", "iid", ["iid", "ood"], "Evaluate in i.i.d. or o.o.d. setting."
)

flags.DEFINE_enum(
    "where",
    "data_space",
    ["data_space", "latent_space"],
    "In which space to conduct analysis.",
)

flags.DEFINE_boolean("logscale", False, "Run marginals in logscale")


def load_single_dfs(expdir, setting="iid", where="data_space", n_markers=None):
    assert setting == "iid" or setting == "ood"

    control, treated, imputed = load_conditions(expdir, where, setting)
    imputed = imputed.to_df()
    config = load_config(expdir / "config.yaml")

    if n_markers is not None:
        data = read_single_anndata(config, path=None)
        sel_mg = (
            data.varm[f"marker_genes-{config.data.condition}-rank"][config.data.target]
            .sort_values()
            .index[:n_markers]
        )

        control = control[sel_mg]
        treated = treated[sel_mg]
        imputed = imputed[sel_mg]
    control.columns = control.columns.astype(str)
    treated.columns = treated.columns.astype(str)
    return control, treated, imputed


def load_single_umap(expdir, setting="iid", where="data_space", knn=None, k=50):
    assert setting == "iid" or setting == "ood"

    umaps = pd.read_csv(expdir / f"evals_{setting}_{where}" / "umap.csv", index_col=0)
    if knn is None:
        knn = pd.read_csv(
            expdir / f"evals_{setting}_{where}" / "knn_enrichment.csv", index_col=0
        )

    umaps["enrichment"] = knn.reindex(umaps.index).iloc[:, :k].mean(1)

    return umaps


def get_dfs(evaldir, config, setting, where, n_markers=None):
    dfs = dict()
    for model in config["marginals"]["models"]:
        dfs["control"], dfs["treated"], dfs[model] = load_single_dfs(
            evaldir / f"model-{model}", setting, where, n_markers
        )
    return dfs


def plot_marginals(config, dfs, outdir, logscale=False):
    fig, axes = viz.create_axes_grid(5, 50)
    for axl in axes:
        _, _, handles = viz.plot_marginals(
            dfs,
            axes=axl,
            features=[str(x) for x in dfs["control"].columns],
            qclip=0.01,
            colors=config["marginals"]["colors"],
            order=["cellot", "treated", "control", "scgen", "cae"],
        )

        for ax in axl:
            ax.set_title(ax.get_title())
            if logscale:
                ax.set_yscale("log")

        axes[0, -1].legend(handles=handles, bbox_to_anchor=(1, 1), loc="upper left")
    plt.tight_layout()
    if logscale:
        plt.savefig(outdir / "marginals-logscale.pdf", bbox_inches="tight")
    else:
        plt.savefig(outdir / "marginals.pdf", bbox_inches="tight")
    plt.close()


def plot_umaps(config, evaldir, outdir, setting="iid", where="data_space"):
    umaps = dict()
    models = config["umaps"]["models"]
    for model in models:
        umaps[model] = load_single_umap(evaldir / f"model-{model}", setting, where)
    viz.plot_umaps_binary(umaps)
    plt.tight_layout()
    plt.savefig(outdir / "umaps.pdf", bbox_inches="tight")
    plt.close()


def plot_knn_mmd(config, evaldir, outdir, setting="iid", where="data_space"):
    # compute knn enrichment
    knn = dict()

    for model in config["enrichment"]["models"]:
        path = (
            evaldir
            / f"model-{model}"
            / f"evals_{setting}_{where}"
            / "knn_enrichment.csv"
        )
        if not path.exists():
            continue
        knn[model] = pd.read_csv(path, index_col=0)

    enrichdf = pd.concat(
        {k: df.iloc[:, :50].mean(1).rename("k50") for k, df in knn.items()},
        names=["model"],
    ).reset_index(["model"])

    # compute mmd
    mmd = dict()
    for model in config["enrichment"]["models"]:
        path = evaldir / f"model-{model}" / f"evals_{setting}_{where}" / "mmd.csv"
        if not path.exists():
            continue
        mmd[model] = pd.read_csv(path, index_col=0)
    mmd = pd.concat(mmd, names=["model"]).reset_index(["model"])
    mmd = mmd[mmd["gamma"] <= 0.05].groupby(["model"])["mmd"].mean().reset_index()

    # save dataframes
    enrichdf.to_csv(outdir / "results_knn_enrich.csv")
    mmd.to_csv(outdir / "results_mmd.csv")

    # plot figures
    fig, axes = viz.create_axes_grid(2, 2, scale=2)
    axit = iter(axes.ravel())

    ax = next(axit)
    sns.boxplot(
        data=enrichdf,
        x="k50",
        y="model",
        order=["cellot", "identity", "random", "cae", "scgen"],
        showfliers=False,
        ax=ax,
    )
    ax.set_xlabel("k50 enrichment")

    ax = next(axit)
    sns.barplot(
        data=mmd,
        y="model",
        x="mmd",
        order=["cellot", "identity", "random", "cae", "scgen"],
        ax=ax,
    )

    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xscale("log")

    plt.savefig(outdir / "combined_metrics.pdf", bbox_inches="tight")
    plt.close()


def plot_iid_vs_ood(evaldir, outdir, setting, where, subset=None, subset_name=None):
    if "scrna-sciplex3" in str(evaldir):
        root = evaldir.parents[2]
        outdir = outdir.parents[2]
    else:
        root = evaldir.parents[1]
        outdir = outdir.parents[1]

    df = load_all_mmd(
        root, setting=setting, where=where, subset=subset, subset_name=subset_name
    )
    if df["target"].nunique() > 1:
        g = sns.FacetGrid(data=df, row="holdout", col="target", sharey=False)
    else:
        g = sns.FacetGrid(data=df, col="holdout", col_wrap=4, sharey=False)

    g.map_dataframe(
        sns.stripplot,
        x="model",
        y="mmd",
        hue="mode",
        order=["cellot", "cae", "scgen"],
        hue_order=["iid", "ood"],
        palette="tab10",
    )

    g.set(yscale="log")
    g.add_legend()

    if "scrna-sciplex3" in str(evaldir):
        g.set_titles(row_template="{row_name}", col_template="{col_name}")

    plt.tight_layout()

    if subset:
        filename = f"iid_vs_ood_{subset}_{subset_name}.pdf"
    else:
        filename = "iid_vs_ood.pdf"
    plt.savefig(outdir / filename, bbox_inches="tight")
    print(f"Wrote {outdir}/{filename}")
    return


def load_evals(path):
    config = load_config(path.parents[1] / "config.yaml")

    evals = pd.read_csv(path, header=None, index_col=0, squeeze=True).to_dict()

    evals.update(
        {
            "holdout": config.datasplit.holdout,
            "setting": config.datasplit.mode,
            "model": config.model.name,
            "path": str(path),
            "target": config.data.target,
        }
    )

    return evals


def load_all_mmd(root, setting, where, subset, subset_name):
    def iterate():
        for path in root.glob("**/config.yaml"):
            config = load_config(path)
            expdir = path.parent
            if subset:
                assert subset_name is not None
                mmd = pd.read_csv(
                    expdir
                    / f"evals_{setting}_{where}_{subset}_{subset_name}"
                    / "mmd.csv",
                    index_col=0,
                )
            else:
                mmd = pd.read_csv(
                    expdir / f"evals_{setting}_{where}" / "mmd.csv", index_col=0
                )
            mmd = mmd.loc[mmd["gamma"] <= 0.05, "mmd"].mean()

            yield config.model.name, config.datasplit.holdout, config.datasplit.mode, config.data.target, mmd

    return pd.DataFrame(
        iterate(), columns=["model", "holdout", "mode", "target", "mmd"]
    )


def main(argv):
    config_plotting = """

        marginals:
            models:
                - cellot
                - cae
                - scgen

            order: [cellot, treated, control, cae, scgen]
            colors:
                cellot: '#F2545B'
                treated: '#114083'
                control: '#A7BED3'
                cae: '#9A8F97'
                scgen: '#C3BABA'

        umaps:
            models:
                - cellot
                - cae
                - scgen
                - random
                - identity

        enrichment:
            models:
                - cellot
                - cae
                - scgen
                - random
                - identity

            k: 50

        mmd:
            models:
                - cellot
                - cae
                - scgen
                - random
                - identity

            gamma: []
    """

    config_plotting = yaml.load(config_plotting, yaml.UnsafeLoader)
    evaldir = Path(FLAGS.evaldir)
    setting = FLAGS.setting
    where = FLAGS.where
    n_markers = FLAGS.n_markers
    subset = FLAGS.subset
    subset_name = FLAGS.subset_name

    outdir = Path(str(evaldir).replace("results", "figures")) / where
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if not FLAGS.comparison_only:
        dfs = get_dfs(evaldir, config_plotting, setting, where, n_markers)
        print("Plotting marginals.")
        plot_marginals(config_plotting, dfs, outdir, logscale=FLAGS.logscale)

        print("Plotting UMAPS.")
        plot_umaps(config_plotting, evaldir, outdir, setting, where)

        print("Plotting kNN Enrichment and MMD Evaluation.")
        plot_knn_mmd(config_plotting, evaldir, outdir, setting, where)

    if setting == "ood":
        sns.set_context("talk")
        plot_iid_vs_ood(
            evaldir,
            outdir,
            setting=setting,
            where=where,
            subset=subset,
            subset_name=subset_name,
        )


if __name__ == "__main__":
    sns.set_context("poster")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "path"

    app.run(main)
