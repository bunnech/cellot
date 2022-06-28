from pathlib import Path
import numpy as np
import pandas as pd
from absl import app, flags
from cellot.utils.evaluate import (
    compute_drug_signature_differences,
    compute_knn_enrichment,
    compute_mmd_df,
    load_conditions,
)
from cellot.utils import load_config
from umap import UMAP
from cellot.data.cell import read_single_anndata


FLAGS = flags.FLAGS
flags.DEFINE_boolean("predictions", True, "Run predictions.")
flags.DEFINE_string("outdir", "", "Path to outdir.")
flags.DEFINE_integer("n_markers", None, "Number of marker genes.")
flags.DEFINE_string("subset", None, "Name of obs entry to use as subset.")
flags.DEFINE_string("subset_name", None, "Name of subset.")

flags.DEFINE_enum(
    "setting", "iid", ["iid", "ood"], "Evaluate iid, ood or via combinations."
)

flags.DEFINE_enum(
    "where",
    "data_space",
    ["data_space", "latent_space"],
    "In which space to conduct analysis",
)

flags.DEFINE_multi_string("via", "", "Directory containing compositional map.")

flags.DEFINE_string("subname", "", "")


def main(argv):
    expdir = Path(FLAGS.outdir)
    setting = FLAGS.setting
    where = FLAGS.where
    subset = FLAGS.subset
    subset_name = FLAGS.subset_name

    if subset is None:
        outdir = expdir / f"evals_{setting}_{where}"
    else:
        assert subset is not None
        outdir = expdir / f"evals_{setting}_{where}_{subset}_{subset_name}"

    if len(FLAGS.subname) > 0:
        outdir = outdir / FLAGS.subname

    outdir.mkdir(exist_ok=True, parents=True)

    config = load_config(expdir / "config.yaml")
    if "ae_emb" in config.data:
        assert config.model.name == "cellot"
        config.data.ae_emb.path = str(expdir.parent / "model-scgen")
    cache = outdir / "imputed.h5ad"

    control, treated, imputed = load_conditions(expdir, where, setting)
    imputed.write(cache)
    imputed = imputed.to_df()

    if FLAGS.n_markers is not None:
        data = read_single_anndata(config, path=None)
        sel_mg = (
            data.varm[f"marker_genes-{config.data.condition}-rank"][config.data.target]
            .sort_values()
            .index[: FLAGS.n_markers]
        )

        control = control[sel_mg]
        treated = treated[sel_mg]
        imputed = imputed[sel_mg]

    if FLAGS.subset is not None:
        data = read_single_anndata(config, path=None)

        if subset != "time":
            control = control[data.obs[subset] == subset_name]
            imputed = imputed[data.obs[subset] == subset_name]
            treated = treated[data.obs[subset] == subset_name]
        else:
            treated = treated[data.obs[subset] == int(subset_name)]

    imputed.columns = imputed.columns.astype(str)
    treated.columns = treated.columns.astype(str)
    control.columns = control.columns.astype(str)

    assert imputed.columns.equals(treated.columns)

    l2ds = compute_drug_signature_differences(control, treated, imputed)
    knn, enrichment, joint = compute_knn_enrichment(imputed, treated, return_joint=True)
    mmddf = compute_mmd_df(imputed, treated, subsample=True, ncells=5000)

    l2ds.to_csv(outdir / "drug_signature_diff.csv")
    enrichment.to_csv(outdir / "knn_enrichment.csv")
    knn.to_csv(outdir / "knn_neighbors.csv")
    mmddf.to_csv(outdir / "mmd.csv")

    summary = pd.Series(
        {
            "l2DS": np.sqrt((l2ds**2).sum()),
            "enrichment-k50": enrichment.iloc[:, :50].values.mean(),
            "enrichment-k100": enrichment.iloc[:, :100].values.mean(),
            "mmd": mmddf["mmd"].mean(),
        }
    )
    summary.to_csv(outdir / "evals.csv", header=None)

    umap = pd.DataFrame(
        UMAP().fit_transform(joint), index=joint.index, columns=["UMAP1", "UMAP2"]
    )
    umap["is_imputed"] = umap.index.isin(imputed.index)
    umap.to_csv(outdir / "umap.csv")

    return


if __name__ == "__main__":
    app.run(main)
