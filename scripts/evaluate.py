from pathlib import Path
import numpy as np
import pandas as pd
from absl import app, flags
from cellot.utils.evaluate import (
    load_conditions,
    compute_knn_enrichment,
)
from cellot.losses.mmd import mmd_distance
from cellot.utils import load_config
from cellot.data.cell import read_single_anndata


FLAGS = flags.FLAGS
flags.DEFINE_boolean('predictions', True, 'Run predictions.')
flags.DEFINE_boolean('debug', False, 'run in debug mode')
flags.DEFINE_string('outdir', '', 'Path to outdir.')
flags.DEFINE_string('n_markers', None, 'comma seperated list of integers')
flags.DEFINE_string(
        'n_cells', '100,250,500,1000,1500',
        'comma seperated list of integers')

flags.DEFINE_integer('n_reps', 10, 'number of evaluation repetitions')
flags.DEFINE_string('embedding', None, 'specify embedding context')
flags.DEFINE_string('evalprefix', None, 'override default prefix')

flags.DEFINE_enum(
    'setting', 'iid', ['iid', 'ood'], 'Evaluate iid, ood or via combinations.'
)

flags.DEFINE_enum(
    'where',
    'data_space',
    ['data_space', 'latent_space'],
    'In which space to conduct analysis',
)

flags.DEFINE_multi_string('via', '', 'Directory containing compositional map.')

flags.DEFINE_string('subname', '', '')


def compute_mmd_loss(lhs, rhs, gammas):
    return np.mean([mmd_distance(lhs, rhs, g) for g in gammas])


def compute_pairwise_corrs(df):
    corr = df.corr().rename_axis(index='lhs', columns='rhs')
    return (
        corr
        .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .set_index(['lhs', 'rhs'])
        .squeeze()
        .rename()
    )


def compute_evaluations(iterator):
    gammas = np.logspace(1, -3, num=50)
    for ncells, nfeatures, treated, imputed in iterator:
        mut, mui = treated.mean(0), imputed.mean(0)
        stdt, stdi = treated.std(0), imputed.std(0)
        pwct = compute_pairwise_corrs(treated)
        pwci = compute_pairwise_corrs(imputed)

        yield ncells, nfeatures, 'l2-means', np.linalg.norm(mut - mui)
        yield ncells, nfeatures, 'l2-stds', np.linalg.norm(stdt - stdi)
        yield ncells, nfeatures, 'r2-means', pd.Series.corr(mut, mui)
        yield ncells, nfeatures, 'r2-stds', pd.Series.corr(stdt, stdi)
        yield ncells, nfeatures, 'r2-pairwise_feat_corrs', pd.Series.corr(pwct, pwci)
        yield ncells, nfeatures, 'l2-pairwise_feat_corrs', np.linalg.norm(pwct - pwci)

        if treated.shape[1] < 1000:
            mmd = compute_mmd_loss(treated, imputed, gammas=gammas)
            yield ncells, nfeatures, 'mmd', mmd

            knn, enrichment = compute_knn_enrichment(imputed, treated)
            k50 = enrichment.iloc[:, :50].values.mean()
            k100 = enrichment.iloc[:, :100].values.mean()

            yield ncells, nfeatures, 'enrichment-k50', k50
            yield ncells, nfeatures, 'enrichment-k100', k100


def main(argv):
    expdir = Path(FLAGS.outdir)
    setting = FLAGS.setting
    where = FLAGS.where
    embedding = FLAGS.embedding
    prefix = FLAGS.evalprefix
    n_reps = FLAGS.n_reps

    if (embedding is None) or len(embedding) == 0:
        embedding = None

    if FLAGS.n_markers is None:
        n_markers = None
    else:
        n_markers = FLAGS.n_markers.split(',')
    all_ncells = [int(x) for x in FLAGS.n_cells.split(',')]

    if prefix is None:
        prefix = f'evals_{setting}_{where}'
    outdir = expdir / prefix

    outdir.mkdir(exist_ok=True, parents=True)

    def iterate_feature_slices():

        assert (expdir / 'config.yaml').exists()
        config = load_config(expdir / 'config.yaml')
        if 'ae_emb' in config.data:
            assert config.model.name == 'cellot'
            config.data.ae_emb.path = str(expdir.parent / 'model-scgen')
        cache = outdir / 'imputed.h5ad'

        _, treateddf, imputed = load_conditions(
                expdir, where, setting, embedding=embedding)

        imputed.write(cache)
        imputeddf = imputed.to_df()

        imputeddf.columns = imputeddf.columns.astype(str)
        treateddf.columns = treateddf.columns.astype(str)

        assert imputeddf.columns.equals(treateddf.columns)

        def load_markers():
            data = read_single_anndata(config, path=None)
            key = f'marker_genes-{config.data.condition}-rank'

            # rebuttal preprocessing stored marker genes using
            # a generic marker_genes-condition-rank key
            # instead of e.g. marker_genes-drug-rank
            # let's just patch that here:
            if key not in data.varm:
                key = 'marker_genes-condition-rank'
                print('WARNING: using generic condition marker genes')

            sel_mg = (
                data.varm[key][config.data.target]
                .sort_values()
                .index
            )
            return sel_mg

        if n_markers is not None:
            markers = load_markers()
            for k in n_markers:
                if k != 'all':
                    feats = markers[:int(k)]
                else:
                    feats = list(markers)

                for ncells in all_ncells:
                    if ncells > min(len(treateddf), len(imputeddf)):
                        break
                    for r in range(n_reps):
                        trt = treateddf[feats].sample(ncells)
                        imp = imputeddf[feats].sample(ncells)
                        yield ncells, k, trt, imp

        else:
            for ncells in all_ncells:
                if ncells > min(len(treateddf), len(imputeddf)):
                    break
                for r in range(n_reps):
                    trt = treateddf.sample(ncells)
                    imp = imputeddf.sample(ncells)
                    yield ncells, 'all', trt, imp

    evals = pd.DataFrame(
            compute_evaluations(iterate_feature_slices()),
            columns=['ncells', 'nfeatures', 'metric', 'value']
            )
    evals.to_csv(outdir / 'evals.csv', index=None)

    return


if __name__ == '__main__':
    app.run(main)
