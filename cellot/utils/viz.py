import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from math import ceil
import pandas as pd
import seaborn as sns


def scale_figsize(nrows, ncols, scale=1):
    figsize = plt.rcParams["figure.figsize"]
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    c, r = scale

    return (c * ncols * figsize[0], r * nrows * figsize[1])


def clean_axes_grid(axes, xlabel=None, ylabel=None):
    for ax in axes.ravel():
        if not ax.has_data():
            ax.axis("off")

    if xlabel is not None:
        for ax in axes.ravel():
            ax.set_xlabel("")

        xaxes = axes[-1] if axes.ndim > 1 else axes
        for cidx, ax in enumerate(xaxes):
            if ax.has_data():
                ax.set_xlabel(xlabel)
            elif axes.ndim > 1 and axes.shape[0] > 2:
                axes[-2, cidx].set_xlabel(xlabel)

    if ylabel is not None:
        for ax in axes.ravel():
            ax.set_ylabel("")
        yaxes = axes[:, 0] if axes.ndim > 1 else axes
        for ax in yaxes.ravel():
            ax.set_ylabel(ylabel)

    return


def create_axes_grid(nitems, ncols, figsize=None, scale=1, **kwargs):
    nrows = ceil(nitems / ncols)
    if figsize is None:
        figsize = scale_figsize(nrows, ncols, scale)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

    if ncols == 1 and nrows == 1:
        axes = np.array([[axes]])
    elif ncols == 1:
        axes = axes[:, np.newaxis]
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    assert axes.ndim == 2

    return fig, axes


def modify_legend(ax, **kwargs):
    """Modify a legend object after it has been created."""
    legend = ax.get_legend()
    if legend is None:
        return

    handles = legend.legendHandles
    ax.legend(handles=handles, **kwargs)
    return


def legend_handle(label, color, cmap="tab10", marker=None, markersize=10, **kwargs):

    cmap = plt.get_cmap(cmap)
    if isinstance(color, int):
        color = cmap(color)

    if marker is None:
        return Patch(color=color, label=label, **kwargs)

    return Line2D(
        [0],
        [0],
        color="white",
        marker=marker,
        markersize=markersize,
        markerfacecolor=color,
        label=label,
        **kwargs,
    )


def legend_from_lut(lut, order=None, **kwargs):
    if order is None:
        order = lut.keys()

    return [legend_handle(key, lut[key], **kwargs) for key in order]


def create_map(mappable):
    if mappable is None:
        return lambda x: x

    if isinstance(mappable, dict):
        return lambda x: mappable.get(x, x)

    return mappable


def plot_marginals(
    dfs,
    features=None,
    qclip=None,
    ncols=5,
    colors=None,
    order=None,
    handle_pprint=None,
    title_pprint=None,
    axes=None,
    **kwargs,
):

    if isinstance(dfs, dict):
        df = (
            pd.concat(dfs, names=["groupby"])
            .reset_index("groupby")
            .reset_index(drop=True)
        )

        groupby = df.pop("groupby")

    else:
        df = dfs
        groupby = None

    if features is None:
        features = df.columns

    if axes is None:
        fig, axes = create_axes_grid(len(features), ncols)
    else:
        fig = None

    if order is None and groupby is not None:
        order = sorted(dfs.keys())

    if colors is None and groupby is not None:
        cmap = plt.get_cmap("tab10")
        colors = {k: cmap(idx) for idx, k in enumerate(order)}

    if isinstance(qclip, float):
        qclip = (qclip, 1 - qclip)

    map_title = create_map(title_pprint)
    map_handle = create_map(handle_pprint)

    for ax, feat in zip(axes.ravel(), features):

        if qclip is not None:
            lb, ub = df[feat].quantile(qclip)
            df[feat] = df[feat].clip(lb, ub)

        sns.kdeplot(
            data=df,
            x=feat,
            hue=groupby,
            common_norm=False,
            palette=colors,
            hue_order=order,
            **kwargs,
            ax=ax,
            legend=False,
        )

        ax.set_title(map_title(feat))
        ax.set_ylabel("")
        ax.set_xlabel("")

    if groupby is not None:
        handles = [legend_handle(map_handle(k), colors[k]) for k in order]

        return fig, axes, handles

    return fig, axes


def pretty_print_feature(name):
    lut = {
        "TotProtein": "Total protein",
        "ClCasp3": "Cl. Caspase 3",
        "MelA": "Melanoma marker",
    }

    if name.startswith("morphology-"):
        measurement, cell_part, feature = name.split("-")
        statistic = None

        feature = feature.replace("_", " ").capitalize()

    elif name.startswith("intensity-"):
        measurement, cell_part, feature, statistic = name.split("-")
        feature = lut.get(feature, feature)
        feature = feature + " Intensity"

    else:
        raise ValueError
    cell_part = cell_part.title()
    pp = f"{feature} ({cell_part})"
    return pp


def plot_corrs(
    corrs, colors=None, order=None, title_pretty_print=None, xtick_pretty_print=None
):

    map_title = create_map(title_pretty_print)
    map_xtick = create_map(xtick_pretty_print)

    g = sns.FacetGrid(data=corrs, col="drug", height=5)
    g.map(
        sns.violinplot,
        "model",
        "value",
        scale="width",
        cut=0,
        palette=dict(colors),
        order=[x for x in order if x in corrs["model"].unique()],
    )

    g.set_titles("{col_name}")
    for ax in g.axes.ravel():
        ax.set_title(map_title(ax.get_title()))
        ax.set_xlabel("")
        ax.set_xticklabels([map_xtick(x.get_text()) for x in ax.get_xticklabels()])
    g.axes.ravel()[0].set_ylabel("Spearman R")
    return g


def plot_umaps_binary(
    umaps, s=2.5, alpha=0.5, order=None, title_pretty_print=None, **kwargs
):

    map_title = create_map(title_pretty_print)

    if order is None:
        order = sorted(umaps.keys())
    order = [x for x in order if x in umaps.keys()]

    cmap = {True: plt.get_cmap("tab10")(0), False: "lightgrey"}

    fig, axes = create_axes_grid(len(umaps), len(umaps))
    for ax, key in zip(axes.ravel(), order):
        df = umaps[key].sample(frac=1)

        ax.scatter(
            x=df["UMAP1"],
            y=df["UMAP2"],
            s=s,
            c=df["is_pushfwd"].map(cmap),
            alpha=alpha,
            **kwargs,
        )

        ax.set_title(key)

    for ax in axes.ravel():
        ax.set_title(map_title(ax.get_title()))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP1")

    axes.ravel()[0].set_ylabel("UMAP2")

    return axes


def plot_umaps(umaps, s=2.5, order=None, title_pretty_print=None, **kwargs):

    map_title = create_map(title_pretty_print)

    if order is None:
        order = sorted(umaps.keys())
    order = [x for x in order if x in umaps.keys()]

    fig, axes = create_axes_grid(len(umaps) + 1, len(umaps) + 1)
    for ax, key in zip(axes.ravel(), order):
        df = umaps[key]
        predictions = df["is_pushfwd"]

        ax.scatter(
            x=df.loc[~predictions, "UMAP1"],
            y=df.loc[~predictions, "UMAP2"],
            c="grey",
            alpha=0.25,
            s=s,
            **kwargs,
        )

        g = ax.scatter(
            x=df.loc[predictions, "UMAP1"],
            y=df.loc[predictions, "UMAP2"],
            c=df.loc[predictions, "enrichment"],
            cmap="RdBu",
            vmin=0,
            vmax=1,
            alpha=0.75,
            s=s,
            **kwargs,
        )
        ax.set_title(key)

    cax = axes.ravel()[-1]
    g = ScalarMappable(cmap="RdBu")
    cbar = fig.colorbar(g, ax=cax, orientation="horizontal")
    cbar.ax.xaxis.set_ticks_position("bottom")
    cbar.ax.set_title("Fraction of kNN\nin prediction set ")

    cax.legend(
        handles=legend_from_lut(
            {"Perturbed": "grey", "Predicted": plt.get_cmap("tab10")(0)},
            marker="o",
            markersize=15,
        ),
        loc="lower center",
        title="Cell set",
    )

    for ax in axes.ravel()[:-1]:
        ax.set_title(map_title(ax.get_title()))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP1")

    axes.ravel()[0].set_ylabel("UMAP2")
    axes.ravel()[-1].axis("off")

    return axes
