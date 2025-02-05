# import matplotlib.colors as mpl_colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bayesflow.computational_utilities import simultaneous_ecdf_bands
from matplotlib.collections import LineCollection

# desaturation_factor = 0.1

# color_dict = {
#     r"$\delta$": mpl_colors.to_hex(
#         sns.desaturate(mpl_colors.to_rgb("#7fc97f"), desaturation_factor)
#     ),
#     r"$\alpha$": mpl_colors.to_hex(
#         sns.desaturate(mpl_colors.to_rgb("#beaed4"), desaturation_factor)
#     ),
#     r"$\beta$": mpl_colors.to_hex(
#         sns.desaturate(mpl_colors.to_rgb("#fdc086"), desaturation_factor)
#     ),
#     r"$\mu_{(e)}$": mpl_colors.to_hex(
#         sns.desaturate(mpl_colors.to_rgb("#f0027f"), 1.0)
#     ),
#     r"$\tau_{(m)}$": mpl_colors.to_hex(
#         sns.desaturate(mpl_colors.to_rgb("#ffff99"), desaturation_factor)
#     ),
#     r"$\sigma$": mpl_colors.to_hex(
#         sns.desaturate(mpl_colors.to_rgb("#666666"), desaturation_factor)
#     ),
#     r"$s_{(\tau)}$": mpl_colors.to_hex(
#         sns.desaturate(mpl_colors.to_rgb("#bf5b17"), desaturation_factor)
#     ),
#     r"$\gamma$": mpl_colors.to_hex(sns.desaturate(mpl_colors.to_rgb("#386cb0"), 1.0)),
#     r"$\lambda$": mpl_colors.to_hex(sns.desaturate(mpl_colors.to_rgb("#386cb0"), 1.0)),
# }

# color_dict = {
#     r"$\delta$": "#beaed4",
#     r"$\alpha$": "#7fc97f",
#     r"$\beta$": "#fdc086",
#     r"$\mu_{(e)}$": "#f0027f",
#     r"$\tau_{(m)}$": "#ffff99",
#     r"$\sigma$": "#666666",
#     r"$s_{(\tau)}$": "#bf5b17",
#     r"$\gamma$": "#386cb0",
#     r"$\lambda$": "#386cb0",
# }

gray_color = "#666666"

color_dict = {
    r"$\delta$": "#beaed4",
    r"$\alpha$": "#bf5b17",
    r"$\beta$": "#fdc086",
    r"$\mu_{(e)}$": "#f0027f",
    r"$\tau_{(m)}$": "#ffff99",
    r"$\sigma$": "#666666",
    r"$s_{(\tau)}$": "#7fc97f",
    r"$\gamma$": "#386cb0",
    r"$\lambda$": "#386cb0",
}

highlight_keys = [
    r"$\alpha$",
    r"$\mu_{(e)}$",
    r"$\gamma$",
    r"$\lambda$",
    r"$s_{(\tau)}$",
]

for param in color_dict.keys():
    if param not in highlight_keys:
        color_dict[param] = gray_color


def plot_sbc_ecdf(
    post_samples,
    prior_samples,
    difference=False,
    stacked=False,
    fig_size=None,
    param_names=None,
    label_fontsize=24,
    legend_fontsize=24,
    title_fontsize=16,
    rank_ecdf_colors=["#009900", "#990000"],
    fill_color="grey",
    legend_spacing=0.8,
    num_legend_cols=2,
    plot_legend=True,
    legend_position="lower center",
    legend_bbox_to_anchor=None,
    ylim=None,
    title=None,
    **kwargs,
):
    # Compute fractional ranks (using broadcasting)
    ranks = (
        np.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1)
        / post_samples.shape[1]
    )

    # Prepare figure
    f, ax = plt.subplots(1, 1, figsize=fig_size)

    patches = [None] * ranks.shape[-1]
    # Plot individual ecdf of parameters
    for j in range(ranks.shape[-1]):
        ecdf_single = np.sort(ranks[:, j])
        xx = ecdf_single
        yy = np.arange(1, xx.shape[-1] + 1) / float(xx.shape[-1])

        # Difference, if specified
        if difference:
            yy -= xx

        ax.plot(
            xx,
            yy,
            color=rank_ecdf_colors[j],
            alpha=0.95,
            **kwargs.pop("ecdf_line_kwargs", {}),
        )
        patches[j] = mpatches.Rectangle(
            [0, 0], 0.1, 0.1, facecolor=rank_ecdf_colors[j], label=param_names[j]
        )

    # Compute uniform ECDF and bands
    alpha, z, L, H = simultaneous_ecdf_bands(
        post_samples.shape[0], **kwargs.pop("ecdf_bands_kwargs", {})
    )
    if ylim is not None:
        ax.set_ylim(ylim)

    # Difference, if specified
    if difference:
        L -= z
        H -= z

    # Add simultaneous bounds
    titles = [title]
    axes = [ax]

    ax.set_box_aspect(1)
    for _ax, title in zip(axes, titles):
        _ax.fill_between(z, L, H, color=fill_color, alpha=0.2)

        # Prettify plot
        sns.despine(ax=_ax)
        _ax.grid(alpha=0.35)
        if plot_legend:
            _ax.legend(
                fontsize=legend_fontsize,
                loc=legend_position,
                ncol=num_legend_cols,
                handles=patches,
                columnspacing=legend_spacing,
                handletextpad=0.3,
                handlelength=1,
                handleheight=1,
                bbox_to_anchor=legend_bbox_to_anchor,
            )
        _ax.set_xlabel("Fractional rank statistic", fontsize=label_fontsize)
        if difference:
            ylab = "ECDF difference"
        else:
            ylab = "ECDF"
        _ax.set_ylabel(ylab, fontsize=label_fontsize)
        _ax.set_title(title, fontsize=title_fontsize)
        # remove x-axis label and ticks
        _ax.set_xticks([])
        _ax.set_xticklabels([])
        _ax.set_yticks([])
        _ax.set_yticklabels([])

    f.tight_layout()
    return f


def plot_sbc_ecdf_axis(
    axis,
    post_samples,
    prior_samples,
    difference=False,
    stacked=False,
    fig_size=None,
    label_fontsize=24,
    title_fontsize=16,
    rank_ecdf_colors=["#009900", "#990000"],
    fill_color="grey",
    ylim=None,
    **kwargs,
):
    ax = axis
    # Store reference to number of parameters

    # Compute fractional ranks (using broadcasting)
    ranks = (
        np.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1)
        / post_samples.shape[1]
    )

    # Plot individual ecdf of parameters
    for j in range(ranks.shape[-1]):
        ecdf_single = np.sort(ranks[:, j])
        xx = ecdf_single
        yy = np.arange(1, xx.shape[-1] + 1) / float(xx.shape[-1])

        # Difference, if specified
        if difference:
            yy -= xx

        ax.plot(
            xx,
            yy,
            color=rank_ecdf_colors[j],
            alpha=0.95,
            **kwargs.pop("ecdf_line_kwargs", {}),
        )

    # Compute uniform ECDF and bands
    alpha, z, L, H = simultaneous_ecdf_bands(
        post_samples.shape[0], **kwargs.pop("ecdf_bands_kwargs", {})
    )
    if ylim is not None:
        ax.set_ylim(ylim)

    # Difference, if specified
    if difference:
        L -= z
        H -= z

    # Add simultaneous bounds
    titles = [None]
    axes = [ax]

    ax.set_box_aspect(1)
    for _ax, title in zip(axes, titles):
        _ax.fill_between(z, L, H, color=fill_color, alpha=0.2)

        # Prettify plot
        sns.despine(ax=_ax)
        # _ax.grid(alpha=0.35)


def plot_true_vs_ci(
    true,
    posterior,
    names,
    limits=(None, None),
    point_colors=None,
    line_color="#AAAAAA",
):
    if point_colors is None:
        point_colors = ["#012F47"] * len(names)

    num_rows, num_cols = 1, true.shape[-1]
    figsize_multiplier = 0.8
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(
            (1.5 + 2 * num_cols) * figsize_multiplier,
            (1.5 + 2 * num_rows) * figsize_multiplier,
        ),
    )

    for i, ax in enumerate(axes):
        ax.scatter(
            true[:, i],
            np.mean(posterior[:, :, i], axis=1),
            color=point_colors[i],
            s=3,
            zorder=2,
        )

        lower_q = np.quantile(posterior[:, :, i], 0.025, axis=1)
        upper_q = np.quantile(posterior[:, :, i], 0.975, axis=1)

        # add errorbars
        lines = [[(x, lower_q[j]), (x, upper_q[j])] for j, x in enumerate(true[:, i])]
        lc = LineCollection(lines, color=line_color, linewidth=0.5, zorder=1)
        ax.add_collection(lc)

        ax.plot(
            [0, 1],
            [0, 1],
            color="black",
            linestyle="dashed",
            transform=ax.transAxes,
            zorder=0,
        )
        ax.set_title(names[i], fontsize="xx-large")
        ax.set_aspect("equal")

        # Set the same x and y limits for all plots
        if limits == (None, None):
            ax.set_xlim([min(true[:, i]), max(true[:, i])])
            ax.set_ylim([min(true[:, i]), max(true[:, i])])
        else:
            ax.set_xlim(limits)
            ax.set_ylim(limits)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_xticklabels([])
        ax.set_yticklabels([])

    axes[0].set_ylabel("Estimate", fontsize="xx-large")
    fig.text(0.5, 0.1, "Ground truth", ha="center", fontsize="xx-large")

    plt.tight_layout()

    sns.despine()

    return fig
