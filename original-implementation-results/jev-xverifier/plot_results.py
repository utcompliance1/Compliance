"""Presentation-ready figures from measured, saved evaluation artifacts only."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
OUT.mkdir(exist_ok=True)
S = json.loads((HERE / "summary.json").read_text())
E = S["experiments"]
SWEEP = pd.read_csv(HERE / "confidence_sweep.csv")
COLORS = {"gpt": "#3867D6", "jev": "#008F83", "native_jev": "#D78324"}
INK = "#172B4D"
MUTED = "#52657C"
DATASETS = [("uncleaned", "Unfiltered"), ("cleaned", "Filtered")]
SERIES = [("gpt", "xverifier", "GPT-5 + xverifier", COLORS["gpt"], "o"),
          ("jev", "xverifier", "Jev + xverifier", COLORS["jev"], "s"),
          ("jev", "native_jev", "Jev's own probability", COLORS["native_jev"], "^")]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 16,
    "axes.titleweight": "bold", "axes.labelsize": 12, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.spines.bottom": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "grid.color": "#E7ECF2", "grid.linewidth": .8, "axes.axisbelow": True,
    "svg.fonttype": "none",
})


def percent(ax, both=False):
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    if both:
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y")


def title(fig, heading, subtitle):
    fig.suptitle(heading, x=.065, y=.97, ha="left", fontsize=23, fontweight="bold")
    fig.text(.065, .917, subtitle, color=MUTED, fontsize=11.5)


def footer(fig, text):
    fig.text(.065, .025, text, color=MUTED, fontsize=10, va="bottom")


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def accuracy(ax, dataset, labels=True):
    for i, judge in enumerate(("gpt", "jev")):
        data = E[f"{judge}-{dataset}"]
        base, after = data["base_accuracy_mean"], data["verifier_accuracy_mean"]
        color = COLORS[judge]
        x = np.array([i-.16, i+.16])
        ax.bar(x[0], base, width=.27, color=color, alpha=.35)
        ax.bar(x[1], after, width=.27, color=color)
        ax.errorbar(x, [base, after], yerr=[data["base_accuracy_sd"], data["verifier_accuracy_sd"]],
                    fmt="none", ecolor=INK, capsize=4, linewidth=1.2)
        for xx, yy, offset in zip(x, (base, after), (.065, .115)):
            ax.text(xx, yy+offset, f"{yy:.1%}", ha="center", fontsize=12, fontweight="bold", zorder=5)
    ax.set_xticks([0, 1], [f"GPT-5\n{E[f'gpt-{dataset}']['accuracy_delta_pp']:+.2f} pp", f"Jev 1.13\n{E[f'jev-{dataset}']['accuracy_delta_pp']:+.2f} pp"])
    ax.set_ylim(0, 1)
    ax.set_xlim(-.6, 1.6)
    percent(ax)
    if labels:
        ax.set_ylabel("Accuracy on all evaluation cases")


fig, axes = plt.subplots(1, 2, figsize=(13.3, 7.5))
title(fig, "Does xverifier improve the model's answers?", "Matched cases and identical evaluation splits for GPT-5 and Jev • five seeds")
for ax, (dataset, label) in zip(axes, DATASETS):
    ax.set_title(f"{label}  ·  {E[f'jev-{dataset}']['input_rows']:,} pilot cases", loc="left", pad=16)
    accuracy(ax, dataset)
    ax.text(.02, .92, "Light = model alone    Solid = with xverifier", transform=ax.transAxes, fontsize=10, color=MUTED)
footer(fig, "Bars = mean; whiskers = SD across seeds, not confidence intervals. Pilot size includes verifier training/calibration; only held-out slices are scored.\nLimited nonrandom sample from source train. Different filtered/unfiltered draws; no causal cleaning claim.")
fig.subplots_adjust(top=.79, bottom=.19, left=.08, right=.97, wspace=.26)
save(fig, "01_accuracy_comparison")


def curve(dataset, judge, method):
    d = SWEEP[(SWEEP.dataset == dataset) & (SWEEP.judge == judge) & (SWEEP.method == method)].copy()
    d["eligible_accuracy"] = d.accuracy.where(d.decided >= 20)
    grouped = d.groupby("threshold").agg(coverage=("coverage", "mean"), accuracy=("eligible_accuracy", "mean"),
                                       accuracy_sd=("eligible_accuracy", "std"), seeds=("eligible_accuracy", "count"))
    return grouped


fig, axes = plt.subplots(2, 2, figsize=(13.3, 9))
title(fig, "Higher confidence: fewer automatic decisions", "Cutoff sweep on held-out cases • diagnostic only; these are not certified operating points")
for col, (dataset, label) in enumerate(DATASETS):
    for judge, method, name, color, marker in SERIES:
        d = curve(dataset, judge, method)
        x = d.index.to_numpy()
        axes[0, col].plot(x, d.coverage, color=color, lw=2.4, label=name)
        axes[1, col].plot(x, d.accuracy, color=color, lw=2.4)
    axes[0, col].set_title(label, loc="left", pad=12)
    for row in (0, 1):
        ax = axes[row, col]
        ax.set_xlim(.5, 1)
        ax.set_ylim(0, 1.02)
        percent(ax, both=True)
        ax.set_xlabel("Confidence cutoff")
    axes[0, col].set_ylabel("Automatic coverage")
    axes[1, col].set_ylabel("Accuracy on handled cases")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.51, .86), ncol=3, frameon=False, fontsize=10.5)
footer(fig, "Coverage averages all five seeds. Accuracy uses only seeds with ≥20 automatic cases at that cutoff; curves can end when too few remain.\nJev confidence = probability of its chosen label; xverifier confidence = calibrated probability that its kept/flipped answer is correct.")
fig.subplots_adjust(top=.75, bottom=.15, left=.085, right=.97, wspace=.28, hspace=.4)
save(fig, "02_confidence_coverage_accuracy")


def operating(judge, dataset):
    points = E[f"{judge}-{dataset}"]["operating_points"]
    levels = [.25, .20, .15, .10, .05]
    return (np.array([1-l for l in levels]),
            np.array([points[f"{l:.2f}"]["coverage_mean"] for l in levels]),
            np.array([1-points[f"{l:.2f}"]["error_rate_mean"] if points[f"{l:.2f}"]["error_rate_mean"] is not None else np.nan for l in levels]),
            [points[f"{l:.2f}"]["n_certified"] for l in levels])


fig, axes = plt.subplots(2, 2, figsize=(13.3, 9))
title(fig, "Higher accuracy goals mean less automation", "Cutoffs selected on calibration data by the original xverifier implementation • statistical confidence: 95%")
for col, (dataset, label) in enumerate(DATASETS):
    for judge, offset, marker in (("gpt", -.003, "o"), ("jev", .003, "s")):
        x, coverage, acc, certified = operating(judge, dataset)
        color = COLORS[judge]
        axes[0, col].plot(x+offset, coverage, marker=marker, color=color, lw=2.4, ms=7, label=f"{('GPT-5' if judge=='gpt' else 'Jev')} + xverifier")
        axes[1, col].plot(x+offset, acc, marker=marker, color=color, lw=2.4, ms=7)
        for xx, yy, count in zip(x+offset, coverage, certified):
            axes[0, col].annotate(f"{count}/5", (xx, yy), xytext=(0, 10 if judge == "gpt" else -18),
                                  textcoords="offset points", ha="center", color=color, fontsize=9)
    axes[0, col].set_title(label, loc="left", pad=12)
    axes[1, col].plot([.73, .97], [.73, .97], ls="--", color=MUTED, lw=1.1, label="Accuracy goal")
    axes[1, col].set_ylim(.4, 1.02)
    axes[0, col].set_ylim(-.08, 1.1)
    for row in (0, 1):
        ax = axes[row, col]
        ax.set_xlim(.735, .965)
        ax.set_xticks([.75, .8, .85, .9, .95])
        percent(ax, both=True)
        ax.set_xlabel("Requested minimum accuracy (1 − target error)")
    axes[0, col].set_ylabel("Automatic coverage")
    axes[1, col].set_ylabel("Accuracy on handled cases")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.51, .86), ncol=2, frameon=False)
footer(fig, "Labels such as 3/5 = runs with a qualifying cutoff. Coverage includes review-all runs; observed accuracy averages only runs that decided cases.\nMissing accuracy points mean no automatic decisions. Below the dashed line = missed the goal on evaluation; calibration does not guarantee each test slice.")
fig.subplots_adjust(top=.75, bottom=.15, left=.085, right=.97, wspace=.28, hspace=.4)
save(fig, "03_accuracy_goal_automation")


fig, axes = plt.subplots(1, 2, figsize=(13.3, 7.5))
title(fig, "Automation coverage versus accuracy", "Confidence sweep on matching held-out cases • direct comparison with Jev's own probability routing")
for ax, (dataset, label) in zip(axes, DATASETS):
    for judge, method, name, color, marker in SERIES:
        d = curve(dataset, judge, method).dropna(subset=["accuracy"])
        ax.plot(d.coverage, d.accuracy, color=color, lw=2.4, label=name)
        if len(d):
            ax.scatter(d.coverage.iloc[0], d.accuracy.iloc[0], color=color, marker=marker, s=45)
    for goal in (.75, .85, .95):
        ax.axhline(goal, color="#D5DDE8", lw=.9, ls="--")
    ax.set_title(label, loc="left", pad=14)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(.4, 1.02)
    ax.set_xlabel("Coverage: share handled automatically")
    ax.set_ylabel("Accuracy on automatic decisions")
    percent(ax, both=True)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.51, .86), ncol=3, frameon=False, fontsize=10.5)
footer(fig, "Upper-right is better: more cases automated at higher accuracy. This diagnostic frontier must not be used to choose a deployment cutoff on test data.\nAccuracy shown only where a seed has ≥20 selected cases; means may use fewer than five seeds. Limited sample; no population-level performance claim.")
fig.subplots_adjust(top=.75, bottom=.2, left=.085, right=.97, wspace=.27)
save(fig, "04_coverage_accuracy_tradeoff")


# A compact 16:9 summary for a presentation slide.
fig, axes = plt.subplots(1, 3, figsize=(16, 9))
title(fig, "Jev + xverifier: limited matched evaluation", f"GPT-5 and Jev on the same case pool • {S['sample_rows']:,} unfiltered / {S['cleaned_sample_rows']:,} filtered cases • five seeds")
for ax, (dataset, label) in zip(axes[:2], DATASETS):
    ax.set_title(f"{label}: overall accuracy", loc="left", pad=16, fontsize=15)
    accuracy(ax, dataset)
    ax.text(.02, .93, "Light: alone  ·  Solid: + xverifier", transform=ax.transAxes, fontsize=9, color=MUTED)
ax = axes[2]
x = np.array([0, 1])
for judge, offset in (("gpt", -.17), ("jev", .17)):
    values = [E[f"{judge}-{dataset}"]["operating_points"]["0.25"]["coverage_mean"] for dataset, _ in DATASETS]
    bars = ax.bar(x+offset, values, width=.29, color=COLORS[judge], label="GPT-5 + xv" if judge=="gpt" else "Jev + xv")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, value+.025, f"{value:.1%}", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(x, ["Unfiltered", "Filtered"])
ax.set_title("Automation coverage\n75% accuracy goal", loc="left", pad=16, fontsize=15)
ax.set_ylim(0, 1)
ax.set_ylabel("Coverage")
percent(ax)
ax.legend(frameon=False, loc="upper right", fontsize=10)
footer(fig, "Accuracy bars: mean ± seed SD. Coverage is measured using calibration-selected cutoffs, including review-all runs. Accuracy goal is not a test-set guarantee.\nLimited nonrandom source-train pilot with group-held-out evaluation slices; not canonical test performance. Full confidence and accuracy plots accompany this summary.")
fig.subplots_adjust(top=.77, bottom=.22, left=.065, right=.97, wspace=.32)
save(fig, "00_presentation_summary")
print(f"Saved five figures as PNG and SVG in {OUT}")
