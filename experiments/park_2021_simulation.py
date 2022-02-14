"""Replcates the experiment from Park 2021 CoDiTE paper"""

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from sparse_shift import KCD, KCDCV
import warnings
warnings.filterwarnings("ignore")


def sample_toy_data(n):
    X = np.random.uniform(0, 1, (n, 1))

    X_below = (X < 0.3)[:, 0]
    X_above = (X >= 0.3)[:, 0]
    y0 = (
        3
        + 5 * X[:, 0]
        + X_below * np.random.normal(0, 1, (n))
        + X_above * np.random.normal(0, np.abs((1 + 7 * (X[:, 0] - 0.3))), (n))
    )
    y1 = (
        4 * X[:, 0]
        + X_below * np.random.normal(0, 1, (n))
        + X_above * np.random.normal(0, np.abs((1 + 7 * (X[:, 0] - 0.3))), (n))
    )

    return X, np.vstack((y0, y1)).T


def main(args):
    n = args.n_samples
    n_reps = args.reps
    stats = {"Y0": [], "Y1": [], "Y01": []}
    pvalues = {"Y0": [], "Y1": [], "Y01": []}

    # Pvalues
    for i in range(args.power_reps):
        np.random.seed(i)
        X, y_outcomes = sample_toy_data(n)
        z = np.random.choice(2, (n))
        y = np.asarray([outcomes[i] for outcomes, i in zip(y_outcomes, z)])
        if args.test == 'kcd':
            kcd = KCD(reg=1.0, n_jobs=-2)
        elif args.test == 'kcdcv':
            kcd = KCDCV(regs=[1e-2, 1e-1, 1, 10, 100], n_jobs=-2)
        else:
            raise ValueError(f'Test {args.test} not valid.')

        stat, pvalue = kcd.test(X, y, z, reps=n_reps, fast_pvalue=args.fast_pvalue)
        stats["Y01"].append(stat)
        pvalues["Y01"].append(pvalue)

        stat, pvalue = kcd.test(X, y_outcomes[:, 0], z, reps=n_reps, fast_pvalue=args.fast_pvalue)
        stats["Y0"].append(stat)
        pvalues["Y0"].append(pvalue)

        stat, pvalue = kcd.test(X, y_outcomes[:, 1], z, reps=n_reps, fast_pvalue=args.fast_pvalue)
        stats["Y1"].append(stat)
        pvalues["Y1"].append(pvalue)

    # Pvalue distribution plots, should be close to uniform
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for i, (key, val) in enumerate(pvalues.items()):
        ax = axes[i]
        n = len(val)
        entries, edges, _ = ax.hist(
            val,
            bins=np.arange(0, 1.1, 0.1),
            weights=np.ones(n) / n,
            color="b",
        )
        # entries = height of each column = proportion in that bin
        # calculate bin centers
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        ax.axhline(y=sum(entries) / len(bin_centers), ls="--", c="#333333")
        # errorbars are binomial proportion confidence intervals
        ax.errorbar(
            bin_centers,
            entries,
            yerr=1.96 * np.sqrt(entries * (1 - entries) / n),
            fmt=".",
            c="#333333",
        )
        ax.set_title(f"Test {key} pvalues")
        # ax.set_xlim(0,1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 0.1, 1])

    plt.savefig(f'./figures/park_pval_dist_{args.test}.pdf')

    np.random.seed(100)
    n = 100
    X, y_outcomes = sample_toy_data(n)
    z = np.random.choice(2, (n))
    y = np.asarray([outcomes[i] for outcomes, i in zip(y_outcomes, z)])

    # Witness function
    X_lin = np.linspace(0, 1, 1000)
    Y_lin = np.linspace(-10, 25, 1000)

    kcd = KCD(reg=1.0, n_jobs=10)
    witness_mat = kcd.witness(X, y, z, X_lin, Y_lin)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    # Raw data and mean trend lines
    ax = axes[0]
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    ax.set_title("(a) Data")
    ax.scatter(X[:, 0], y_outcomes[:, 0], label=r"$Y_0$", c="blue", marker="x", alpha=0.2)
    ax.plot(
        np.linspace(0, 1, 10),
        3 + 5 * np.linspace(0, 1, 10),
        label=r"$E[Y_0|X]$",
        c="darkblue",
        linewidth=2,
        ls="-",
    )
    ax.scatter(X[:, 0], y_outcomes[:, 1], label=r"$Y_1$", c="orange", marker="o", alpha=0.2)
    ax.plot(
        np.linspace(0, 1, 10),
        4 * np.linspace(0, 1, 10),
        label=r"$E[Y_1|X]$",
        c="orangered",
        linewidth=2,
        ls="-",
    )
    ax.legend(loc="upper left")

    # Hypothesis test results
    ax = axes[1]
    ax.set_ylabel("Proportion of tests rejected")
    ax.set_xlabel("Hypothesis")
    ax.set_title("(b) Hypothesis Test")

    alpha = 0.05
    ax.bar(
        [1, 2, 3],
        [
            np.mean(np.asarray(pvalues["Y0"]) <= alpha),
            np.mean(np.asarray(pvalues["Y1"]) <= alpha),
            np.mean(np.asarray(pvalues["Y01"]) <= alpha),
        ],
        align="center",
    )
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(
        [
            r"$P_{Y_0|X}$ vs $P_{Y_0|X}$",
            r"$P_{Y_1|X}$ vs $P_{Y_1|X}$",
            r"$P_{Y_0|X}$ vs $P_{Y_1|X}$",
        ]
    )
    ax.axhline(0.05, ls='--', c='grey')

    # Witness function
    ax = axes[2]
    ax.set_ylabel("Y")
    ax.set_xlabel("X")
    ax.set_title("(c) Conditional Witness Function")
    cm = ax.pcolormesh(X_lin, Y_lin, witness_mat, cmap="magma")
    fig.colorbar(cm, ax=ax)
    plt.tight_layout()
    plt.savefig(f'./figures/park_3plot_{args.test}.pdf')
    return


if __name__ == "__main__":
    parser = ArgumentParser(description="Experiment parameters")

    parser.add_argument(
        "--test", choices=["kcd", "kcdcv"], help="Hypothesis test to use"
    )
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--power_reps", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--fast_pvalue", action="store_true", default=False)

    args = parser.parse_args()

    main(args)
