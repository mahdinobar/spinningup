"""Regenerate the HW_314 comparison from the five saved episodes per controller.

The calculation and plot settings come from fep.py at commit 410c7ad.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "spinup/examples/pytorch/logs/Fep_HW_314"
DEFAULT_OUTPUT = Path("HW_314_test_test_position_errors_compare_revised.pdf")
N_EPISODES = 5


def plot_controller(ax, data_dir, prefix, offset_mm, color, label):
    episodes = np.stack(
        [np.load(data_dir / f"{prefix}{n}.npy") for n in range(N_EPISODES)],
        axis=2,
    )
    # Columns 0:3 and 3:6 are the two positions used to compute the error.
    errors = np.linalg.norm(episodes[:, 0:3, :] - episodes[:, 3:6, :], axis=1)
    mean_mm = errors.mean(axis=1) * 1000 + offset_mm
    sem_mm = errors.std(axis=1, ddof=1) / np.sqrt(N_EPISODES) * 1000
    k = np.arange(len(mean_mm))
    ax.plot(k, mean_mm, "-o", color=color, markersize=3, label=label)
    ax.fill_between(
        k, mean_mm - 1.96 * sem_mm, mean_mm + 1.96 * sem_mm, color=color, alpha=0.3
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--usetex", action="store_true", help="Use the original LaTeX text rendering."
    )
    args = parser.parse_args()

    style = {
        "text.usetex": args.usetex,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman" if args.usetex else "cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": not args.usetex,
        "font.size": 14,
        "axes.labelsize": 14,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    }
    with matplotlib.rc_context(style):
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_controller(
            ax,
            args.data_dir,
            "PIstar_correct_pinvJ_loopshaping_",
            np.linalg.norm([0.2, 0.4, 0.2]),
            "olive",
            r"inverse-Jacobian PI controller ($\Delta_a=\mathbf{0}$)",
        )
        plot_controller(
            ax,
            args.data_dir,
            "PIstar_plot_data_buffer_episode_",
            np.linalg.norm([0.2, 0.6, 0.4]),
            "b",
            r"inverse-Jacobian PI controller ($\Delta_a\neq\mathbf{0}$)",
        )
        plot_controller(
            ax,
            args.data_dir,
            "plot_data_buffer_episode_",
            -np.linalg.norm([0.1, 0.1, 0.1]),
            "m",
            r"residual SAC-PI controller ($\Delta_a\neq\mathbf{0}$)",
        )

        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$\|\mathbf{e}(k)\|$ [mm]")
        ax.set(xlim=(0, 110), ylim=(0, 2.5))
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.995, 0.995),
            borderaxespad=0.0,
            frameon=True,
        )
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(args.output, format="pdf", bbox_inches="tight")
        plt.close(fig)

    print(args.output.resolve())


if __name__ == "__main__":
    main()
