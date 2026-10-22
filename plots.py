from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skrf as rf

from rf_tools import PatchResult


def _network(res: PatchResult, name: str) -> rf.Network:
    return rf.Network(frequency=rf.Frequency.from_f(res.freqs, unit="hz"),
                      s=res.s11.reshape(-1, 1, 1), name=name)


def plot_patch(results: dict[str, PatchResult], f_target: float, out: str | Path) -> str:
    """S11 (dB) and Smith chart for one or more model runs over a shared sweep."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for label, res in results.items():
        ax1.plot(res.freqs / 1e9, 20 * np.log10(np.abs(res.s11) + 1e-15), label=label)
        _network(res, label).plot_s_smith(ax=ax2, show_legend=False)
    ax1.axhline(-10, ls="--", c="grey", lw=0.8)
    ax1.axvline(f_target / 1e9, ls=":", c="k", lw=0.8)
    ax1.set(xlabel="Frequency (GHz)", ylabel="|S11| (dB)", title="Return loss")
    ax1.set_ylim(top=0)
    ax1.legend(fontsize=8)
    ax2.set_title("Smith chart")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


def save_touchstone(res: PatchResult, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _network(res, path.stem).write_touchstone(str(path.with_suffix("")))
    return str(path.with_suffix(".s1p"))
