from pathlib import Path
import matplotlib.pyplot as plt


def save_current_figure(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")