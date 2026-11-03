"""Compact E7-1 inputs; all physical integer arrays decode exactly to int64."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "data/seven_entropy"


def load_bases():
    with np.load(ROOT / "bases.npz", allow_pickle=False) as data:
        assert int(data["format"]) == 1
        return {key: data[key].astype(np.int64) if key.startswith("u") else data[key]
                for key in data.files if key != "format"}


def load_constraints():
    with np.load(ROOT / "constraints.npz", allow_pickle=False) as data:
        assert int(data["format"]) == 1
        out = {}
        for i in range(17):
            columns = data[f"c{i}"].astype(np.int64)
            index = data[f"i{i}"].astype(np.int64)
            assert columns.ndim == 3 and columns.shape[0] == 6
            assert index.ndim == 1 and np.all((index >= 0) & (index < columns.shape[2]))
            out[f"c{i}"] = columns[:, :, index]
        return out
