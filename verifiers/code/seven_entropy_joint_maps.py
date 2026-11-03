"""Exact colored-permutation source maps for E7-1."""

import json, math
from pathlib import Path
import numpy as np, sympy as sp
import verify_five_copy_site_algebra as s5

ROOT = Path(__file__).resolve().parent.parent / "data" / "seven_entropy"


def build():
    # use only the feature list from the search, reconstruct every map exactly
    features = [
        tuple(tuple(w) for w in ff)
        for ff in json.loads((ROOT / "joint_features.json").read_text())
    ]
    RR = {m: np.array(s5.RHO[m], np.int64) for m in range(3)}
    RI = {m: RR[m][s5.PINV] for m in range(3)}
    trace = np.concatenate(
        [s5.UDIM[m] * RR[m].transpose(0, 2, 1).reshape(120, -1) for m in range(3)],
        axis=1,
    )
    out = {"trace": trace}
    for no, ff in enumerate(features):
        if not ff:
            continue
        perm = list(range(5))
        flip = []
        off = 0
        for w in ff:
            for j, v in enumerate(w):
                perm[off + j] = off + (j + 1) % len(w)
                if v:
                    flip.append(off + j)
            off += len(w)
        B = sp.Matrix(s5.flip(s5.OPS[s5.PINDEX[tuple(perm)]], tuple(flip)))
        bb = {m: s5.L32[m] * B * s5.V32[m] for m in range(3)}
        den = math.lcm(*[int(z.q) for M in bb.values() for z in M])
        BT = {m: RR[m] @ np.array(bb[m] * den, np.int64) @ RI[m] for m in range(3)}
        _, unique, cnt = np.unique(
            np.concatenate([BT[m].reshape(120, -1) for m in range(3)], axis=1),
            axis=0,
            return_index=True,
            return_counts=True,
        )
        maps = np.concatenate(
            [
                s5.UDIM[m]
                * np.einsum("tij,sjk->tsik", BT[m][unique], RR[m])
                .transpose(0, 1, 3, 2)
                .reshape(len(unique), 120, -1)
                for m in range(3)
            ],
            axis=2,
        )
        out["w%d" % no] = maps
        out["n%d" % no] = cnt
        out["d%d" % no] = den
        print("EXACT JOINT MAP", no, len(unique), den, flush=True)
    return out
