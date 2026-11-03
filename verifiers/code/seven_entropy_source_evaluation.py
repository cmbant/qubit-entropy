"""Verified real polynomial evaluation of the joint and cubic source terms."""

import numpy as np
from numba import njit

u = 2.0**-53
inflate = 1 + 2**-20


def initialize(joint_maps, joint_coefficients, localizer, pp):
    global JM, JR, LP, probes
    JM, JR, LP, probes = joint_maps, joint_coefficients, localizer, pp
    assert np.all((probes["exponents"] >= 0) & (probes["exponents"] <= 32))
    for src in (JR, LP):
        for key in ("coeff", "radius"):
            a = abs(src[key])
            a = a[a != 0]
            assert np.isfinite(a).all() and a.min() > 2.0**-400 and a.max() < 2.0**400


def pairwise(x):
    x = np.asarray(x).reshape(-1)
    while len(x) > 1:
        n = len(x) // 2
        y = x[: 2 * n : 2] + x[1 : 2 * n : 2]
        x = np.r_[y, x[-1:]] if len(x) % 2 else y
    return float(x[0])


@njit
def poly_values(AB, mm, co, aco):
    ans = np.zeros(len(AB))
    bound = np.zeros(len(AB))
    terms = np.empty(len(mm))
    abterms = np.empty(len(mm))
    for p in range(len(AB)):
        power = np.ones((10, 8))
        for j in range(10):
            for k in range(1, 8):
                power[j, k] = power[j, k - 1] * AB[p, j]
        for t in range(len(mm)):
            v = 1.0
            for j in range(10):
                v *= power[j, mm[t, j]]
            terms[t] = v * co[t]
            abterms[t] = abs(v) * aco[t]
        size = len(mm)
        while size > 1:
            n = size // 2
            for j in range(n):
                terms[j] = terms[2 * j] + terms[2 * j + 1]
                abterms[j] = abterms[2 * j] + abterms[2 * j + 1]
            if size % 2:
                terms[n] = terms[size - 1]
                abterms[n] = abterms[size - 1]
            size = n + size % 2
        ans[p] = terms[0]
        bound[p] = abterms[0]
    return ans, bound


def run(i):
    num = probes["inputs"][i, 0].astype(np.int64)
    den = int(probes["denominator"])
    exp = int(probes["exponents"][i])
    scale = 2.0**exp
    AA = JM["trace"]
    assert 42 * int(abs(AA).max()) * int(abs(num).max()) < 2**53
    a = (AA @ num).astype(float) / den
    a2 = a * a
    a4 = a2 * a2
    vals = [pairwise(a4 * a2 * a) / 120]
    absvals = [pairwise(abs(a4 * a2 * a)) / 120]
    for j in range(1, 20):
        M = JM["w%d" % j]
        assert 42 * int(abs(M).max()) * int(abs(num).max()) < 2**53
        dd = den * int(JM["d%d" % j])
        assert dd < 2**53
        b = (M @ num).astype(float) / dd
        bb = b * b * b
        v = JM["n%d" % j][:, None] * bb * a4[None, :]
        vals.append(pairwise(v) / 14400)
        absvals.append(pairwise(abs(v)) / 14400)
    co = JR["coeff"]
    aco = abs(co) + JR["radius"] / (64 * u)
    vv = np.array(vals)
    av = np.array(absvals)
    jval = pairwise(vv * co) / scale
    jerr = 64 * u / (1 - 64 * u) * pairwise(av * aco) / scale * inflate
    orb = LP["orbits"]
    idx = orb[:, 0] * 10 + orb[:, 1] if probes["kinds"][i] == 4 else np.arange(100)
    w = orb[:, 2] if probes["kinds"][i] == 4 else np.ones(100)
    dd = den * int(LP["denominator"])
    M = np.concatenate([LP["MI"][idx], LP["MS"][idx]], axis=1)
    assert 42 * int(abs(M).max()) * int(abs(num).max()) < 2**53
    assert dd < 2**53
    AB = (M @ num).astype(float) / dd
    co = LP["coeff"]
    aco = abs(co) + LP["radius"] / (64 * u)
    lv, lb = poly_values(AB, LP["monomials"], co, aco)
    lval = pairwise(w * lv) / 100 / scale
    lerr = 64 * u / (1 - 64 * u) * pairwise(w * lb) / 100 / scale * inflate
    return (
        jval - lval,
        (jerr + lerr + u * (abs(jval) + abs(lval))) * inflate,
        jval,
        lval,
        jerr,
        lerr,
    )
