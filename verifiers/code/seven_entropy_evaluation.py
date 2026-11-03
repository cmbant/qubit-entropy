"""Direct five-copy evaluation, with physical input and rounding error bounds."""

from five_copy_probe_evaluation import *


def symbol(models, maps, input_num, input_den, kind, batch=8):
    outputs = [np.zeros((M["repair_rank"], M["VR"].shape[1])) for M in models]
    bounds = [np.zeros_like(x.real) for x in outputs]
    traces = np.zeros(len(models))
    trace_abs = np.zeros(len(models))
    tradius = np.zeros(len(models))
    for flip in [0, 1, 2]:
        dd = maps[flip]
        N = dd["n"]
        pairs = (
            dd["orbits"]
            if kind == 4
            else [(a, b, 1) for a in range(N) for b in range(N)]
        )
        mm = [(i, M) for i, M in enumerate(models) if M["flip"] == flip]
        raw = []
        weights = []
        for a, b, w in pairs:
            K = dd["mats"][a, b]
            assert 42 * int(abs(K).max()) * int(abs(input_num).max()) < 2**53
            rr = K @ input_num[0]
            ii = K @ input_num[1]
            assert 0 < input_den * dd["den"] < 2**53
            den = float(input_den * dd["den"])
            assert not np.any(ii)
            raw.append(rr / den)
            weights.append(w)
        for lo in range(0, len(raw), batch):
            z = np.array(raw[lo : lo + batch])
            P = len(z)
            rp, ra = powers_real(z[:, 17:].reshape(P, 5, 5))
            hh = z[:, 1:17].reshape(P, 4, 4)
            rp[4, 1] = hh
            ra[4, 1] = abs(hh)
            ws = np.array(weights[lo : lo + P])
            for index, M in mm:
                D, k = M["VR"].shape
                if (4, 2) in M["fac"]:
                    ps = list(
                        itertools.combinations(range(4), 2)
                        if M["wedge"]
                        else itertools.combinations_with_replacement(range(4), 2)
                    )
                    rp[4, 2] = np.array(
                        [
                            [
                                hh[:, c, a] * hh[:, d, b]
                                + (
                                    -hh[:, d, a] * hh[:, c, b]
                                    if M["wedge"]
                                    else (hh[:, d, a] * hh[:, c, b] if c != d else 0)
                                )
                                for a, b in ps
                            ]
                            for c, d in ps
                        ]
                    ).transpose(2, 0, 1)
                    ha = abs(hh)
                    ra[4, 2] = np.array(
                        [
                            [
                                ha[:, c, a] * ha[:, d, b]
                                + (
                                    ha[:, d, a] * ha[:, c, b]
                                    if M["wedge"] or c != d
                                    else 0
                                )
                                for a, b in ps
                            ]
                            for c, d in ps
                        ]
                    ).transpose(2, 0, 1)
                r = M["repair_rank"]
                X = act_real(M, np.broadcast_to(M["VR"][None], (P, D, k)).copy(), rp)
                Y = act_real(M, np.broadcast_to(M["VRU"][None], (P, D, r)).copy(), rp)
                AX = act(M, np.broadcast_to(M["AVR"][None], (P, D, k)).copy(), ra)
                AY = act(M, np.broadcast_to(M["AVRU"][None], (P, D, r)).copy(), ra)
                Z = left_product(M["VLU"], X) + left_product(M["VL"], Y).transpose(
                    0, 2, 1
                )
                AZ = M["AVLU"].T @ AX + (M["AVL"].T @ AY).transpose(0, 2, 1)
                tv = trace_product(M["VL"], X)
                at = np.einsum("ij,pij->p", M["AVL"], AX)
                coef = M["coef"] / (N * N)
                outputs[index] += np.einsum("p,pij->ij", ws, Z) * coef
                bounds[index] += np.einsum("p,pij->ij", ws, AZ) * coef
                traces[index] += np.dot(ws, tv) * coef
                trace_abs[index] += np.dot(ws, at) * coef
        for index, M in mm:
            D, k = M["VR"].shape
            actor = sum(
                min(di, 32) + math.ceil(math.log2(math.ceil(di / 32)))
                for di in M["dims"]
            )
            final = min(D, 128) + math.ceil(math.log2(math.ceil(D / 128)))
            diag = (
                min(D, 32)
                + math.ceil(math.log2(math.ceil(D / 32)))
                + math.ceil(math.log2(k))
                + 1
            )
            degree = sum(n for d, n in M["fac"])
            summation = batch + math.ceil(len(pairs) / batch) + 16
            ops = actor + final + 6 * degree + summation
            tops = actor + diag + 6 * degree + summation
            absops = D * k + sum(M["dims"]) + 6 * degree + 2 * N * N + 32
            u = 2**-53
            g = ops * u / (1 - ops * u)
            tg = tops * u / (1 - tops * u)
            ag = absops * u / (1 - absops * u)
            infl = 1 / (1 - ag)
            assert ops >= 64 and tops >= 64
            bounds[index] *= g * infl * (1 + 2**-20)
            tradius[index] = tg * infl * (1 + 2**-20) * (1 + 2**-20) * trace_abs[index]
    return (
        np.concatenate([z.ravel() for z in outputs]),
        np.concatenate([b.ravel() for b in bounds]),
        pairwise(list(traces)),
        sum(tradius),
    )
