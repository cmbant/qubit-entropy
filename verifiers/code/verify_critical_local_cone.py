#!/usr/bin/env python3
"""LCC: local cone lemma at the equality points of the critical chart (critical_sharp_coefficient.tex).

At each of the eight chart images of the equality points (four per parity sector) exactly four planes are tied
(G_P = 0 there, all others < -0.05); their gradients g_P sum to 0 and have rank 3; the Hessian of G_P is enclosed over the
box of side RHO around the point; the sphere of the sup norm (orthant of the four chart coordinates times the w-square)
is covered by cells on which some weighted average of the four tied G_P is positive along every ray from the apex
(critical_cone_lib.py).  Hence G > 0 on the box except at the point itself.
"""
import time
from fractions import Fraction
import numpy as np
import critical_float as cf
from critical_cone_lib import apex_data, cone_cells

RHO = Fraction(1, 256)
APEXES = [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)]

EXPECT = {(1, (0, 0, 0, 1)): [0, 2, 6, 7], (1, (0, 0, 1, 0)): [2, 3, 5, 7], (1, (0, 1, 0, 0)): [0, 2, 4, 5], (1, (1, 0, 0, 0)): [0, 1, 5, 7],
          (-1, (0, 0, 0, 1)): [8, 11, 13, 15], (-1, (0, 0, 1, 0)): [9, 11, 13, 14], (-1, (0, 1, 0, 0)): [9, 10, 13, 15],
          (-1, (1, 0, 0, 0)): [8, 10, 13, 14]}          # the tied planes proved exactly in verify_critical_apex_exact.py
tot = 0
for sector in (1, -1):
    for ap in APEXES:
        t0 = time.time()
        tied, g, H, vals = apex_data(sector, ap, RHO)
        assert tied == EXPECT[(sector, ap)], tied
        sv = np.linalg.svd(np.array([[float(g[p][k].mid()) for k in range(6)] for p in tied]))[1]
        assert sv[2] > 0.07 and sv[3] < 1e-12, sv           # the tied gradients have rank three
        ok, st = cone_cells(sector, ap, RHO, maxdepth=40)
        assert ok, (sector, ap, st)
        tot += st['cells']
        # audit (numerical regression, not a proof): sum_P mu_P G_P(t w) >= t l_min + t^2 q_min at random points of the certified cells
        rng = np.random.default_rng(1)
        sig = np.array([1 if a == 0 else -1 for a in ap]+[1, 1], float)
        base = np.array(list(ap)+[0, 0], float)
        for lo_, hi_, (nu, lmin, qmin) in st['data'][::max(1, len(st['data'])//40)]:
            for _ in range(20):
                om = np.array([rng.uniform(float(a), float(b)) for a, b in zip(lo_, hi_)])
                t_ = float(RHO)*rng.uniform(0.01, 1)
                z = t_*om
                y = base+sig*z*np.array([1, 1, 1, 1, 0, 0])+np.array([0, 0, 0, 0, z[4], z[5]])
                Gp = cf.Gp_f(y[None], sector)[0]
                avg = sum((0.25+float(nu[i]))*Gp[tied[i]] for i in range(4))
                assert avg >= t_*lmin+t_*t_*qmin-1e-14, (sector, ap, lo_, hi_)
        print('[PASS] LCC sector %+d apex %s: tied planes %s (G_P = 0 there, others < -0.05), sum of tied gradients 0 of rank 3 (singular values %.4f, %.4f, %.4f), '
              '%d cone cells found (depth <= %d), audit at sampled points passed'
              % (sector, ap, tied, sv[0], sv[1], sv[2], st['cells'], st['depth']))
print('[PASS] LCC local cone lemma at all eight chart equality points, box side 1/256: %d cone cells' % tot)
