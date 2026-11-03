#!/usr/bin/env python3
"""CCV: interval covering of the critical locus (tau = |w|^2 <= 37/100) for  delta >= C* d^2.

Chart of critical_cells.py: y=(P,Q,P',Q',w1,w2) in [0,1]^4 x {|w|^2 <= 37/100}, both parity sectors.  A box is
discharged by critical_cover_lib.test_cell (ball arithmetic): (G) a certified positive lower bound of delta - C* d^2
(second-order interval Taylor bound of a convex combination of plane functions), (HS) d^2 <= 1/400, or the box lies in a
local cone box of an equality point, which verify_critical_local_cone.py proves.  Bisection of the widest side.  A float
twin (critical_float.py) only decides which boxes are worth a ball-arithmetic attempt; it certifies nothing.
Usage: certify_critical_covering.py [--workers N] [--quick]   (--quick covers a sample of the initial boxes and certifies nothing)
"""
import itertools
import sys
import time
from fractions import Fraction
from multiprocessing import Pool
import numpy as np
from critical_cover_lib import test_cell, outside_disc
import critical_float as cf

RHO = Fraction(1, 256)                                  # side of the local cone boxes (verify_critical_local_cone.py)
APEXES = [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)]   # equality points (w=0), both sectors
MINW = 2.0**-30
TAU_SKIP = 0.37+1e-9
CHUNK = 20000


def in_core_arr(lo, hi, rho):
    rho = float(rho)
    res = np.zeros(len(lo), bool)
    for ap in APEXES:
        ok = np.ones(len(lo), bool)
        for i in range(4):
            if ap[i] == 0:
                ok &= (lo[:, i] >= 0) & (hi[:, i] <= rho)
            else:
                ok &= (hi[:, i] <= 1) & (lo[:, i] >= 1-rho)
        ok &= (lo[:, 4] >= -rho) & (hi[:, 4] <= rho) & (lo[:, 5] >= -rho) & (hi[:, 5] <= rho)
        res |= ok
    return res


def outside_arr(lo, hi):
    m = np.zeros(len(lo))
    for i in (4, 5):
        mn = np.where((lo[:, i] <= 0) & (hi[:, i] >= 0), 0.0, np.minimum(abs(lo[:, i]), abs(hi[:, i])))
        m += mn*mn
    return m > TAU_SKIP


def work(task):
    sector, lo0, hi0 = task
    lo = np.array([[float(x) for x in lo0]])
    hi = np.array([[float(x) for x in hi0]])
    st = dict(G=0, HS=0, core=0, out=0, arb=0, arbfail=0)
    fail = None
    while len(lo) and fail is None:
        nlo, nhi = [], []
        for s0 in range(0, len(lo), CHUNK):
            l, h = lo[s0:s0+CHUNK], hi[s0:s0+CHUNK]
            keep = ~(outside_arr(l, h) | (l[:, 0] > h[:, 2]))      # |w|^2 > 0.37, or P > P' everywhere (transposition)
            st['out'] += int((~keep).sum())
            l, h = l[keep], h[keep]
            if len(l) == 0:
                continue
            core = in_core_arr(l, h, RHO)
            st['core'] += int(core.sum())
            l, h = l[~core], h[~core]
            if len(l) == 0:
                continue
            cand = cf.candidates(l, h, sector)
            todo = np.ones(len(l), bool)
            for i in np.nonzero(cand)[0]:
                st['arb'] += 1
                tag = test_cell([float(x) for x in l[i]], [float(x) for x in h[i]], sector)
                if tag:
                    st[tag] += 1
                    todo[i] = False
                else:
                    st['arbfail'] += 1
            l, h = l[todo], h[todo]
            if len(l) == 0:
                continue
            w = h-l
            idx = np.argmax(w, axis=1)
            ar = np.arange(len(l))
            if (w[ar, idx] < MINW).any():
                j = int(np.argmax(w[ar, idx] < MINW))
                fail = (sector, tuple(l[j]), tuple(h[j]))
                break
            m = (l[ar, idx]+h[ar, idx])/2
            h1 = h.copy()
            h1[ar, idx] = m
            l2 = l.copy()
            l2[ar, idx] = m
            nlo += [l, l2]
            nhi += [h1, h]
        lo = np.concatenate(nlo) if nlo else np.zeros((0, 6))
        hi = np.concatenate(nhi) if nhi else np.zeros((0, 6))
    return sector, st, fail


def tasks():
    root_w = Fraction(5, 8)
    parts = [(Fraction(i, 2), Fraction(i+1, 2)) for i in range(2)]
    wparts = [(-root_w, Fraction(0)), (Fraction(0), root_w)]
    out = []
    for sector in (1, -1):
        for cs in itertools.product(parts, parts, parts, parts):
            for ws in itertools.product(wparts, wparts):
                out.append((sector, tuple([c[0] for c in cs]+[w[0] for w in ws]),
                            tuple([c[1] for c in cs]+[w[1] for w in ws])))
    return out


if __name__ == '__main__':
    import os
    nw = 4
    args = sys.argv[1:]
    if '--workers' in args:
        nw = int(args[args.index('--workers')+1])
    tk = tasks()
    n_required = len(tk)
    quick = '--quick' in args
    if quick:
        tk = tk[::17]                               # a sample of the tasks only: no global certificate
    prog = os.environ.get('CCV_PROGRESS')           # optional progress file (not part of the captured log)
    t0 = time.time()
    keys = ('G', 'HS', 'core', 'out', 'arb', 'arbfail')
    tot = {1: dict.fromkeys(keys, 0), -1: dict.fromkeys(keys, 0)}
    nfail = 0
    with Pool(nw) as pool:
        for k, (sector, st, fail) in enumerate(pool.imap_unordered(work, tk)):
            for key in keys:
                tot[sector][key] += st[key]
            if fail:
                nfail += 1
                print('UNDECIDED', fail, flush=True)
            if prog:
                with open(prog, 'a') as fh:
                    fh.write('tasks %d/%d  %.0f s  %s\n' % (k+1, len(tk), time.time()-t0, tot))
    if nfail:
        raise SystemExit('FAIL: %d undecided boxes' % nfail)
    # the counts are this script's own output; they depend on the float ordering heuristic only through which boxes are
    # tried first (a different numpy may change them slightly without affecting the proof)
    for s in (1, -1):
        print('[PARTIAL] ' if quick else '[PASS] ', end='')
        print('CCV sector %+d: %d boxes discharged by a certified bound G>0, %d by d^2<=1/400, %d inside the local '
              'cone boxes (verify_critical_local_cone.py); %d boxes skipped (tau>0.37 or P>P\', transposition); '
              '%d ball-arithmetic attempts, %d unsuccessful'
              % (s, tot[s]['G'], tot[s]['HS'], tot[s]['core'], tot[s]['out'], tot[s]['arb'], tot[s]['arbfail']))
    print('CCV tasks completed: %d of %d required initial boxes' % (len(tk), n_required))
    if quick:
        print('[PARTIAL] --quick: only %d of %d initial boxes were covered; NO GLOBAL CERTIFICATE (run without --quick)'
              % (len(tk), n_required))
    else:
        print('[PASS] CCV covering complete: every box of the chart with |w|^2 <= 37/100, both parity sectors, is '
              'discharged')
