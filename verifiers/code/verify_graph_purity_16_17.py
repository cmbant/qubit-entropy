#!/usr/bin/env python3
"""GL-1--GL-2: exact Rains-LP dual and graph-state witnesses at n=16,17.

No optimization software is used: dual multipliers and binary cut ranks are
checked in exact rational/integer arithmetic.  The graph at n=16 is obtained
by deleting vertex 16 from the circulant C_17(1,3,4,5).
"""
from fractions import Fraction as F
from itertools import combinations
from collections import Counter
from math import comb

def forms(n):
    a = [[F(comb(n-j,s-j), 2**s*comb(n,s)) if j <= s else F(0)
          for j in range(n+1)] for s in range(n+1)]
    # Coefficient of x^(n-i) y^i in
    # ((x+3y)/2)^(n-j) * ((y-x)/2)^j.
    shadow = [[F(sum(
        comb(j,h)*(-1)**(j-h)*comb(n-j,i-h)*3**(i-h)
        for h in range(max(0,i-(n-j)), min(i,j)+1)
    ), 2**n) for j in range(n+1)] for i in range(n+1)]
    return a, shadow

# Dual: a_k - L*A_0 =
#   sum_{s<k} e_s*(a_s-a_{n-s}) + sum_i y_i*S_i + sum_j w_j*A_j,
# where y_i,w_j >= 0.  e_s are free.
CERT = {
    16: (F(1,156),
         ["21/286","-96/143","1120/429","-784/143",
          "70/11","-112/33","0","0"],
         {0:"1/1716",2:"17/102960",4:"1/34320"}),
    17: (F(1,208),
         ["4/143","-63/286","96/143","-120/143",
          "-24/143","18/11","-16/11","0"],
         {3:"1/97240",4:"1/1361360",5:"1/272272"}),
}

def verify_lp(n):
    L, es, ys = CERT[n]
    es = list(map(F, es)); ys = {i:F(y) for i,y in ys.items()}
    k=n//2
    assert len(es)==k and all(y >= 0 for y in ys.values())
    a,sh=forms(n)
    w = [a[k][j] - (L if j==0 else 0)
         - sum(es[s]*(a[s][j]-a[n-s][j]) for s in range(k))
         - sum(y*sh[i][j] for i,y in ys.items())
         for j in range(n+1)]
    assert all(x>=0 for x in w), w
    assert w[0]==0
    print("GL-1 PASS: n=%d, exact lower bound %s, nonnegative Rains multipliers and residual"
          % (n,L))
    print("  Positive Pauli-weight penalties:", [(j,str(x)) for j,x in enumerate(w) if x])
    return L

def rank_gf2(rows,n):
    rows=list(rows)
    r=0
    for c in range(n):
        pivot=next((i for i in range(r,len(rows)) if rows[i] & (1<<c)),None)
        if pivot is None:
            continue
        rows[r],rows[pivot]=rows[pivot],rows[r]
        for i in range(len(rows)):
            if i != r and rows[i] & (1<<c):
                rows[i] ^= rows[r]
        r+=1
    return r

def graph_hist(n):
    adj=[0]*n
    for i in range(17):
        for d in (1,3,4,5):
            j=(i+d)%17
            if i<n and j<n:
                adj[i] |= 1<<j
                adj[j] |= 1<<i
    counts=Counter()
    for S in combinations(range(n), n//2):
        mask=sum(1<<i for i in S)
        rest=((1<<n)-1) ^ mask
        r=rank_gf2([adj[i]&rest for i in S],n)
        counts[r]+=1
    assert sum(counts.values())==comb(n,n//2)
    purity=sum(F(c,2**r) for r,c in counts.items())/comb(n,n//2)
    return purity,dict(sorted(counts.items()))

if __name__ == "__main__":
    for n,expected_hist in (
        (16,{5:10,6:500,7:6680,8:5680}),
        (17,{6:510,7:4080,8:19720}),
    ):
        bound=verify_lp(n)
        purity,hist=graph_hist(n)
        assert hist==expected_hist,(n,hist)
        assert purity==bound
        print("GL-2 PASS: n=%d, graph cuts %s, purity %s = exact LP lower bound" %
              (n,hist,purity))
