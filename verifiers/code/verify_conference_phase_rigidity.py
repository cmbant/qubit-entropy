#!/usr/bin/env python3
"""CP-4: exact complete order-six symmetric conference classification.

All 243 opposite-pairing patterns are checked. This supersedes the
local Jacobian test; the LU lift is proved in prop:confphase.
"""
from itertools import combinations

def edge(i,j): return (min(i,j),max(i,j))

# CP-4: complete phase classification, not only local rigidity.
# Each of the five dephased core rows is a zero-sum quartet of unit
# complex numbers and therefore has an opposite pairing. Enumerate the
# 3^5 local pairings; the 10 core edges carry paired minus-relations.
from collections import Counter
from itertools import product
CORE=tuple(range(1,6))
EDG5=list(combinations(CORE,2))
EID5={e:i for i,e in enumerate(EDG5)}
MATCH=[]
for vertex in CORE:
    a,b,c,d=[x for x in CORE if x!=vertex]
    MATCH.append([[(EID5[edge(vertex,i)],EID5[edge(vertex,j)]) for i,j in pairs]
                  for pairs in (((a,b),(c,d)),((a,c),(b,d)),((a,d),(b,c)))])
def components(choice):
    adj=[[] for e in EDG5]
    for v,pid in enumerate(choice):
        for a,b in MATCH[v][pid]:adj[a].append(b);adj[b].append(a)
    assert all(len(a)==2 for a in adj)
    sign=[0]*10;comp=[-1]*10;nc=0
    for i in range(10):
        if sign[i]:continue
        sign[i]=1;comp[i]=nc;todo=[i]
        while todo:
            a=todo.pop()
            for b in adj[a]:
                if sign[b] and sign[b]!=-sign[a]:return None
                if not sign[b]:sign[b]=-sign[a];comp[b]=nc;todo.append(b)
        nc+=1
    return sign,comp,nc
def orthogonality(sgn,comp):
    eq=[]
    for i,j in combinations(CORE,2):
        coeff={-1:0,0:1,1:0}
        for k in CORE:
            if k in (i,j):continue
            a=EID5[edge(i,k)];b=EID5[edge(j,k)]
            coeff[comp[a]-comp[b]]+=sgn[a]*sgn[b]
        eq.append(tuple(coeff[k] for k in (-1,0,1)))
    return eq
def pentagon(sgn,comp,t):
    B={e:sgn[i]*(-1 if comp[i] and t==-1 else 1) for i,e in enumerate(EDG5)}
    assert all(sum(B[edge(i,j)] for j in CORE if j!=i)==0 for i in CORE)
    assert all(1+sum(B[edge(i,k)]*B[edge(j,k)] for k in CORE if k not in (i,j))==0
               for i,j in combinations(CORE,2))
    neg=frozenset(e for e,v in B.items() if v==-1)
    assert len(neg)==5 and all(sum(i in e for e in neg)==2 for i in CORE)
    return neg

counts=Counter();pentagons=set();reps=0
for choice in product(range(3),repeat=5):
    z=components(choice)
    if z is None:
        counts['odd cycles']+=1;continue
    signs,comp,nc=z
    eq=orthogonality(signs,comp)
    if nc==1:
        assert all(not any(row) for row in eq)
        pentagons.add(pentagon(signs,comp,1))
        counts['one phase']+=1;reps+=1
    elif nc==2:
        first=next((row for row in eq if any(row)),None)
        assert first in ((1,0,-1),(-1,0,1)),first
        # t^-1 = t, hence t^2=1; all other equations checked.
        for t in (-1,1):
            assert all(row[0]*t+row[1]+row[2]*t==0 for row in eq)
            pentagons.add(pentagon(signs,comp,t));reps+=1
        counts['two phases']+=1
    else:raise AssertionError(('unexpected phase components',nc))
assert counts==Counter({'odd cycles':81,'one phase':132,'two phases':30}),counts
assert reps==192 and len(pentagons)==6
assert len(pentagons|{frozenset(set(EDG5)-set(x)) for x in pentagons})==12
print('CP-4 PASS: 243 core pairings: 81 inconsistent, 132 one-phase, 30 two-phase.')
print('CP-4 PASS: all two-phase cases force t^2=1; no nonreal phase ratio.')
print('CP-4 PASS: all resulting real cores are five-cycles (12 labelled choices).')
print('THEOREM: every complex symmetric conference C of order 6 equals D P S P^T D.')
print('COROLLARY: every CC-1 optimal seven-qubit state is LU + ordinary-qubit-permutation equivalent to Psi_ico.')
print('CP-4 scope: classification of conference phases only.')
