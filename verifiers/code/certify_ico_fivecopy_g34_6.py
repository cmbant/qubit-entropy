"""Exact icosahedral g34 one-[4,1]/six-[3,2] [6] block rank and kernel."""
import argparse
import json
from pathlib import Path

import numpy as np

from certify_ico_fivecopy_g34_51 import (CYCLE, PATTERNS,
    ico_integer_states, local_maps, physical_sector, spinflip, interpolate)
from five_copy_symmetric_models import (invariant_bases, symmetric_lookup,
    sym6_image, exact_projection)
from five_copy_exact_kernel import exact_rank_and_kernel


def check_pentagon_orbit(T,Y,order,starts):
    """The six GL(5) slots symmetrize; site cycles permute right spin columns."""
    old_ord=[i for i in range(7) if i!=2]
    new_special=CYCLE.index(2)
    new_pat=tuple(1 if i==new_special else 2 for i in range(7))
    new_ord=[i for i in range(7) if i!=new_special]
    pi=[old_ord.index(CYCLE[i]) for i in new_ord]
    assert sorted(pi)==list(range(6))
    Yn=sym6_image(T.transpose(CYCLE),new_pat,order,starts)
    expect=Y.reshape(840,4,*([2]*6)).transpose([0,1]+[2+pi[i] for i in range(6)]).reshape(840,256)
    assert np.array_equal(Yn,expect)
    print('exact D5 pentagon placement covariance PASS',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path)
    args=ap.parse_args()
    u,v=ico_integer_states();maps=local_maps();_,U12=invariant_bases()
    order,starts=symmetric_lookup(6)
    print('exact Q4 tensor Sym^6 H-invariants: ambient 840, block 77',flush=True)
    xs=[]
    for t in range(6):
        z=u+t*v;jz=spinflip(z);blocks=[]
        for pat in PATTERNS:
            T=physical_sector(z,jz,pat,maps)
            Y=sym6_image(T,pat,order,starts)
            if t==0 and pat==PATTERNS[2]:check_pentagon_orbit(T,Y,order,starts)
            blocks.append(exact_projection(U12,Y))
        X=np.hstack(blocks);xs.append(X)
        print('t',t,'three placement reps; max entry',int(np.abs(X).max()),flush=True)
    A,B,cm=interpolate(np.stack(xs))
    print('coefficient maxima',cm,'target pair maxima',int(np.abs(A).max()),int(np.abs(B).max()),flush=True)
    kernel=exact_rank_and_kernel(A,B,8)
    print('target exact rank 8 of 77; exact left kernel 69, all-column annihilation PASS',flush=True)
    data_dir=Path(__file__).resolve().parent.parent/'data'
    pair_path=data_dir/'five_copy_ico_g34_pair6.npz'
    kernel_path=data_dir/'five_copy_ico_g34_kernel6.json'
    assert pair_path.exists()==kernel_path.exists(), 'partial committed kernel artifacts'
    if pair_path.exists():
        old=np.load(pair_path)
        assert np.array_equal(old['A'],A) and np.array_equal(old['B'],B)
        assert json.loads(kernel_path.read_text())==kernel
        print('committed compact artifacts reproduced exactly',flush=True)
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output_dir/'five_copy_ico_g34_pair6.npz',A=A,B=B)
        with open(args.output_dir/'five_copy_ico_g34_kernel6.json','w') as f:json.dump(kernel,f)
    print('PASS',flush=True)


if __name__=='__main__':main()
