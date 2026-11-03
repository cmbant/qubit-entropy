"""Exact icosahedral g34 all-[3,2] [7] block rank and left kernel."""
import argparse
import json
from pathlib import Path

import numpy as np

from certify_ico_fivecopy_g34_51 import (CYCLE, ico_integer_states,
    local_maps, physical_sector, spinflip, interpolate)
from five_copy_symmetric_models import (invariant_bases, symmetric_lookup,
    sym7_image, exact_projection)
from five_copy_exact_kernel import exact_rank_and_kernel


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path)
    args=ap.parse_args()
    u,v=ico_integer_states();maps=local_maps();U0,_=invariant_bases()
    order,starts=symmetric_lookup(7)
    print('exact Sym^7 H-invariants: ambient 330, block 33',flush=True)
    xs=[]
    for t in range(6):
        z=u+t*v;jz=spinflip(z)
        T=physical_sector(z,jz,(2,)*7,maps)
        Y=sym7_image(T,order,starts)
        if t==0:
            Yn=sym7_image(T.transpose(CYCLE),order,starts)
            expect=Y.reshape(330,*([2]*7)).transpose([0]+[1+CYCLE[i] for i in range(7)]).reshape(330,128)
            assert np.array_equal(Yn,expect)
            print('exact D5 site covariance PASS',flush=True)
        X=exact_projection(U0,Y)
        xs.append(X)
        print('t',t,'max entry',int(np.abs(X).max()),flush=True)
    A,B,cm=interpolate(np.stack(xs))
    print('coefficient maxima',cm,'target pair maxima',int(np.abs(A).max()),int(np.abs(B).max()),flush=True)
    kernel=exact_rank_and_kernel(A,B,2)
    print('target exact rank 2 of 33; exact left kernel 31, all-column annihilation PASS',flush=True)
    data_dir=Path(__file__).resolve().parent.parent/'data'
    pair_path=data_dir/'five_copy_ico_g34_pair7.npz'
    kernel_path=data_dir/'five_copy_ico_g34_kernel7.json'
    assert pair_path.exists()==kernel_path.exists(), 'partial committed kernel artifacts'
    if pair_path.exists():
        old=np.load(pair_path)
        assert np.array_equal(old['A'],A) and np.array_equal(old['B'],B)
        assert json.loads(kernel_path.read_text())==kernel
        print('committed compact artifacts reproduced exactly',flush=True)
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output_dir/'five_copy_ico_g34_pair7.npz',A=A,B=B)
        with open(args.output_dir/'five_copy_ico_g34_kernel7.json','w') as f:json.dump(kernel,f)
    print('PASS',flush=True)


if __name__=='__main__':main()
