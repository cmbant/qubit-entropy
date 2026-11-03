#!/usr/bin/env python3
"""Vector comparison of the three balanced cuts of HS and cluster states.

Layout adapted from an earlier plotting script. Spectra and
entropies are checked from physical state vectors; every chart uses the same
linear scale. Labels are sized for inclusion at the manuscript's 164 mm width.
The public make_cuts_figure(out=None) entry point also supports companion builds.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'paper' / 'figures' if (ROOT.parent / 'paper').is_dir() else ROOT / 'figures'
CUTS = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))

# Blue HS panel; original cool/colourful pairing codes; cream-and-sage cluster panel.
INK='#17202B'; NAVY='#182E65'
BLUE_FRAME='#367DB1'; BLUE_PANEL='#EEF6FD'; BLUE_INFO='#F2F8FE'
SAGE='#426650'; SAGE_FRAME='#829782'; SAGE_PANEL='#FAFBF7'; SAGE_INFO='#F4F7F0'
WHITE='#FFFFFF'; PALE_BLUE='#C1D6E7'; PALE_GREEN='#CFD9C6'
CUT_LEFT=['#1959BC','#158843','#8729B1']
CUT_RIGHT=['#CA2830','#DD790A','#8F751C']
BAR='#5D9CE7'; BAR_EDGE='#235FA9'


def spectra(psi):
    tensor = psi.reshape((2,) * 4)
    rows = []
    for perm in CUTS:
        a = tensor.transpose(perm).reshape(4, 4)
        rows.append(np.sort(np.maximum(np.linalg.eigvalsh(a @ a.conj().T), 0))[::-1])
    return np.asarray(rows)

def state_spectra():
    omega = np.exp(2j * np.pi / 3)
    m = np.zeros(16, complex)
    m[[3, 12, 5, 10, 6, 9]] = [1, 1, omega, omega, omega**2, omega**2]
    m /= np.sqrt(6)
    c = np.zeros(16, complex)
    c[[0, 3, 12, 15]] = [1, 1, 1, -1]
    c /= 2
    lm, lc = spectra(m), spectra(c)
    assert np.allclose(lm, [[.5, 1/6, 1/6, 1/6]]*3, atol=1e-13)
    assert np.allclose(lc, [[.5, .5, 0, 0], [.25]*4, [.25]*4], atol=1e-13)
    assert np.allclose(-np.sum(lm*np.log2(lm), axis=1), 1+.5*np.log2(3))
    assert np.allclose(-np.sum(lc*np.log2(np.where(lc>0,lc,1)), axis=1), [1,2,2])
    return lm, lc


def make_cuts_figure(out=None):
    hs, cl = state_spectra()
    plt.rcParams.update({
        'font.family':'serif', 'font.serif':['STIXGeneral','DejaVu Serif'],
        'mathtext.fontset':'stix', 'pdf.fonttype':42, 'ps.fonttype':42,
        'font.size':10, 'axes.linewidth':0.8,
    })
    fig=plt.figure(figsize=(14, 10.3), facecolor='white')
    base=fig.add_axes([0,0,1,1]); base.set(xlim=(0,1),ylim=(0,1)); base.axis('off')
    check_text=[]

    def box(x,y,w,h,fc=WHITE,ec='#C7D0D3',lw=.9,r=.009,z=2):
        p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',
                         fc=fc,ec=ec,lw=lw,zorder=z)
        base.add_patch(p)
        return p

    def text(x,y,string,size=14,color=INK,weight='normal',ha='center',va='center',
             check=None, **kwargs):
        artist=base.text(x,y,string,fontsize=size,color=color,weight=weight,
                         ha=ha,va=va,zorder=20,**kwargs)
        if check is not None: check_text.append((artist, check, string))
        return artist

    PANELS=[dict(x=.010,w=.480,title='Higuchi–Sudbery state $M_4$',
                 outer=BLUE_PANEL,edge=BLUE_FRAME,inner=BLUE_INFO,
                 inset_edge=PALE_BLUE,headline=NAVY),
            dict(x=.510,w=.480,title='cluster state $C_4$',
                 outer=SAGE_PANEL,edge=SAGE_FRAME,inner=SAGE_INFO,
                 inset_edge=PALE_GREEN,headline=SAGE)]
    PANEL_Y=.095; PANEL_H=.745
    ROW_TOP=.737; ROW_H=.173; ROW_GAP=.014
    PAD=.012; CUT_W=.127; SPEC_W=.199; ENT_W=.114; CELL_GAP=.008
    assert np.isclose(CUT_W+SPEC_W+ENT_W+2*CELL_GAP+2*PAD,.480)
    CUTS=[('12','34',(1,2),(3,4)),('13','24',(1,3),(2,4)),('14','23',(1,4),(2,3))]

    for side,p in enumerate(PANELS):
        x=p['x']; w=p['w']
        box(x,PANEL_Y,w,PANEL_H,p['outer'],p['edge'],1.3,r=.008,z=0)
        text(x+w/2,.792,p['title'],size=26.6,weight='bold',color=p['headline'],
             check=(x+.010,.755,x+w-.010,.835))


    def pair_drawing(cx,cy,nums,color):
        # Ellipses denote sets of two qubits separated by the bipartition.
        base.add_patch(Ellipse((cx,cy),width=.041,height=.086,fill=False,
                               edgecolor=color,lw=1.65,zorder=5))
        ys=[cy+.021, cy-.021]
        # Use scatter markers (sized in points^2) so the qubit markers stay truly circular
        # regardless of the page aspect ratio in vector export.
        base.scatter([cx, cx], ys, s=[560, 560], facecolors='white', edgecolors=INK,
                     linewidths=1.25, zorder=6)
        for yy,n in zip(ys,nums):
            text(cx,yy,str(n),size=15,weight='bold',check=(cx-.0125,yy-.0125,cx+.0125,yy+.0125))


    def draw_cut_cell(p,y,i,cut):
        x=p['x']+PAD; bounds=(x,y,x+CUT_W,y+ROW_H)
        box(x,y,CUT_W,ROW_H,WHITE,p['inset_edge'],lw=.85,r=.008)
        left_lbl,right_lbl,left_nodes,right_nodes=cut
        text(x+.045,y+.145,left_lbl,size=22,weight='bold',color=CUT_LEFT[i])
        text(x+.064,y+.145,'|',size=21)
        text(x+.083,y+.145,right_lbl,size=22,weight='bold',color=CUT_RIGHT[i])
        base.plot([x+.064,x+.064],[y+.027,y+.103],color='#666D70',ls=(0,(3,3)),lw=1.0,zorder=6)
        pair_drawing(x+.032,y+.067,left_nodes,CUT_LEFT[i]); pair_drawing(x+.096,y+.067,right_nodes,CUT_RIGHT[i])


    def draw_spectrum_cell(p,y,values):
        x=p['x']+PAD+CUT_W+CELL_GAP
        box(x,y,SPEC_W,ROW_H,WHITE,p['inset_edge'],lw=.85,r=.008)
        text(x+SPEC_W/2,y+.149,'cut spectrum',size=16,
             check=(x+.004,y+.131,x+SPEC_W-.004,y+.167))
        # Matplotlib Axes are genuine (not bars drawn against a fake vertical scale).
        cax=fig.add_axes([x+.031,y+.033,SPEC_W-.045,.094],zorder=5)
        cax.set_facecolor('none')
        cax.bar(range(1,5),values,width=.66,color=BAR,edgecolor=BAR_EDGE,lw=.95,zorder=2)
        cax.set_xlim(.46,4.54); cax.set_ylim(0,.575)
        cax.spines[['top','right']].set_visible(False)
        for k in ('left','bottom'): cax.spines[k].set_color(INK);cax.spines[k].set_linewidth(.9)
        cax.set_xticks([1,2,3,4],[r'$\lambda_1$',r'$\lambda_2$',r'$\lambda_3$',r'$\lambda_4$'])
        # Identical ticks and limits allow direct comparison across all six cuts.
        yticks=[0,.25,.5]
        ylabels=['0',r'$1/4$',r'$1/2$']
        cax.set_yticks(yticks,ylabels)
        cax.tick_params(axis='x',direction='out',length=2.2,pad=1.5,labelsize=14)
        cax.tick_params(axis='y',direction='out',length=2.2,pad=2.2,labelsize=13)
        for pos,value in enumerate(values,1):
            if np.isclose(value,0): lab='0'
            elif np.isclose(value,.5): lab=r'$1/2$'
            elif np.isclose(value,.25):lab=r'$1/4$'
            else:lab=r'$1/6$'
            cax.text(pos, value+.026 if value else .029,lab,ha='center',va='bottom',fontsize=13.5)
        return cax


    def draw_entropy_cell(p,y,side,i):
        x=p['x']+PAD+CUT_W+CELL_GAP+SPEC_W+CELL_GAP
        box(x,y,ENT_W,ROW_H,WHITE,p['inset_edge'],lw=.85,r=.008)
        if side==0:
            text(x+ENT_W/2,y+.105,r'$S=1+\frac{1}{2}\log_2 3$',size=15.4,
                 check=(x+.002,y+.087,x+ENT_W-.002,y+.125))
            text(x+ENT_W/2,y+.068,r'$\approx 1.792\ \mathrm{bits}$',size=14.2,
                 check=(x+.002,y+.049,x+ENT_W-.002,y+.087))
        else:
            s=r'$S=1\ \mathrm{bit}$' if i==0 else r'$S=2\ \mathrm{bits}$'
            text(x+ENT_W/2,y+.087,s,size=16.8,
                 check=(x+.002,y+.060,x+ENT_W-.002,y+.117))


    for side,p in enumerate(PANELS):
        for i,cut in enumerate(CUTS):
            y=ROW_TOP-ROW_H-i*(ROW_H+ROW_GAP)
            draw_cut_cell(p,y,i,cut)
            draw_spectrum_cell(p,y,hs[i] if side==0 else cl[i])
            draw_entropy_cell(p,y,side,i)

    # One-line summary of the mean entropy, inside each panel.
    for side,p in enumerate(PANELS):
        x=p['x']+PAD; w=p['w']-2*PAD
        box(x,.107,w,.068,p['inner'],p['inset_edge'],lw=.9,r=.008)
        if side==0:
            text(x+w/2,.141,r'$E(M_4)=1+\frac{1}{2}\log_2 3\;\approx\;1.792$',size=23.0,
                 check=(x+.004,.110,x+w-.004,.172))
        else:
            text(x+w/2,.141,r'$E(C_4)=\frac{1+2+2}{3}=\frac{5}{3}\approx1.667$',size=23.0,
                 check=(x+.004,.110,x+w-.004,.172))

    # Programmatically check all sensitive text has a footprint contained in its cell.
    fig.canvas.draw()
    errors=[]
    for artist,(x0,y0,x1,y1),label in check_text:
        bb=artist.get_window_extent(fig.canvas.get_renderer()).transformed(base.transData.inverted())
        if bb.x0 < x0-1e-3 or bb.x1>x1+1e-3 or bb.y0<y0-1e-3 or bb.y1>y1+1e-3:
            errors.append((label,tuple(round(z,4) for z in (bb.x0,bb.y0,bb.x1,bb.y1)),(x0,y0,x1,y1)))
    if errors:
        print('BOUNDING BOX WARNINGS:')
        for e in errors:print(e)
        raise RuntimeError('Content overflows: adjust the relevant font size or cell')

    path = Path(out) if out is not None else OUT / 'fig_cuts.pdf'
    path.parent.mkdir(parents=True, exist_ok=True)
    # Crop to the panels: the space above and below them is unused.
    crop=matplotlib.transforms.Bbox([[0,.087*10.3],[14,.848*10.3]])
    fig.savefig(path, facecolor='white', bbox_inches=crop,
                metadata={'CreationDate':None, 'ModDate':None})
    plt.close(fig)
    return path


if __name__ == '__main__':
    print('wrote', make_cuts_figure())
