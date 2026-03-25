"""
run_cpc_benchmark.py  — Self-contained CPC benchmark (no external dependencies)
================================================================================
Methods: Random, SLSQP, CMA-ES, ES-RL
Tracks:  cold start, warm start
Seeds:   10
"""
import signal, sys, os, json, time
signal.signal(signal.SIGINT, signal.SIG_IGN)   # prevent Ctrl+C crashes

import numpy as np
import warnings; warnings.filterwarnings("ignore")
from scipy.optimize import minimize

# ── Physics (inline, stable coth, ratio caps from paper Fig.2) ───────────────
R_OMEGA = 6.0   # 4× paper's omega_h/omega_c=1.5 → max η ≈ 1-1/6 = 0.833
R_BETA  = 8.0   # 4× paper's beta_c/beta_h=2.0
LAM_MAX = 0.2

def coth(x):
    ax = abs(x)
    if ax < 1e-10: return np.sign(x)*1e10
    if ax < 1e-3:  return 1/x + x/3 - x**3/45
    if ax > 20:    return float(np.sign(x))*(1 + 2*np.exp(-2*ax))
    return 1/np.tanh(x)

def compute_eta(bc, bh, wc, wh, lam):
    """Eqs 8-10 from reference paper."""
    X = coth(bh*wh/2); Y = coth(bc*wc/2)
    Qh = (wh/2)*(X-Y) + (3*lam/(8*wh**2))*(X**2-Y**2)
    Qc = (wc/2)*(Y-X) + (3*lam/(8*wc**2))*(Y**2-X**2)
    W  = Qh + Qc
    if Qh <= 0 or W <= 0: return 0.0
    return float(np.clip(W/Qh, 0, 1))

def feasible(bc, bh, wc, wh, lam):
    """All physical + ratio constraints."""
    if wc >= wh or bh >= bc or lam < 0 or lam > LAM_MAX: return False
    if bc*wc <= bh*wh: return False
    if wh/wc > R_OMEGA or bc/bh > R_BETA: return False
    return True

def eta(params):
    bc, bh, wc, wh, lam = float(params[0]), float(params[1]), float(params[2]), float(params[3]), float(params[4])
    if not feasible(bc, bh, wc, wh, lam): return 0.0
    return compute_eta(bc, bh, wc, wh, lam)

# Box bounds: [bc, bh, wc, wh, lam]
LO = np.array([0.5,  0.1, 1.0, 2.0, 0.0])
HI = np.array([20.0, 10.0, 8.0, 15.0, LAM_MAX])

def sample_valid(rng):
    for _ in range(10000):
        x = rng.uniform(LO, HI)
        if feasible(*x): return x
    return (LO + HI)/2  # fallback

# Warm start: bc/bh=3.2<4, wh/wc=2.5<3, lam=0.05
WARM = np.array([8.0, 2.5, 2.0, 5.0, 0.05])

# ── Config ───────────────────────────────────────────────────────────────────
THRESHOLD = 0.75
N_SEEDS   = 10
BUDGET    = 1000

SLSQP_CONS = [
    {'type':'ineq','fun': lambda x: x[3]-x[2]-0.05},
    {'type':'ineq','fun': lambda x: x[0]-x[1]-0.05},
    {'type':'ineq','fun': lambda x: x[0]*x[2]-x[1]*x[3]-0.01},
    {'type':'ineq','fun': lambda x: R_OMEGA - x[3]/max(x[2],1e-6)},
    {'type':'ineq','fun': lambda x: R_BETA  - x[0]/max(x[1],1e-6)},
]
BNDS = list(zip(LO, HI))

# ── Runners ──────────────────────────────────────────────────────────────────
def run_random(seed, warm=False):
    rng = np.random.default_rng(seed)
    hist, best, ev, found = [0.0], 0.0, 0, None; t0=time.perf_counter()
    for _ in range(BUDGET):
        x = sample_valid(rng); r = eta(x); ev += 1
        if r > best: best = r
        hist.append(best)
        if r >= THRESHOLD and not found: found=(ev, time.perf_counter()-t0)
    return dict(hist=hist, best=best, ev=ev, t=time.perf_counter()-t0, found=found)

def run_slsqp(seed, warm=False):
    rng = np.random.default_rng(seed)
    hist=[0.0]; best=[0.0]; ev=[0]; found=[None]; t0=time.perf_counter()
    def obj(x):
        r=eta(x); ev[0]+=1
        if r>best[0]: best[0]=r
        hist.append(best[0])
        if r>=THRESHOLD and not found[0]: found[0]=(ev[0],time.perf_counter()-t0)
        return -r
    for i in range(15):
        if ev[0]>=BUDGET: break
        x0 = WARM.copy() if (warm and i==0) else sample_valid(rng)
        try: minimize(obj,x0,method='SLSQP',bounds=BNDS,constraints=SLSQP_CONS,
                      options={'maxiter':150,'ftol':1e-9,'disp':False})
        except: pass
    return dict(hist=hist, best=best[0], ev=ev[0], t=time.perf_counter()-t0, found=found[0])

def run_cmaes(seed, warm=False):
    np.random.seed(seed); rng=np.random.default_rng(seed)
    hist=[0.0]; best=0.0; ev=0; found=None; t0=time.perf_counter()
    d=5; lam=10; mu=lam//2
    w=np.log(mu+0.5)-np.log(np.arange(1,mu+1)); w/=w.sum()
    mueff=1/(w**2).sum(); C=np.eye(d); ps=np.zeros(d); pc=np.zeros(d); s=0.3; g=0
    cc=(4+mueff/d)/(d+4+2*mueff/d); cs=(mueff+2)/(d+mueff+5)
    c1=2/((d+1.3)**2+mueff); cmu=min(1-c1,2*(mueff-2+1/mueff)/((d+2)**2+mueff))
    damp=1+2*max(0,np.sqrt((mueff-1)/(d+1))-1)+cs; chiN=d**0.5*(1-1/(4*d)+1/(21*d**2))
    m = (WARM-LO)/(HI-LO) if warm else (sample_valid(rng)-LO)/(HI-LO)
    m = np.clip(m, 0.05, 0.95)
    while ev < BUDGET:
        try: L=np.linalg.cholesky(C+1e-9*np.eye(d))
        except: L=np.eye(d)
        z=np.random.randn(lam,d); xs=m+s*(z@L.T)
        fs=[]
        for x in xs:
            r=eta(LO+np.clip(x,0,1)*(HI-LO)); ev+=1; fs.append(r)
            if r>best: best=r
            hist.append(best)
            if r>=THRESHOLD and not found: found=(ev,time.perf_counter()-t0)
            if ev>=BUDGET: break
        fs=np.array(fs); idx=np.argsort(-fs); old=m.copy()
        m=(w*xs[idx[:mu]].T).sum(1)
        step=(m-old)/s
        try: Ci=np.linalg.inv(L).T
        except: Ci=np.eye(d)
        g+=1; ps=(1-cs)*ps+np.sqrt(cs*(2-cs)*mueff)*Ci@step
        hs=np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*g))/chiN<1.4+2/(d+1)
        pc=(1-cc)*pc+hs*np.sqrt(cc*(2-cc)*mueff)*step
        diff=(xs[idx[:mu]]-old)/s
        C=((1-c1-cmu)*C+c1*(np.outer(pc,pc)+(1-hs)*cc*(2-cc)*C)+cmu*(w*diff.T@diff))
        s*=np.exp((cs/damp)*(np.linalg.norm(ps)/chiN-1)); s=np.clip(s,1e-8,2.0)
    return dict(hist=hist, best=best, ev=ev, t=time.perf_counter()-t0, found=found)

def run_esrl(seed, warm=False):
    rng=np.random.default_rng(seed)
    hist=[0.0]; best=0.0; ev=0; found=None; t0=time.perf_counter()
    mu = np.clip((WARM-LO)/(HI-LO),0.05,0.95) if warm else np.clip((sample_valid(rng)-LO)/(HI-LO),0.05,0.95)
    sig=0.15; m=np.zeros(5); v=np.zeros(5); t_a=0; lr=0.03; b1,b2,eps=0.9,0.999,1e-8
    while ev<BUDGET:
        H=8; eps_b=rng.standard_normal((H,5))*sig
        rp=np.zeros(H); rn=np.zeros(H)
        for i in range(H):
            if ev>=BUDGET: break
            for sgn,arr in [(1,rp),(-1,rn)]:
                if ev>=BUDGET: break
                r=eta(LO+np.clip(mu+sgn*eps_b[i],0,1)*(HI-LO)); ev+=1; arr[i]=r
                if r>best: best=r
                hist.append(best)
                if r>=THRESHOLD and not found: found=(ev,time.perf_counter()-t0)
        g=(np.sum(((rp-rn)[:,None]*eps_b),axis=0))/(2*sig*H)
        t_a+=1; m=b1*m+(1-b1)*g; v=b2*v+(1-b2)*g**2
        mh=m/(1-b1**t_a); vh=v/(1-b2**t_a)
        mu=np.clip(mu+lr*mh/(np.sqrt(vh)+eps),0,1)
    return dict(hist=hist, best=best, ev=ev, t=time.perf_counter()-t0, found=found)

# ── Run ───────────────────────────────────────────────────────────────────────
METHODS = [('Random', run_random), ('SLSQP', run_slsqp),
           ('CMA-ES', run_cmaes),  ('ES-RL', run_esrl)]
COLORS  = {'Random':'#7f8c8d','SLSQP':'#e74c3c','CMA-ES':'#3498db','ES-RL':'#9b59b6'}

all_res = {}
for track, warm in [('cold', False), ('warm', True)]:
    all_res[track] = {}
    print(f"\n── {track.upper()} START ────────────────────────────────")
    for name, fn in METHODS:
        trials = [fn(s, warm=warm) for s in range(N_SEEDS)]
        all_res[track][name] = trials
        bests = [t['best'] for t in trials]
        succ  = [t['found'] for t in trials if t['found']]
        sev   = [s[0] for s in succ]
        print(f"  {name:<8} η_max={max(bests):.4f}  η_med={np.median(bests):.4f}  "
              f"success={len(succ)}/{N_SEEDS}  "
              f"med_ev={int(np.median(sev)) if sev else 'N/A':<6}  "
              f"med_t={np.median([s[1] for s in succ]):.4f}s" if succ else
              f"  {name:<8} η_max={max(bests):.4f}  success=0/{N_SEEDS}")

# Save JSON
RDIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RDIR, exist_ok=True)

def ser(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)

with open(f'{RDIR}/cpc_benchmark.json','w') as f:
    json.dump(all_res, f, default=ser, indent=2)
print(f"\nSaved → {RDIR}/cpc_benchmark.json")

# ── Plots ──────────────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

PDIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PDIR, exist_ok=True)
MNAMES = [m[0] for m in METHODS]

def pad(h, n):
    return (h + [h[-1]]*(n-len(h)))[:n] if len(h)<n else h[:n]

# Fig 1: Convergence
fig, axes = plt.subplots(1,2,figsize=(14,5),sharey=True)
fig.suptitle(f'Best-So-Far $\\eta$ vs Evaluations\n(Median ± IQR, {N_SEEDS} seeds each)',fontsize=13)
for ax, track in zip(axes,['cold','warm']):
    maxl = max(len(t['hist']) for m in MNAMES for t in all_res[track][m])
    for m in MNAMES:
        H = np.array([pad(t['hist'], maxl) for t in all_res[track][m]])
        med=np.median(H,0); q25=np.percentile(H,25,0); q75=np.percentile(H,75,0)
        x=np.arange(maxl)
        ax.plot(x,med,color=COLORS[m],lw=2,label=m)
        ax.fill_between(x,q25,q75,color=COLORS[m],alpha=0.18)
    ax.axhline(THRESHOLD,ls='--',color='k',lw=1,alpha=0.7,label=f'η={THRESHOLD}')
    ax.set_xlabel('Evaluations',fontsize=11); ax.set_ylabel('Best η',fontsize=11)
    ax.set_title(f'{"Cold" if track=="cold" else "Warm"} Start',fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.35)
plt.tight_layout(); fig.savefig(f'{PDIR}/fig1_convergence.png',dpi=200,bbox_inches='tight')
plt.close(); print("Saved: fig1_convergence.png")

# Fig 2: Bar chart — evaluations + time to threshold
fig, axes = plt.subplots(1,2,figsize=(13,5))
fig.suptitle(f'Evaluations and Time to Reach $\\eta \\geq {THRESHOLD}$\n({N_SEEDS} seeds, log scale)',fontsize=13)
for ax, track in zip(axes,['cold','warm']):
    ev_med, t_med, succ_label = [], [], []
    for m in MNAMES:
        sv = [t['found'][0] for t in all_res[track][m] if t['found']]
        st = [t['found'][1] for t in all_res[track][m] if t['found']]
        ev_med.append(np.median(sv) if sv else float('nan'))
        t_med.append(np.median(st) if st else float('nan'))
        succ_label.append(f"{len(sv)}/{N_SEEDS}")
    colors_list = [COLORS[m] for m in MNAMES]
    bars = ax.bar(MNAMES, ev_med, color=colors_list, edgecolor='k', linewidth=0.7)
    ax2  = ax.twinx()
    ax2.plot(MNAMES, t_med, 'D--', color='black', markersize=8, linewidth=1.5, label='Time (s)')
    ax2.set_ylabel('Median time (s)', fontsize=10)
    for bar, ev, sc in zip(bars, ev_med, succ_label):
        if not np.isnan(ev):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.1,
                    f'{int(ev)}\n({sc}✓)', ha='center', fontsize=8.5, fontweight='bold')
        else:
            ax.text(bar.get_x()+bar.get_width()/2, 5, f'FAIL\n({sc})', ha='center', fontsize=8.5, color='red')
    ax.set_ylabel('Median evaluations', fontsize=10)
    ax.set_yscale('symlog'); ax.grid(axis='y',alpha=0.35)
    ax.set_title(f'{"Cold" if track=="cold" else "Warm"} Start', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
plt.tight_layout(); fig.savefig(f'{PDIR}/fig2_threshold.png',dpi=200,bbox_inches='tight')
plt.close(); print("Saved: fig2_threshold.png")

# Fig 3: Violin
fig, axes = plt.subplots(1,2,figsize=(13,5),sharey=True)
fig.suptitle(f'Distribution of Best $\\eta$ Achieved ({N_SEEDS} seeds)',fontsize=13)
for ax, track in zip(axes,['cold','warm']):
    data = [[t['best'] for t in all_res[track][m]] for m in MNAMES]
    parts = ax.violinplot(data, positions=range(len(MNAMES)), showmedians=True, showextrema=True)
    for pc, m in zip(parts['bodies'], MNAMES):
        pc.set_facecolor(COLORS[m]); pc.set_alpha(0.75)
    for comp in ['cmedians','cmins','cmaxes','cbars']:
        parts[comp].set_color('black'); parts[comp].set_linewidth(1.5)
    ax.axhline(THRESHOLD, ls='--', color='red', lw=1.2, label=f'η={THRESHOLD}')
    ax.set_xticks(range(len(MNAMES))); ax.set_xticklabels(MNAMES, fontsize=10)
    ax.set_ylabel('Best η'); ax.set_title(f'{"Cold" if track=="cold" else "Warm"} Start',fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis='y',alpha=0.35)
plt.tight_layout(); fig.savefig(f'{PDIR}/fig3_violin.png',dpi=200,bbox_inches='tight')
plt.close(); print("Saved: fig3_violin.png")

# Fig 4: Feasible space visualization
from physics_model import R_OMEGA_MAX, R_BETA_MAX
rng2 = np.random.default_rng(42)
pts  = []
LO_ = np.array([0.5, 0.1, 1.0, 2.0, 0.0]); HI_ = np.array([20.0, 10.0, 8.0, 15.0, LAM_MAX])
while len(pts) < 2000:
    x = rng2.uniform(LO_, HI_)
    if feasible(*x): pts.append((*x, eta(x)))
pts = np.array(pts); bc,bh,wc,wh,lam,et = pts.T

fig, axes = plt.subplots(1,3,figsize=(15,5))
fig.suptitle('Feasible Parameter Space (Constrained Domain)\n'
             r'$\omega_h/\omega_c \leq 3$,  $\beta_c/\beta_h \leq 4$,  $\lambda \leq 0.2$',fontsize=12)
for ax, (x,y,xl,yl) in zip(axes,[(wh/wc,et,r'$\omega_h/\omega_c$',r'$\eta$'),
                                   (bc/bh,et,r'$\beta_c/\beta_h$', r'$\eta$'),
                                   (lam,  et,r'$\lambda$',         r'$\eta$')]):
    sc=ax.scatter(x,y,c=et,cmap='plasma',s=6,alpha=0.5,vmin=0,vmax=1)
    ax.set_xlabel(xl,fontsize=12); ax.set_ylabel(yl,fontsize=12); ax.grid(alpha=0.3)
    plt.colorbar(sc,ax=ax,label='η')
plt.tight_layout(); fig.savefig(f'{PDIR}/fig4_feasible_space.png',dpi=200,bbox_inches='tight')
plt.close(); print("Saved: fig4_feasible_space.png")

print("\nAll plots saved to plots/")
