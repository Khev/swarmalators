"""
compute_SK.py

Computes S(K) for the linear-parabolic swarmalator model across N values.

Order parameters (time-averaged over last 10% of run):
  S+, S-   : |<exp(i(phi ± theta))>|  (unweighted, S+ >= S- by convention)
  S+1, S-1 : |<r exp(i(phi ± theta))>|  (radially weighted)
  v        : mean speed = <sqrt(vx^2 + vy^2 + vtheta^2)>_i, time-averaged

Parameters:
  J=1, sigma=2, K from -5 to 0 (step 0.25), T=1000, dt=0.1
  N = 10, 50, 100, 500, 1000, 5000, 10000

Usage:
  python3 -u compute_SK.py                    # all N
  python3 -u compute_SK.py --N 100 1000       # specific N values
  python3 -u compute_SK.py --workers 8        # set n_workers

Output:
  SK_results_N{N}.txt  — columns: K, S+, S-, S+1, S-1, v
  SK_results_N{N}.npz  — same data as numpy arrays
"""
import numpy as np
import argparse
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── model parameters ──────────────────────────────────────────────────────────
J_DEFAULT     = 1.0
SIGMA_DEFAULT = 2.0
T_DEFAULT     = 1000.0
DT_DEFAULT    = 0.1
KS_DEFAULT    = np.arange(-5.0, 0.25, 0.25)
N_DEFAULT     = [10, 50, 100, 500, 1000, 5000, 10000]
OUTDIR        = '.'   # save in current directory


# ── RHS (vectorized numpy, float64) ──────────────────────────────────────────
def rhs(z, J, K, sigma, N):
    x, y, th = z[:N], z[N:2*N], z[2*N:]
    xd = x[:,None] - x
    yd = y[:,None] - y
    td = th[:,None] - th
    dist2 = xd**2 + yd**2
    np.fill_diagonal(dist2, np.inf)
    inv2 = 1.0 / dist2
    np.fill_diagonal(inv2, 0.0)
    parab = np.maximum(0.0, 1.0 - dist2 / sigma**2)
    factor = (1.0 + J * np.cos(td)) - inv2
    vx = np.sum(-xd * factor, axis=1) / N
    vy = np.sum(-yd * factor, axis=1) / N
    vt = np.sum(-K  * np.sin(td) * parab, axis=1) / N
    return np.concatenate([vx, vy, vt])


def rk4_step(z, dt, J, K, sigma, N):
    k1 = rhs(z, J, K, sigma, N)
    k2 = rhs(z + 0.5*dt*k1, J, K, sigma, N)
    k3 = rhs(z + 0.5*dt*k2, J, K, sigma, N)
    k4 = rhs(z + dt*k3,     J, K, sigma, N)
    return z + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ── order parameters ──────────────────────────────────────────────────────────
def compute_ops(z, vel, N):
    """Compute all order parameters from state z and velocity vel."""
    x, y, th = z[:N], z[N:2*N], z[2*N:]
    vx, vy, vt = vel[:N], vel[N:2*N], vel[2*N:]

    r   = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    # unweighted
    Wp = np.mean(np.exp(1j * (phi + th)))
    Wm = np.mean(np.exp(1j * (phi - th)))
    Sp_raw = abs(Wp)
    Sm_raw = abs(Wm)

    # radially weighted
    Wp1 = np.mean(r * np.exp(1j * (phi + th)))
    Wm1 = np.mean(r * np.exp(1j * (phi - th)))
    Sp1_raw = abs(Wp1)
    Sm1_raw = abs(Wm1)

    # mean speed
    v = np.mean(np.sqrt(vx**2 + vy**2 + vt**2))

    return Sp_raw, Sm1_raw, Sp1_raw, Sm1_raw, v


def run_one(args):
    """Run one (K, N) simulation and return all order parameters."""
    K, N, T, dt, J, sigma, seed = args

    np.random.seed(seed)
    r0   = np.sqrt(np.random.uniform(0, 2.25, N))
    phi0 = np.random.uniform(0, 2*np.pi, N)
    z = np.concatenate([
        r0 * np.cos(phi0),
        r0 * np.sin(phi0),
        np.random.uniform(-np.pi, np.pi, N)
    ])

    NT     = int(T / dt)
    cutoff = int(0.9 * NT)

    # accumulators
    Sp_sum = Sm_sum = Sp1_sum = Sm1_sum = v_sum = 0.0
    count  = 0

    for t in range(NT):
        vel = rhs(z, J, K, sigma, N)   # compute vel before step (for v)
        z   = rk4_step(z, dt, J, K, sigma, N)

        if t >= cutoff:
            x, y, th = z[:N], z[N:2*N], z[2*N:]
            vx, vy, vt = vel[:N], vel[N:2*N], vel[2*N:]

            r   = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)

            Sp_sum  += abs(np.mean(np.exp(1j * (phi + th))))
            Sm_sum  += abs(np.mean(np.exp(1j * (phi - th))))
            Sp1_sum += abs(np.mean(r * np.exp(1j * (phi + th))))
            Sm1_sum += abs(np.mean(r * np.exp(1j * (phi - th))))
            v_sum   += np.mean(np.sqrt(vx**2 + vy**2 + vt**2))
            count   += 1

    Sp  = Sp_sum  / count
    Sm  = Sm_sum  / count
    Sp1 = Sp1_sum / count
    Sm1 = Sm1_sum / count
    v   = v_sum   / count

    # enforce S+ >= S- by swapping if needed
    if Sp < Sm:
        Sp, Sm = Sm, Sp
    if Sp1 < Sm1:
        Sp1, Sm1 = Sm1, Sp1

    return K, Sp, Sm, Sp1, Sm1, v


def run_sweep(N, Ks, T, dt, J, sigma, n_workers, outdir):
    print(f"\n{'='*60}", flush=True)
    print(f"N={N}, {len(Ks)} K values, n_workers={n_workers}", flush=True)
    print(f"{'K':>6} {'S+':>8} {'S-':>8} {'S+1':>8} {'S-1':>8} {'v':>8}", flush=True)
    print("-"*50, flush=True)

    t0   = time.time()
    args = [(K, N, T, dt, J, sigma, 42) for K in Ks]

    results = {}
    partial_path = os.path.join(outdir, f'SK_results_N{N}_partial.txt')

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(run_one, a): a[0] for a in args}
        done = 0
        for f in as_completed(futs):
            K_val, Sp, Sm, Sp1, Sm1, v = f.result()
            results[K_val] = (Sp, Sm, Sp1, Sm1, v)
            done += 1
            print(f"  [{done:2d}/{len(Ks)}] K={K_val:>6.2f}  "
                  f"S+={Sp:.4f}  S-={Sm:.4f}  "
                  f"S+1={Sp1:.4f}  S-1={Sm1:.4f}  v={v:.4f}", flush=True)

            # save partial after every result
            Ks_done = np.sort(list(results.keys()))
            rows = np.array([[K] + list(results[K]) for K in Ks_done])
            np.savetxt(partial_path, rows,
                       header='K S+ S- S+1 S-1 v', fmt='%.6f')

    elapsed = (time.time() - t0) / 60
    print(f"N={N} done in {elapsed:.1f} min", flush=True)

    # final save
    Ks_s = np.sort(list(results.keys()))
    rows = np.array([[K] + list(results[K]) for K in Ks_s])

    txt_path = os.path.join(outdir, f'SK_results_N{N}.txt')
    npz_path = os.path.join(outdir, f'SK_results_N{N}.npz')

    np.savetxt(txt_path, rows, header='K S+ S- S+1 S-1 v', fmt='%.6f')
    np.savez(npz_path,
             K=Ks_s,
             Sp=rows[:,1], Sm=rows[:,2],
             Sp1=rows[:,3], Sm1=rows[:,4],
             v=rows[:,5],
             N=N, J=J, sigma=sigma, T=T, dt=dt)

    print(f"Saved {txt_path}", flush=True)
    print(f"Saved {npz_path}", flush=True)

    # remove partial file now that full file is saved
    if os.path.exists(partial_path):
        os.remove(partial_path)

    return rows


def plot_all(N_list, Ks, J, sigma, outdir):
    """3 x num_N figure: row 1 = S+/S-(K), row 2 = S+1/S-1(K), row 3 = v(K)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Km  = -J * sigma**2
    K_th = np.linspace(Km, 0.5, 500)
    Sp_th = np.sqrt(np.maximum(0, (4/3)*(J + K_th/sigma**2)))

    num_N = len(N_list)
    fig, axes = plt.subplots(num_N, 3, figsize=(12, 3*num_N),
                             sharex=True, sharey='col')
    if num_N == 1:
        axes = axes[None, :]  # ensure 2D

    for row, N in enumerate(N_list):
        path = os.path.join(outdir, f'SK_results_N{N}.npz')
        if not os.path.exists(path):
            print(f"  Skipping N={N} (no data)", flush=True)
            continue
        d = np.load(path)
        K   = d['K']
        Sp  = d['Sp'];  Sm  = d['Sm']
        Sp1 = d['Sp1']; Sm1 = d['Sm1']
        v   = d['v']

        # col 0: S+, S-
        ax = axes[row, 0]
        ax.plot(K, Sp, 'b-o', ms=4, lw=1.5, label='$S_+$')
        ax.plot(K, Sm, 'r-s', ms=4, lw=1.5, label='$S_-$')
        ax.plot(K_th, Sp_th, 'k--', lw=1.5, label='Theory')
        ax.axvline(Km, color='gray', ls=':', lw=1)
        ax.set_ylabel(f'N={N}', fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title('$S_\\pm(K)$', fontsize=12)
            ax.legend(fontsize=7)

        # col 1: S+1, S-1
        ax = axes[row, 1]
        ax.plot(K, Sp1, 'b-o', ms=4, lw=1.5, label='$S_{+1}$')
        ax.plot(K, Sm1, 'r-s', ms=4, lw=1.5, label='$S_{-1}$')
        ax.axvline(Km, color='gray', ls=':', lw=1)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title('$S_{\\pm 1}(K)$', fontsize=12)
            ax.legend(fontsize=7)

        # col 2: v
        ax = axes[row, 2]
        ax.plot(K, v, 'g-^', ms=4, lw=1.5)
        ax.axvline(Km, color='gray', ls=':', lw=1)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title('$v(K)$', fontsize=12)
        if row == num_N - 1:
            for col in range(3):
                axes[row, col].set_xlabel('K', fontsize=11)

    fig.suptitle(f'Swarmalator order parameters: J={J}, σ={sigma}\n'
                 f'$K_m = -J\\sigma^2 = {Km:.0f}$  |  '
                 r'Theory: $S_+=\sqrt{\frac{4}{3}(J+K/\sigma^2)}$',
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(outdir, 'SK_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute S(K) for swarmalator model')
    parser.add_argument('--N',       type=int,   nargs='+', default=N_DEFAULT)
    parser.add_argument('--workers', type=int,   default=5)
    parser.add_argument('--T',       type=float, default=T_DEFAULT)
    parser.add_argument('--dt',      type=float, default=DT_DEFAULT)
    parser.add_argument('--J',       type=float, default=J_DEFAULT)
    parser.add_argument('--sigma',   type=float, default=SIGMA_DEFAULT)
    parser.add_argument('--Kmin',    type=float, default=-5.0)
    parser.add_argument('--Kmax',    type=float, default=0.25)
    parser.add_argument('--Kstep',   type=float, default=0.25)
    parser.add_argument('--outdir',  type=str,   default='.')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    Ks = np.arange(args.Kmin, args.Kmax, args.Kstep)

    print(f"Swarmalator S(K) sweep", flush=True)
    print(f"J={args.J}, sigma={args.sigma}, T={args.T}, dt={args.dt}", flush=True)
    print(f"K: {Ks[0]:.2f} to {Ks[-1]:.2f}, step {args.Kstep}", flush=True)
    print(f"N values: {args.N}", flush=True)
    print(f"n_workers: {args.workers}", flush=True)
    print(f"outdir: {args.outdir}", flush=True)

    for N in args.N:
        run_sweep(N, Ks, args.T, args.dt, args.J, args.sigma,
                  args.workers, args.outdir)

    # plot summary of all completed N values
    plot_all(args.N, Ks, args.J, args.sigma, args.outdir)

    print("\n=== All done! ===", flush=True)

