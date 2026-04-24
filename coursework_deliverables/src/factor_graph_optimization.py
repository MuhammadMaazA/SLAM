#!/usr/bin/env python3
"""
Synthetic 2-D pose-graph demo (reference only). Q3d uses q3_lidar_slam_complete.py.

Three poses: T0 fixed at origin, optimise T1,T2 with odometry edges 0-1, 1-2
and a slightly wrong loop edge 2-0. Writes data/q3_results/factor_graph_demo_synthetic.png.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
OUT_DIR = os.environ.get('SLAM_OUT', os.path.join(_ROOT, 'data', 'q3_results'))
os.makedirs(OUT_DIR, exist_ok=True)


def T_from_xyt(xyt: np.ndarray) -> np.ndarray:
    th, x, y = xyt
    c, s = np.cos(th), np.sin(th)
    T = np.eye(3)
    T[0, 0], T[0, 1] = c, -s
    T[1, 0], T[1, 1] = s, c
    T[0, 2], T[1, 2] = x, y
    return T


def rel_err(Ti: np.ndarray, Tj: np.ndarray, z: np.ndarray) -> np.ndarray:
    dT = np.linalg.inv(Ti) @ Tj
    th = float(np.arctan2(dT[1, 0], dT[0, 0]))
    dx = dT[0, 2] - z[0]
    dy = dT[1, 2] - z[1]
    dth = th - z[2]
    dth = (dth + np.pi) % (2 * np.pi) - np.pi
    return np.array([dx, dy, dth])


def main() -> None:
    T0 = np.eye(3)
    z01 = np.array([0.0, 1.0, 0.0])
    z12 = np.array([0.0, 1.0, 0.0])
    z20 = np.array([0.0, -2.05, 0.02])  # small loop inconsistency

    def fun(flat: np.ndarray) -> np.ndarray:
        T1 = T_from_xyt(flat[0:3])
        T2 = T_from_xyt(flat[3:6])
        r = []
        r.append(np.sqrt(80.0) * rel_err(T0, T1, z01))
        r.append(np.sqrt(80.0) * rel_err(T1, T2, z12))
        r.append(np.sqrt(400.0) * rel_err(T2, T0, z20))
        return np.concatenate(r)

    # Before: pure odometry (ignore loop misfit) — plot open chain + gap back to origin
    T1_b = T0 @ T_from_xyt(z01)
    T2_b = T1_b @ T_from_xyt(z12)
    before = np.array([[0, 0], [T1_b[0, 2], T1_b[1, 2]], [T2_b[0, 2], T2_b[1, 2]]])

    x0 = np.array([0.0, 1.0, 0.0, 0.0, 2.0, 0.0])
    sol = least_squares(fun, x0, method='lm', max_nfev=200)
    T1_a = T_from_xyt(sol.x[0:3])
    T2_a = T_from_xyt(sol.x[3:6])
    after = np.array([[0, 0], [T1_a[0, 2], T1_a[1, 2]], [T2_a[0, 2], T2_a[1, 2]]])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(before[:, 0], before[:, 1], 'o--', lw=1.5, label='Before (odom only)')
    ax.plot(np.r_[after[:, 0], after[0, 0]], np.r_[after[:, 1], after[0, 1]],
            's-', lw=1.5, label='After GN + loop')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Synthetic factor graph (reference demo)')
    out = os.path.join(OUT_DIR, 'factor_graph_demo_synthetic.png')
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
