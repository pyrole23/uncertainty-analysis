"""
ME3112 Metrology and Instrumentation - Uncertainty Analysis Assignment
Problem 1: Cylindricity measurement
Problem 2: Zinc concentration

Each problem is solved two ways:
  (a) GUM analytical approach (law of propagation of uncertainty)
  (b) Monte Carlo simulation (JCGM 101:2008), M = 2e5 trials

Requires: numpy, scipy, matplotlib
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

M = 200_000            # number of Monte Carlo trials
SEED = 42              # fixed seed so results are reproducible
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def shortest_interval(samples, p=0.95):
    """Shortest coverage interval (JCGM 101, clause 7.7)."""
    y = np.sort(samples)
    n = len(y)
    q = int(np.floor(p * n))
    widths = y[q:] - y[:n - q]
    i = np.argmin(widths)
    return y[i], y[i + q]


def symmetric_interval(samples, p=0.95):
    """Probabilistically symmetric coverage interval (2.5 % and 97.5 % quantiles)."""
    return np.percentile(samples, [100 * (1 - p) / 2, 100 * (1 + p) / 2])


def welch_satterthwaite(uc, contributions, dofs):
    """Effective degrees of freedom. contributions = |c_i|*u_i, dofs = nu_i (np.inf for Type B)."""
    denom = sum((ci ** 4) / nu for ci, nu in zip(contributions, dofs) if np.isfinite(nu))
    return np.inf if denom == 0 else uc ** 4 / denom


def print_budget(title, rows, y, uc, nu_eff, k, unit):
    """rows = list of (name, estimate, type, distribution, u_i, c_i)."""
    print("\n" + "=" * 92)
    print(f"GUM UNCERTAINTY BUDGET - {title}")
    print("=" * 92)
    print(f"{'Source':<26}{'Estimate':>12}{'Type':>6}{'PDF':>13}"
          f"{'u(xi)':>12}{'ci':>12}{'|ci|u(xi)':>12}")
    print("-" * 92)
    for name, est, typ, pdf, u, c in rows:
        print(f"{name:<26}{est:>12.5g}{typ:>6}{pdf:>13}{u:>12.4e}{c:>12.4e}{abs(c) * u:>12.4e}")
    print("-" * 92)
    U = k * uc
    print(f"Estimate of output y            = {y:.5f} {unit}")
    print(f"Combined standard uncertainty uc = {uc:.5f} {unit}")
    print(f"Effective degrees of freedom     = {nu_eff:.1f}" if np.isfinite(nu_eff)
          else "Effective degrees of freedom     = infinite")
    print(f"Coverage factor k (95 %)         = {k:.3f}")
    print(f"Expanded uncertainty U           = {U:.5f} {unit}")
    print(f"95 % interval (GUM)              = [{y - U:.5f}, {y + U:.5f}] {unit}")
    return U


def print_mc(title, samples, unit):
    mean = samples.mean()
    sd = samples.std(ddof=1)
    lo_s, hi_s = symmetric_interval(samples)
    lo_sh, hi_sh = shortest_interval(samples)
    print("\n" + "-" * 92)
    print(f"MONTE CARLO RESULTS - {title}  (M = {len(samples):,} trials)")
    print("-" * 92)
    print(f"Mean                              = {mean:.5f} {unit}")
    print(f"Standard uncertainty (std. dev.)  = {sd:.5f} {unit}")
    print(f"95 % symmetric interval           = [{lo_s:.5f}, {hi_s:.5f}] {unit}")
    print(f"95 % shortest interval            = [{lo_sh:.5f}, {hi_sh:.5f}] {unit}")
    print(f"Half-width of symmetric interval  = {(hi_s - lo_s) / 2:.5f} {unit}")
    return mean, sd, (lo_s, hi_s)


def validate_gum(y, U, mc_interval, uc, ndig=2):
    """GUM validation against MCM (JCGM 101, clause 8).
    Numerical tolerance delta = 0.5 * 10^l, where uc is written with ndig significant digits."""
    l = int(np.floor(np.log10(uc))) - (ndig - 1)
    delta = 0.5 * 10 ** l
    d_low = abs((y - U) - mc_interval[0])
    d_high = abs((y + U) - mc_interval[1])
    ok = d_low <= delta and d_high <= delta
    print(f"\nGUM validation: delta = {delta:.2e}, d_low = {d_low:.2e}, d_high = {d_high:.2e}"
          f"  ->  {'GUM VALIDATED' if ok else 'GUM NOT validated'} at this tolerance")


def plot_histogram(samples, y_gum, uc_gum, interval, xlabel, title, fname):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(samples, bins=150, density=True, color="steelblue", alpha=0.65,
            edgecolor="none", label="Monte Carlo PDF")
    x = np.linspace(samples.min(), samples.max(), 600)
    ax.plot(x, stats.norm.pdf(x, y_gum, uc_gum), "r-", lw=2,
            label="GUM (Gaussian, y, $u_c$)")
    ax.axvline(samples.mean(), color="k", ls="-", lw=1.5,
               label=f"MC mean = {samples.mean():.4f}")
    ax.axvline(interval[0], color="green", ls="--", lw=1.5,
               label=f"95 % interval [{interval[0]:.4f}, {interval[1]:.4f}]")
    ax.axvline(interval[1], color="green", ls="--", lw=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    print(f"Histogram saved as {fname}")


# ======================================================================
# PROBLEM 1 : CYLINDRICITY   C = C_ind + d_cal + d_res + d_align
# ======================================================================
def problem1():
    readings = np.array([8.2, 8.5, 8.1, 8.4, 8.3])           # um
    n = len(readings)
    C_ind = readings.mean()
    s = readings.std(ddof=1)
    u_rep = s / np.sqrt(n)                                    # Type A
    nu_rep = n - 1

    U_cal, k_cal = 0.6, 2.0
    u_cal = U_cal / k_cal                                     # Normal
    a_res = 0.05
    u_res = a_res / np.sqrt(3)                                # Uniform, +/-0.05 um
    a_align = 0.30
    u_align = a_align / np.sqrt(6)                            # Symmetric triangular, +/-0.30 um

    # ---- GUM: model is linear, all sensitivity coefficients = 1 ----
    c = [1.0, 1.0, 1.0, 1.0]
    u = [u_rep, u_cal, u_res, u_align]
    dofs = [nu_rep, np.inf, np.inf, np.inf]
    C = C_ind + 0 + 0 + 0
    uc = np.sqrt(sum((ci * ui) ** 2 for ci, ui in zip(c, u)))
    nu_eff = welch_satterthwaite(uc, [ci * ui for ci, ui in zip(c, u)], dofs)
    k = stats.t.ppf(0.975, nu_eff) if np.isfinite(nu_eff) else stats.norm.ppf(0.975)

    rows = [("Repeatability (C_ind)", C_ind, "A", "t (nu=4)", u_rep, c[0]),
            ("Calibration d_cal", 0.0, "B", "Normal", u_cal, c[1]),
            ("Resolution d_res", 0.0, "B", "Uniform", u_res, c[2]),
            ("Alignment d_align", 0.0, "B", "Triangular", u_align, c[3])]
    U = print_budget("PROBLEM 1 (Cylindricity)", rows, C, uc, nu_eff, k, "um")

    # ---- Monte Carlo ----
    # Repeatability: scaled & shifted t-distribution with n-1 dof (JCGM 101, 6.4.9)
    C_ind_s = C_ind + u_rep * rng.standard_t(nu_rep, M)
    d_cal_s = rng.normal(0.0, u_cal, M)
    d_res_s = rng.uniform(-a_res, a_res, M)
    d_align_s = rng.triangular(-a_align, 0.0, a_align, M)
    C_s = C_ind_s + d_cal_s + d_res_s + d_align_s

    mean, sd, interval = print_mc("PROBLEM 1 (Cylindricity)", C_s, "um")
    validate_gum(C, U, interval, uc)
    plot_histogram(C_s, C, uc, interval, "Cylindricity C (µm)",
                   "Problem 1: PDF of cylindricity (Monte Carlo, M = 2×10⁵)",
                   "problem1_cylindricity_histogram.png")
    return C, uc, U, mean, sd, interval


# ======================================================================
# PROBLEM 2 : ZINC CONCENTRATION   c = 1000 * m * P / V
# ======================================================================
def problem2():
    # Purity: Uniform [0.9985, 1.0005]
    P_min, P_max = 0.9985, 1.0005
    P = (P_min + P_max) / 2
    u_P = (P_max - P_min) / (2 * np.sqrt(3))

    # Mass: Gaussian
    m, u_m = 300.75, 0.009                                    # mg

    # Volume = V_fill + dV_rep + dV_temp
    V_min, V_max, V_mode = 149.6, 150.4, 150.0                # Triangular
    V0 = V_mode
    u_Vfill = (V_max - V_min) / 2 / np.sqrt(6)

    range_rep = 0.09                                          # Rectangular, full range 0.09 mL
    a_rep = range_rep / 2                                     # -> +/-0.045 mL
    u_Vrep = a_rep / np.sqrt(3)

    U_temp95 = 0.1                                            # Normal, 0.1 mL at 95 %
    u_Vtemp = U_temp95 / stats.norm.ppf(0.975)                # divide by 1.96

    V = V0
    u_V = np.sqrt(u_Vfill ** 2 + u_Vrep ** 2 + u_Vtemp ** 2)

    # ---- GUM ----
    cZn = 1000 * m * P / V
    c_m = 1000 * P / V                       # dc/dm
    c_P = 1000 * m / V                       # dc/dP
    c_V = -1000 * m * P / V ** 2             # dc/dV (same for each volume component)

    contrib = [c_P * u_P, c_m * u_m, c_V * u_Vfill, c_V * u_Vrep, c_V * u_Vtemp]
    uc = np.sqrt(sum(x ** 2 for x in contrib))
    nu_eff = np.inf                          # all inputs treated as known PDFs
    k = stats.norm.ppf(0.975)

    rows = [("Purity P", P, "B", "Uniform", u_P, c_P),
            ("Mass m (mg)", m, "B", "Gaussian", u_m, c_m),
            ("Volume - filling (mL)", V0, "B", "Triangular", u_Vfill, c_V),
            ("Volume - repeat. (mL)", 0.0, "A", "Rectangular", u_Vrep, c_V),
            ("Volume - temp. (mL)", 0.0, "B", "Normal", u_Vtemp, c_V)]
    U = print_budget("PROBLEM 2 (Zinc concentration)", rows, cZn, uc, nu_eff, k, "mg/L")
    print(f"(Combined u(V) = {u_V:.5f} mL, relative uc = {uc / cZn * 100:.4f} %)")

    # ---- Monte Carlo ----
    P_s = rng.uniform(P_min, P_max, M)
    m_s = rng.normal(m, u_m, M)
    V_s = (rng.triangular(V_min, V_mode, V_max, M)
           + rng.uniform(-a_rep, a_rep, M)
           + rng.normal(0.0, u_Vtemp, M))
    c_s = 1000 * m_s * P_s / V_s

    mean, sd, interval = print_mc("PROBLEM 2 (Zinc concentration)", c_s, "mg/L")
    validate_gum(cZn, U, interval, uc)
    plot_histogram(c_s, cZn, uc, interval, "Zinc concentration $c_{Zn}$ (mg/L)",
                   "Problem 2: PDF of zinc concentration (Monte Carlo, M = 2×10⁵)",
                   "problem2_zinc_histogram.png")
    return cZn, uc, U, mean, sd, interval


# ======================================================================
def summary(name, unit, res):
    y, uc, U, mean, sd, (lo, hi) = res
    print(f"\n{name}")
    print(f"{'':<12}{'Estimate':>14}{'Std. unc.':>14}{'95 % interval':>32}")
    print(f"{'GUM':<12}{y:>14.4f}{uc:>14.4f}{'[' + f'{y - U:.4f}, {y + U:.4f}' + ']':>32}")
    print(f"{'Monte Carlo':<12}{mean:>14.4f}{sd:>14.4f}{'[' + f'{lo:.4f}, {hi:.4f}' + ']':>32}  ({unit})")


if __name__ == "__main__":
    r1 = problem1()
    r2 = problem2()
    print("\n" + "=" * 92)
    print("SUMMARY: GUM vs MONTE CARLO")
    print("=" * 92)
    summary("Problem 1 - Cylindricity", "um", r1)
    summary("Problem 2 - Zinc concentration", "mg/L", r2)
    plt.show()
