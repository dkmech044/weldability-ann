import math
import numpy as np
import pandas as pd

# -----------------------------
# Material data (from Tables 2–3)
# -----------------------------
materials = {
    0: {  # Al–Fe–Fe
        "p1": "A6451P-T4",
        "p2": "SPCC",
        "p3": "SPCC",
        "k1p": 96.2,      # k′ for Interface 1
        "alpha1p": 3.7e-5,
        "k2p": 79.0,      # k″ for Interface 2
        "alpha2p": 2.1e-5,
    },
    1: {  # Al–Cu–Cu
        "p1": "AA1060",
        "p2": "CU1100",
        "p3": "CU1100",
        "k1p": 298.6,
        "alpha1p": 1.05e-4,
        "k2p": 398.0,
        "alpha2p": 1.16e-4,
    },
}

props = {
    "A6451P-T4": dict(
        rho=2.70e3,
        C0=5.32e3,
        S=1.34,
        k=121.0,
        alpha=9.7e-5,
        Tm=855.0,
        sigma_y=260.0,
    ),
    "SPCC": dict(
        rho=7.85e3,
        C0=4.57e3,
        S=1.49,
        k=73.0,
        alpha=2.1e-5,
        Tm=1809.0,
        sigma_y=260.0,
    ),
    "AA1060": dict(
        rho=2.70e3,
        C0=5.35e3,
        S=1.34,
        k=237.0,
        alpha=9.8e-5,
        Tm=933.0,
        sigma_y=70.0,
    ),
    "CU1100": dict(
        rho=8.93e3,
        C0=3.94e3,
        S=1.49,
        k=398.0,
        alpha=1.16e-4,
        Tm=1356.0,
        sigma_y=210.0,
    ),

    # ---- Literature tri-material case: 5A06 Al / 3003 Al / 321 SS ----
    # NOTE: k (W/m-K) and alpha (m^2/s) are representative room-temperature values used for screening.
    "Al_5A06": dict(
        rho=2.73e3,
        C0=5.240e3,
        S=1.40,
        k=117.0,
        alpha=9.0e-5,
        Tm=893.0,
        sigma_y=167.0,
    ),
    "Al_3003": dict(
        rho=2.73e3,
        C0=5.240e3,
        S=1.40,
        k=193.0,
        alpha=9.0e-5,
        Tm=930.0,
        sigma_y=43.0,
    ),
    "SS_321": dict(
        rho=7.80e3,
        C0=4.578e3,
        S=1.33,
        k=16.0,
        alpha=4.2e-6,
        Tm=1800.0,
        sigma_y=371.0,
    ),
}


# -----------------------------
# Helper: effective interface thermal properties (harmonic mean)
# -----------------------------
def _harmonic_mean(a, b):
    a = float(a); b = float(b)
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)

# Add tri-material literature case as an additional material flag (for case studies / stress tests).
# This DOES NOT change the training dataset generation below (which remains focused on the two validated stacks).
materials[2] = {  # Al–Al–SS (5A06 / 3003 / 321SS)
    "p1": "Al_5A06",
    "p2": "Al_3003",
    "p3": "SS_321",
    # Effective interface properties used in the 1D moving-source temperature metric:
    "k1p": _harmonic_mean(props["Al_5A06"]["k"],  props["Al_3003"]["k"]),
    "alpha1p": _harmonic_mean(props["Al_5A06"]["alpha"], props["Al_3003"]["alpha"]),
    "k2p": _harmonic_mean(props["Al_3003"]["k"], props["SS_321"]["k"]),
    "alpha2p": _harmonic_mean(props["Al_3003"]["alpha"], props["SS_321"]["alpha"]),
}

# -----------------------------
# Design grids (your ranges)
# -----------------------------
V1_vals    = [600.0, 800.0, 1200.0, 1400.0, 1600.0, 1800.0]
wtop_vals  = [6.0, 8.0, 10.0, 12.0, 14.0]
wbot_vals  = [0.75, 1.0, 2.0, 3.0, 4.0, 5.0]
d23_vals   = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
h_vals     = [0.8, 1.0, 1.5]


# -----------------------------
# Helper: interpolation
# -----------------------------
def _interp_at(t_query, t_array, val_array):
    if t_query <= t_array[0]:
        return float(val_array[0])
    if t_query >= t_array[-1]:
        return float(val_array[-1])
    return float(np.interp(t_query, t_array, val_array))


# -----------------------------
# Dynamic β1(t) integration
# -----------------------------
def integrate_beta1_dynamic(U_p1, d12_m, w_top_m, w_bot_m, h1_m, h2_m,
                            U_s1, U_s2, beta1_0, n_steps=400):
    """
    Implements Eq. (1) and Eq. (15) with β1(t) varying as the
    collision front x1(t) advances along the wedge.

    t1* = 2 h1 / (Us1 cos β1_0)
    t2* = 2 h2 / (Us2 cos β1_0)
    t*  = min(t1*, t2*)

    Also defines:
      t2_ret = 2 h2 / (Us2 cos β1_0)
      t1_ret = 2 h1 / (Us1 cos β1_0)

    Returns:
        t_star, t2_ret, t1_ret, times[], x1_array[], beta1_array[]
    """
    cosb1_0 = math.cos(beta1_0)
    if cosb1_0 <= 0.0:
        cosb1_0 = 1e-6

    t1_star = 2.0 * h1_m / (U_s1 * cosb1_0)
    t2_star = 2.0 * h2_m / (U_s2 * cosb1_0)
    t_star  = min(t1_star, t2_star)

    t2_ret = 2.0 * h2_m / (U_s2 * cosb1_0)
    t1_ret = 2.0 * h1_m / (U_s1 * cosb1_0)

    t_end = max(t_star, t2_ret, t1_ret)
    if t_end <= 0.0:
        return t_star, t2_ret, t1_ret, np.array([0.0]), np.array([0.0]), np.array([beta1_0])

    times = np.linspace(0.0, t_end, n_steps + 1)
    dt    = times[1] - times[0]

    x1 = 0.0
    beta1 = beta1_0

    x1_hist = [x1]
    beta1_hist = [beta1]

    for _ in range(n_steps):
        # Vc12(t) from Eq. (15): Up1 / [2 sin(β1/2)]
        sin_half = math.sin(0.5 * beta1)
        if abs(sin_half) < 1e-8:
            sin_half = 1e-8
        Vc12 = U_p1 / (2.0 * sin_half)

        x1 = x1 + Vc12 * dt
        if x1 > w_bot_m:
            x1 = w_bot_m

        denom = w_top_m - x1
        if denom <= 0.0:
            denom = 1e-9
        beta1 = math.atan2(d12_m, denom)

        x1_hist.append(x1)
        beta1_hist.append(beta1)

    return t_star, t2_ret, t1_ret, times, np.array(x1_hist), np.array(beta1_hist)


# -----------------------------
# Dynamic β2(τ) integration
# -----------------------------
def integrate_beta2_dynamic(U_p23, d12_m, d23_m, w_top_m, w_bot_m, h3_m,
                            beta2_0, C03, S3, n_steps=400):
    """
    Implements Eq. (31) and the dynamics used in Eqs. (32)–(35):

    t_delay = d23 / Vimpact (Eq. 32)
    t3*     = t_delay + 2 h3 / (Us3 cos β2_0) (Eq. 34)
    with Us3 = C03 + S3 Up23, Vimpact = 2 Up23.

    Returns:
        t3_star, times[], x3_array[], beta2_array[]
    """
    Vimpact23 = 2.0 * U_p23
    if Vimpact23 <= 0.0:
        return 0.0, np.array([0.0]), np.array([0.0]), np.array([beta2_0])

    t_delay = d23_m / Vimpact23
    U_s3    = C03 + S3 * U_p23

    cosb2_0 = math.cos(beta2_0)
    if cosb2_0 <= 0.0:
        cosb2_0 = 1e-6

    t3_star = t_delay + 2.0 * h3_m / (U_s3 * cosb2_0)
    if t3_star <= 0.0:
        return t3_star, np.array([0.0]), np.array([0.0]), np.array([beta2_0])

    times = np.linspace(0.0, t3_star, n_steps + 1)
    dt    = times[1] - times[0]

    x3 = 0.0
    beta2 = beta2_0

    x3_hist = [x3]
    beta2_hist = [beta2]

    for _ in range(n_steps):
        sin_half = math.sin(0.5 * beta2)
        if abs(sin_half) < 1e-8:
            sin_half = 1e-8
        Vc23 = U_p23 / (2.0 * sin_half)

        x3 = x3 + Vc23 * dt
        if x3 > w_bot_m:
            x3 = w_bot_m

        denom = w_top_m - x3
        if denom <= 0.0:
            denom = 1e-9
        beta2 = math.atan2(d12_m + d23_m, denom)

        x3_hist.append(x3)
        beta2_hist.append(beta2)

    return t3_star, times, np.array(x3_hist), np.array(beta2_hist)


# -----------------------------
# Forward model: full §2.3
# -----------------------------
def forward_model_v2(material_flag, V1, w_top, w_bot, d12, d23,
                     h1, h2, h3, f_heat=0.9, big_T=1e9):
    """
    Implements Section 2.3 (Interfaces 1, 2, and §2.3.4) in code form.

    Returns dict with:
      beta1_deg, Vc12, Vc23,
      T_Interface1, T_Interface2,
      Clearance_OK, Weldable, Vimpact23, Crack_OK
    """

    # mm -> m
    w_top_m = w_top / 1000.0
    w_bot_m = w_bot / 1000.0
    d12_m   = d12   / 1000.0
    d23_m   = d23   / 1000.0
    h1_m    = h1    / 1000.0
    h2_m    = h2    / 1000.0
    h3_m    = h3    / 1000.0

    mat = materials[material_flag]
    p1  = props[mat["p1"]]
    p2  = props[mat["p2"]]
    p3  = props[mat["p3"]]

    rho1, C01, S1, k1, alpha1, Tm1, sig_y1 = p1["rho"], p1["C0"], p1["S"], p1["k"], p1["alpha"], p1["Tm"], p1["sigma_y"]
    rho2, C02, S2, k2, alpha2, Tm2, sig_y2 = p2["rho"], p2["C0"], p2["S"], p2["k"], p2["alpha"], p2["Tm"], p2["sigma_y"]
    rho3, C03, S3, k3, alpha3, Tm3, sig_y3 = p3["rho"], p3["C0"], p3["S"], p3["k"], p3["alpha"], p3["Tm"], p3["sigma_y"]

    k1p, alpha1p = mat["k1p"], mat["alpha1p"]
    k2p, alpha2p = mat["k2p"], mat["alpha2p"]

    # -------- Interface 1 initial geometry (Eq. 1 at x1 = 0) --------
    beta1_0 = math.atan2(d12_m, w_top_m)
    cosb1_0 = math.cos(beta1_0)
    if cosb1_0 <= 0.0:
        cosb1_0 = 1e-6

    # -------- Two-sided R-H at Interface 1 (Eqs. 3–8) --------
    V1n = V1 * cosb1_0

    a_q = rho1 * S1 - rho2 * S2
    b_q = rho1 * C01 + rho2 * C02 + 2.0 * rho2 * S2 * V1n
    c_q = -(rho2 * C02 * V1n + rho2 * S2 * V1n**2)

    disc = b_q * b_q - 4.0 * a_q * c_q
    # Robust solve for U_p1 (handles degenerate case a_q≈0, e.g., similar/identical materials)
    if abs(a_q) < 1e-16:
        if abs(b_q) < 1e-16:
            return {
                "beta1_deg": math.degrees(beta1_0),
                "Vc12": 0.0,
                "Vc23": 0.0,
                "T_Interface1": big_T,
                "T_Interface2": big_T,
                "Clearance_OK": 0,
                "Weldable": 0,
                "Vimpact23": 0.0,
                "Crack_OK": 0,
            }
        U_p1 = -c_q / b_q
    else:
        if disc < 0.0:
            return {
                "beta1_deg": math.degrees(beta1_0),
                "Vc12": 0.0,
                "Vc23": 0.0,
                "T_Interface1": big_T,
                "T_Interface2": big_T,
                "Clearance_OK": 0,
                "Weldable": 0,
                "Vimpact23": 0.0,
                "Crack_OK": 0,
            }
        U_p1 = (-b_q + math.sqrt(disc)) / (2.0 * a_q)

    U_p2 = V1n - U_p1
    if U_p1 <= 0.0 or U_p2 <= 0.0:
        return {
            "beta1_deg": math.degrees(beta1_0),
            "Vc12": 0.0,
            "Vc23": 0.0,
            "T_Interface1": big_T,
            "T_Interface2": big_T,
            "Clearance_OK": 0,
            "Weldable": 0,
            "Vimpact23": 0.0,
            "Crack_OK": 0,
        }

    # Shock velocities (Eqs. 9–10)
    U_s1 = C01 + S1 * U_p1
    U_s2 = C02 + S2 * U_p2

    # -------- Dynamic β1(t) & x1(t) --------
    t_star, t2_ret, t1_ret, times1, x1_arr, beta1_arr = integrate_beta1_dynamic(
        U_p1, d12_m, w_top_m, w_bot_m, h1_m, h2_m, U_s1, U_s2, beta1_0, n_steps=400
    )

    beta1_tstar = _interp_at(t_star,   times1, beta1_arr)
    x1_tstar    = _interp_at(t_star,   times1, x1_arr)

    # Effective Vc12 from front advance
    if t_star > 0.0:
        Vc12_eff = x1_tstar / t_star
    else:
        sin_half = math.sin(0.5 * beta1_0)
        if abs(sin_half) < 1e-8:
            sin_half = 1e-8
        Vc12_eff = U_p1 / (2.0 * sin_half)

    # -------- Interface 1 melt zone & clearance (Eqs. 16–18) --------
    x_melt1 = 2.0 * h1_m * math.tan(beta1_tstar)
    b1_max  = x1_tstar - x_melt1
    clearance_ok_1 = (b1_max > 0.0) and (Vc12_eff > 0.0)

    # -------- Interface 1 temperature (Eqs. 19, 21, 22, 23) --------
    if not clearance_ok_1:
        T_int1 = big_T
    else:
        Q1 = 0.5 * f_heat * rho1 * h1_m * V1**2 * Vc12_eff
        B1 = (b1_max * Vc12_eff) / (2.0 * alpha1p)
        if B1 <= 0.0:
            T_int1 = big_T
        else:
            T_int1 = (Q1 * b1_max) / (k1p * math.sqrt(math.pi * B1))

    Tm_crit1 = min(Tm1, Tm2)
    overheat1 = T_int1 > Tm_crit1

    # =========================================================
    # Interface 2: Deceleration & dynamic β2(τ)
    # =========================================================
    if U_p2 <= 0.0:
        Vimpact23 = 0.0
        Vc23_eff  = 0.0
        T_int2    = big_T
        clearance_ok_2 = False
    else:
        # Eqs. (26–30) for deceleration
        sigma_resist = 0.65 * sig_y2 * 1e6
        a_dec = sigma_resist / (rho2 * h2_m)

        rad = 1.0 - 2.0 * a_dec * d23_m / (U_p2**2)
        if rad <= 0.0:
            Vimpact23 = 0.0
            Vc23_eff  = 0.0
            T_int2    = big_T
            clearance_ok_2 = False
        else:
            Vimpact23 = U_p2 * math.sqrt(rad)
            if Vimpact23 <= 0.0:
                Vc23_eff  = 0.0
                T_int2    = big_T
                clearance_ok_2 = False
            else:
                beta2_0 = math.atan2(d12_m + d23_m, w_top_m)

                U_p23 = 0.5 * Vimpact23
                t3_star, times2, x3_arr, beta2_arr = integrate_beta2_dynamic(
                    U_p23, d12_m, d23_m, w_top_m, w_bot_m, h3_m, beta2_0, C03, S3, n_steps=400
                )

                beta2_t3 = _interp_at(t3_star, times2, beta2_arr)
                x3_t3    = _interp_at(t3_star, times2, x3_arr)

                if t3_star > 0.0:
                    Vc23_eff = x3_t3 / t3_star
                else:
                    sin_half2 = math.sin(0.5 * beta2_0)
                    if abs(sin_half2) < 1e-8:
                        sin_half2 = 1e-8
                    Vc23_eff = U_p23 / (2.0 * sin_half2)

                # Interface 2 melt-zone & clearance (Eqs. 36–38)
                x_melt2 = 2.0 * h2_m * math.tan(beta2_t3)
                b2_max  = x3_t3 - x_melt2
                clearance_ok_2 = (b2_max > 0.0) and (Vc23_eff > 0.0)

                # Interface 2 temperature (Eqs. 39–43)
                if not clearance_ok_2:
                    T_int2 = big_T
                else:
                    Q2 = 0.5 * f_heat * rho2 * h2_m * Vimpact23**2 * Vc23_eff
                    B2 = (b2_max * Vc23_eff) / (2.0 * alpha2p)
                    if B2 <= 0.0:
                        T_int2 = big_T
                    else:
                        T_int2 = (Q2 * b2_max) / (k2p * math.sqrt(math.pi * B2))

    Tm_crit2 = min(Tm2, Tm3)
    overheat2 = T_int2 > Tm_crit2

    # =========================================================
    # Recursive shock reflections (Eqs. 44–49)
    # =========================================================
    cosb1_0 = math.cos(beta1_0)
    if cosb1_0 <= 0.0:
        cosb1_0 = 1e-6

    t2_return = 2.0 * h2_m / (U_s2 * cosb1_0)  # Eq. 44
    t1_return = 2.0 * h1_m / (U_s1 * cosb1_0)  # Eq. 47

    beta1_t2 = _interp_at(t2_return, times1, beta1_arr)
    x1_t2    = _interp_at(t2_return, times1, x1_arr)
    beta1_t1 = _interp_at(t1_return, times1, beta1_arr)
    x1_t1    = _interp_at(t1_return, times1, x1_arr)

    # b3, b4 from melt-zone length analogue: 2 h1 tan β1(return)
    b3 = 2.0 * h1_m * math.tan(beta1_t2)
    b4 = 2.0 * h1_m * math.tan(beta1_t1)

    # Eq. 46: x1(t2_return) - b3 ≥ 2 h1 tan β1(t2_return)
    crack_fail_2 = (x1_t2 - b3) >= (2.0 * h1_m * math.tan(beta1_t2))
    # Eq. 49: x1(t_return) ≥ 2 h1 tan β1(t_return)
    crack_fail_1 = x1_t1 >= (2.0 * h1_m * math.tan(beta1_t1))

    crack_ok = not (crack_fail_1 or crack_fail_2)

    # =========================================================
    # Aggregate weldability (MAIN physics criteria only)
    # =========================================================
    clearance_ok = int(clearance_ok_1 and clearance_ok_2)

    weldable = int(
        clearance_ok and
        (not overheat1) and
        (not overheat2) and
        (Vimpact23 > 0.0)
    )

    beta1_rep = beta1_tstar

    return {
        "beta1_deg": math.degrees(beta1_rep),
        "Vc12": Vc12_eff,
        "Vc23": Vc23_eff,
        "T_Interface1": T_int1,
        "T_Interface2": T_int2,
        "Clearance_OK": clearance_ok,
        "Weldable": weldable,
        "Vimpact23": Vimpact23,
        "Crack_OK": int(crack_ok),
    }


# -----------------------------
# Dataset generation (full grid)
# -----------------------------
def generate_dataset():
    rows = []
    for material_flag in [0, 1]:  # 0 = Al–Fe–Fe, 1 = Al–Cu–Cu
        d12 = 3.0 if material_flag == 0 else 2.0
        for V1 in V1_vals:
            for w_top in wtop_vals:
                for w_bot in wbot_vals:
                    if w_bot > w_top:
                        continue
                    for d23 in d23_vals:
                        for h1 in h_vals:
                            for h2 in h_vals:
                                for h3 in h_vals:
                                    res = forward_model_v2(
                                        material_flag, V1, w_top, w_bot, d12, d23,
                                        h1, h2, h3
                                    )
                                    rows.append(dict(
                                        Material_Flag=material_flag,
                                        **{
                                            "V1 (m/s)": V1,
                                            "w_top (mm)": w_top,
                                            "w_bot (mm)": w_bot,
                                            "d12 (mm)": d12,
                                            "d23 (mm)": d23,
                                            "h1 (mm)": h1,
                                            "h2 (mm)": h2,
                                            "h3 (mm)": h3,
                                        },
                                        beta1_deg=res["beta1_deg"],
                                        Vc12=res["Vc12"],
                                        Vc23=res["Vc23"],
                                        T_Interface1=res["T_Interface1"],
                                        T_Interface2=res["T_Interface2"],
                                        Clearance_OK=res["Clearance_OK"],
                                        Weldable=res["Weldable"],
                                        Vimpact23=res["Vimpact23"],
                                        Crack_OK=res["Crack_OK"],
                                    ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset()
    print("Dataset shape:", df.shape)
    df.to_csv("ann_dataset_full_v23.csv", index=False)
    print("Saved ann_dataset_full_v23.csv")
