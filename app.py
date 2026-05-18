from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
import math, json

app = FastAPI(title="PetroCalc Pro")

# ═══════════════════════════════════════════════════════════════
#  PHYSICS CORE
# ═══════════════════════════════════════════════════════════════

def beggs_brill(
    q_l_m3d: float,   # liquid flow rate m3/day
    q_g_m3d: float,   # gas flow rate m3/day (at surface)
    d_m: float,       # pipe inner diameter, m
    rho_l: float,     # liquid density kg/m3
    rho_g: float,     # gas density kg/m3
    mu_l: float,      # liquid viscosity Pa·s
    mu_g: float,      # gas viscosity Pa·s
    sigma: float,     # surface tension N/m
    theta_deg: float, # pipe inclination angle from horizontal, degrees (90=vertical)
    depth_m: float,   # total depth m
    n_steps: int = 50
) -> dict:
    """
    Beggs & Brill (1973) multiphase flow pressure drop correlation.
    Returns pressure profile vs depth.
    """
    g = 9.81
    theta = math.radians(theta_deg)
    A = math.pi * d_m**2 / 4.0

    q_l = q_l_m3d / 86400.0   # m3/s
    q_g = q_g_m3d / 86400.0

    v_sl = q_l / A             # superficial liquid velocity
    v_sg = q_g / A             # superficial gas velocity
    v_m  = v_sl + v_sg         # mixture velocity
    lam_l = v_sl / v_m if v_m > 0 else 0  # no-slip holdup

    # ── Flow pattern determination (Beggs & Brill original) ──
    # Froude number
    Fr = v_m**2 / (g * d_m)
    lam_l_safe = max(lam_l, 1e-6)

    # Transition boundaries
    L1 = 316  * lam_l_safe**0.302
    L2 = 0.0009252 * lam_l_safe**(-2.4684)
    L3 = 0.10  * lam_l_safe**(-1.4516)
    L4 = 0.5   * lam_l_safe**(-6.738)

    if lam_l < 0.01 and Fr < L1:
        pattern = "Segregated"
    elif lam_l >= 0.01 and Fr < L2:
        pattern = "Segregated"
    elif lam_l >= 0.01 and Fr > L3 and Fr <= L1:
        pattern = "Transition"
    elif (0.01 <= lam_l < 0.4 and Fr >= L1 and Fr <= L4) or \
         (lam_l >= 0.4 and Fr >= L3 and Fr <= L4):
        pattern = "Intermittent"
    else:
        pattern = "Distributed"

    # ── Holdup at horizontal (H_l0) ──
    coeffs = {
        "Segregated":   (0.980, 0.4846, 0.0868),
        "Intermittent": (0.845, 0.5351, 0.0173),
        "Distributed":  (1.065, 0.5824, 0.0609),
        "Transition":   (0.845, 0.5351, 0.0173),
    }
    a, b, c = coeffs[pattern]
    Hl0 = a * lam_l**b / (Fr**c)
    Hl0 = min(max(Hl0, lam_l), 1.0)

    # ── Inclination correction ──
    incl_coeffs = {
        "Segregated":   {"up": (0.011, -3.768, 3.539, -1.614),
                         "down": (4.700, -0.3692, 0.1244, -0.5056)},
        "Intermittent": {"up": (2.960, 0.3050, -0.4473, 0.0978),
                         "down": (4.700, -0.3692, 0.1244, -0.5056)},
        "Distributed":  {"up": (1.0, 0, 0, 0),
                         "down": (4.700, -0.3692, 0.1244, -0.5056)},
        "Transition":   {"up": (2.960, 0.3050, -0.4473, 0.0978),
                         "down": (4.700, -0.3692, 0.1244, -0.5056)},
    }
    direction = "up" if theta_deg >= 0 else "down"
    e1, e2, e3, e4 = incl_coeffs[pattern][direction]
    Nlv = v_sl * (rho_l / (g * sigma))**0.25
    psi = 1 + e1 * (math.sin(1.8*abs(theta)) - math.sin(1.8*abs(theta))**3 / 3) * Nlv**e2 * Fr**e3 * lam_l**e4
    Hl = Hl0 * psi
    Hl = min(max(Hl, 0.0), 1.0)

    # ── Mixture density ──
    rho_m = rho_l * Hl + rho_g * (1 - Hl)

    # ── Friction factor ──
    rho_ns = rho_l * lam_l + rho_g * (1 - lam_l)  # no-slip density
    mu_m   = mu_l ** lam_l * mu_g ** (1 - lam_l)   # no-slip viscosity
    Re = rho_ns * v_m * d_m / mu_m if mu_m > 0 else 1e5
    # Moody friction (Colebrook simplified for smooth pipe)
    if Re < 2300:
        fn = 64 / Re
    else:
        fn = (0.0055 * (1 + (20000 / d_m + 1e6 / Re)**(1/3)))
    # B&B friction ratio
    y = lam_l / (Hl**2 + 1e-9)
    if y > 1.0 and y < 1.2:
        s = math.log(2.2*y - 1.2)
    else:
        lny = math.log(y) if y > 0 else -1
        s = lny / (-0.0523 + 3.182*lny - 0.8725*lny**2 + 0.01853*lny**4)
    ftp = fn * math.exp(s)

    # ── Pressure gradient components ──
    dpdz_elev   = rho_m * g * math.sin(theta)
    dpdz_frict  = ftp * rho_ns * v_m**2 / (2 * d_m)
    dpdz_total  = dpdz_elev + dpdz_frict

    # ── Pressure profile vs depth ──
    dz = depth_m / n_steps
    depths   = []
    pressures = []
    p = 101325.0  # Pa surface pressure
    for i in range(n_steps + 1):
        z = i * dz
        depths.append(round(z, 1))
        pressures.append(round(p / 1e5, 3))  # bar
        p += dpdz_total * dz

    dp_total_bar = (pressures[-1] - pressures[0])

    return {
        "pattern": pattern,
        "holdup_Hl": round(Hl, 4),
        "v_sl": round(v_sl, 3),
        "v_sg": round(v_sg, 3),
        "v_m":  round(v_m,  3),
        "rho_mixture": round(rho_m, 2),
        "Re": round(Re, 0),
        "dpdz_elevation_Pa_m":  round(dpdz_elev, 2),
        "dpdz_friction_Pa_m":   round(dpdz_frict, 2),
        "dpdz_total_Pa_m":      round(dpdz_total, 2),
        "dp_total_bar":          round(dp_total_bar, 3),
        "depths":    depths,
        "pressures": pressures,
    }


def stiff_davis_index(
    ca2: float,   # Ca2+  mg/L
    mg2: float,   # Mg2+  mg/L
    na:  float,   # Na+   mg/L
    hco3: float,  # HCO3- mg/L
    so4:  float,  # SO4 2- mg/L
    cl:   float,  # Cl-   mg/L
    temp_c: float,
    pressure_atm: float,
    n_points: int = 30
) -> dict:
    """
    Stiff-Davis saturation index for CaCO3 scale.
    SI = pH_actual - pH_saturation
    SI > 0 → scaling tendency
    SI < 0 → corrosive / undersaturated
    """
    # Convert mg/L to mol/L
    mCa  = ca2  / 40080.0
    mMg  = mg2  / 24305.0
    mNa  = na   / 22990.0
    mHCO3= hco3 / 61017.0
    mSO4 = so4  / 96060.0
    mCl  = cl   / 35450.0

    # Ionic strength
    I = 0.5 * (mCa*4 + mMg*4 + mNa*1 + mHCO3*1 + mSO4*4 + mCl*1)
    I = max(I, 1e-9)

    # Activity coefficients (Davies equation)
    def activity_coeff(z, I):
        return 10**(-0.5 * z**2 * (math.sqrt(I)/(1+math.sqrt(I)) - 0.3*I))

    gamma_ca  = activity_coeff(2, I)
    gamma_hco3= activity_coeff(1, I)
    gamma_co3 = activity_coeff(2, I)

    # pK2 and pKsp for CaCO3 temperature correction (empirical)
    T_K = temp_c + 273.15
    pK2  = 2902.39/T_K + 0.02379*T_K - 6.498
    pKsp = 171.9065 + 0.077993*T_K - 2839.319/T_K - 71.595*math.log10(T_K)

    # Pressure correction (Dodson & Standing simplified)
    pK2  += -pressure_atm * 0.00041
    pKsp += -pressure_atm * 0.00041

    # pH at saturation (Stiff-Davis modified)
    if mCa > 0 and mHCO3 > 0:
        pH_s = pK2 - pKsp + math.log10(mCa * gamma_ca) + math.log10(mHCO3 * gamma_hco3)
    else:
        pH_s = 7.0

    # Approximate actual pH from alkalinity
    pH_actual = 6.35 + math.log10(mHCO3 / max(mCa * 0.5, 1e-9))
    pH_actual = min(max(pH_actual, 4.0), 10.0)

    SI = pH_actual - pH_s

    # Temperature profile: SI vs T from 20°C to temp_c
    temps = [20 + (temp_c - 20) * i / n_points for i in range(n_points + 1)]
    si_profile = []
    for t in temps:
        T_k = t + 273.15
        pk2_t  = 2902.39/T_k + 0.02379*T_k - 6.498
        pksp_t = 171.9065 + 0.077993*T_k - 2839.319/T_k - 71.595*math.log10(T_k)
        pk2_t  += -pressure_atm * 0.00041
        pksp_t += -pressure_atm * 0.00041
        if mCa > 0 and mHCO3 > 0:
            phs_t = pk2_t - pksp_t + math.log10(mCa * gamma_ca) + math.log10(mHCO3 * gamma_hco3)
        else:
            phs_t = 7.0
        si_t = pH_actual - phs_t
        si_profile.append(round(si_t, 4))

    # Depth of first scaling (linear approximation based on T gradient 3°C/100m)
    scale_depth = None
    if SI > 0 and temp_c > 20:
        t_crit = temp_c - (temp_c - 20) * abs(SI) / max(abs(si_profile[-1] - si_profile[0]), 0.01)
        dt_per_m = (temp_c - 20) / 2000  # assume 2000m well
        if dt_per_m > 0:
            scale_depth = round((temp_c - t_crit) / dt_per_m, 0)

    verdict = ""
    if SI > 0.5:
        verdict = "🔴 Высокий риск солеотложения CaCO₃"
    elif SI > 0:
        verdict = "🟡 Умеренный риск. Рекомендуется мониторинг"
    elif SI > -0.5:
        verdict = "🟢 Стабильная вода. Риск минимален"
    else:
        verdict = "🔵 Вода агрессивна (коррозионная)"

    return {
        "SI": round(SI, 3),
        "pH_actual": round(pH_actual, 2),
        "pH_saturation": round(pH_s, 2),
        "ionic_strength": round(I, 4),
        "scale_depth_m": scale_depth,
        "verdict": verdict,
        "temps": [round(t, 1) for t in temps],
        "si_profile": si_profile,
    }


def esp_degradation(
    q_nominal: list,   # [m3/day] nominal flow points
    h_nominal: list,   # [m] nominal head points
    gor: float,        # gas-oil ratio m3/m3
    bsw: float,        # water cut fraction 0-1
    sand_ppm: float,   # sand content ppm
    viscosity_cp: float,  # fluid viscosity cP
) -> dict:
    """
    ESP performance degradation model.
    Applies correction factors for GOR, viscosity, and sand.
    """
    # ── GAS CORRECTION (Turpin correlation) ──
    # Free gas fraction at pump intake
    f_gas = gor / (gor + 1000)  # simplified
    k_gas = 1.0 - 0.5 * f_gas - 1.5 * f_gas**2
    k_gas = max(k_gas, 0.2)

    # ── VISCOSITY CORRECTION (Stepanoff / HI method simplified) ──
    if viscosity_cp <= 1:
        k_visc_q = 1.0
        k_visc_h = 1.0
    elif viscosity_cp <= 100:
        k_visc_q = 1.0 - 0.0012 * (viscosity_cp - 1)**0.9
        k_visc_h = 1.0 - 0.001  * (viscosity_cp - 1)**0.85
    else:
        k_visc_q = max(0.55, 1.0 - 0.0012 * 99**0.9 - 0.002 * (viscosity_cp - 100) / 100)
        k_visc_h = max(0.60, 1.0 - 0.001  * 99**0.85)

    # ── SAND/ABRASION CORRECTION ──
    k_sand = 1.0 - sand_ppm / 500000  # linear degradation, 500k ppm = complete failure
    k_sand = max(k_sand, 0.5)

    # Combined correction
    k_total = k_gas * k_visc_q * k_sand

    # Apply corrections to performance curve
    q_degraded = [round(q * k_total, 1) for q in q_nominal]
    h_degraded = [round(h * k_visc_h * k_gas, 1) for h in h_nominal]

    # Best efficiency point shift
    bep_idx = h_nominal.index(max(h_nominal))
    q_bep_nom  = q_nominal[bep_idx]
    q_bep_deg  = q_degraded[bep_idx]

    return {
        "k_gas":   round(k_gas, 3),
        "k_visc":  round(k_visc_q, 3),
        "k_sand":  round(k_sand, 3),
        "k_total": round(k_total, 3),
        "q_nominal":  q_nominal,
        "h_nominal":  h_nominal,
        "q_degraded": q_degraded,
        "h_degraded": h_degraded,
        "q_bep_nominal":  q_bep_nom,
        "q_bep_degraded": q_bep_deg,
        "head_loss_pct": round((1 - k_visc_h * k_gas) * 100, 1),
        "flow_loss_pct":  round((1 - k_total) * 100, 1),
    }


# ═══════════════════════════════════════════════════════════════
#  HTML
# ═══════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,viewport-fit=cover">
<meta name="theme-color" content="#07090f">
<title>PetroCalc Pro</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{
  --bg:#07090f;--bg2:#0c1018;--bg3:#111720;
  --panel:rgba(11,17,28,0.95);
  --b:rgba(251,191,36,0.12);--bb:rgba(251,191,36,0.35);
  --ac:#fbbf24;--ac2:#f59e0b;
  --gr:#10b981;--re:#f43f5e;--bl:#38bdf8;--vi:#8b5cf6;--or:#f97316;
  --tx:#e2e8f0;--mu:#64748b;
  --ff:'Rajdhani',sans-serif;--fm:'JetBrains Mono',monospace;--fb:'Inter',sans-serif;
  --nav-h:66px;--bot-h:62px;
}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--tx);font-family:var(--fb);min-height:100vh;overflow-x:hidden}

/* Oil-field grid background */
body::before{content:'';position:fixed;inset:0;
  background-image:
    radial-gradient(circle at 20% 20%, rgba(251,191,36,.03) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(16,185,129,.03) 0%, transparent 50%),
    linear-gradient(rgba(251,191,36,.02) 1px,transparent 1px),
    linear-gradient(90deg,rgba(251,191,36,.02) 1px,transparent 1px);
  background-size:100% 100%,100% 100%,50px 50px,50px 50px;
  pointer-events:none;z-index:0}

/* ── NAV ── */
nav{position:sticky;top:0;z-index:200;height:var(--nav-h);
    background:rgba(7,9,15,.97);backdrop-filter:blur(24px);
    border-bottom:1px solid var(--b);padding:0 24px;
    display:flex;align-items:center;justify-content:space-between}
.logo{font-family:var(--ff);font-size:24px;font-weight:700;letter-spacing:2px;
      display:flex;align-items:center;gap:10px}
.logo-drop{width:32px;height:32px;position:relative;flex-shrink:0}
.logo-drop svg{width:100%;height:100%;animation:drip 3s ease-in-out infinite}
@keyframes drip{0%,100%{transform:scaleY(1)}50%{transform:scaleY(1.08)}}
.logo span{color:var(--ac)}
.nav-r{display:flex;align-items:center;gap:8px}
.lbtn{padding:5px 14px;border-radius:20px;border:1px solid var(--bb);background:transparent;
      color:var(--ac);font-family:var(--fm);font-size:11px;cursor:pointer;transition:all .2s;min-height:36px}
.lbtn.on{background:rgba(251,191,36,.15)}
.ubtn{padding:5px 14px;border-radius:20px;border:1px solid rgba(56,189,248,.35);background:transparent;
      color:var(--bl);font-family:var(--fm);font-size:11px;cursor:pointer;transition:all .2s;min-height:36px}
.ubtn.on{background:rgba(56,189,248,.15)}
.ver{font-family:var(--fm);font-size:10px;border:1px solid var(--bb);color:var(--ac);padding:4px 10px;border-radius:20px}

/* ── DESKTOP TABS ── */
.desk-tabs{display:flex;gap:3px;background:var(--bg3);border:1px solid var(--b);
           border-radius:14px;padding:4px;margin:20px 24px 0}
.dtb{flex:1;padding:10px 14px;background:transparent;border:none;cursor:pointer;
     color:var(--mu);font-family:var(--ff);font-size:14px;font-weight:600;
     border-radius:10px;transition:all .2s;display:flex;align-items:center;justify-content:center;gap:7px}
.dtb:hover{color:var(--tx);background:rgba(251,191,36,.06)}
.dtb.on{background:linear-gradient(135deg,rgba(251,191,36,.15),rgba(16,185,129,.1));
        color:var(--ac);border:1px solid var(--bb)}

/* ── MOBILE BOTTOM NAV ── */
.mob-nav{display:none;position:fixed;bottom:0;left:0;right:0;z-index:200;
         height:var(--bot-h);background:rgba(7,9,15,.98);
         backdrop-filter:blur(24px);border-top:1px solid var(--b);
         padding-bottom:env(safe-area-inset-bottom)}
.mob-nav-inner{display:flex;height:100%}
.mnb{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
     gap:3px;background:transparent;border:none;cursor:pointer;color:var(--mu);
     font-size:9px;font-family:var(--fb);font-weight:500;padding:4px 2px;min-height:44px}
.mnb.on{color:var(--ac)}
.mnb.on .mni{background:rgba(251,191,36,.12);border-radius:10px}
.mni{font-size:20px;padding:3px 12px;transition:background .2s}

/* ── MAIN ── */
main{padding:20px 24px 40px;position:relative;z-index:1}
.tc{display:none}.tc.on{display:block;animation:fi .3s ease}
@keyframes fi{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}

/* ── CARD ── */
.card{background:var(--panel);border:1px solid var(--b);border-radius:20px;
      backdrop-filter:blur(20px);overflow:hidden;margin-bottom:16px}
.ch{padding:18px 24px;border-bottom:1px solid var(--b);display:flex;align-items:center;gap:12px}
.ci{width:40px;height:40px;background:linear-gradient(135deg,rgba(251,191,36,.2),rgba(16,185,129,.15));
    border:1px solid var(--bb);border-radius:10px;display:flex;align-items:center;
    justify-content:center;font-size:19px;flex-shrink:0}
.ct{font-family:var(--ff);font-size:17px;font-weight:700;letter-spacing:.8px}
.cs{font-size:11px;color:var(--mu);margin-top:2px}
.cb{padding:24px}

/* ── FORM ── */
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.g2{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.fg{display:flex;flex-direction:column;gap:5px}
.fl{font-family:var(--fm);font-size:10px;color:var(--mu);letter-spacing:1.5px;text-transform:uppercase}
.fu{font-family:var(--fm);font-size:9px;color:rgba(251,191,36,.6);margin-top:2px}

input[type=number],input[type=text],select{
  background:rgba(7,9,15,.85);border:1px solid var(--b);border-radius:9px;
  color:var(--tx);font-family:var(--fm);font-size:14px;
  padding:11px 13px;width:100%;transition:border-color .2s,box-shadow .2s;outline:none;
  min-height:46px;-webkit-appearance:none;appearance:none}
input:focus,select:focus{border-color:var(--ac);box-shadow:0 0 0 3px rgba(251,191,36,.1)}
select option{background:var(--bg2)}
select{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%2364748b' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");
       background-repeat:no-repeat;background-position:right 12px center;padding-right:34px}

/* ── BUTTONS ── */
.btn{width:100%;padding:14px;border:none;border-radius:11px;color:#fff;
     font-family:var(--ff);font-size:16px;font-weight:700;letter-spacing:1.5px;
     cursor:pointer;transition:transform .15s,box-shadow .15s;min-height:50px}
.btn:hover{transform:translateY(-2px)}
.btn:active{transform:scale(.98)}
.b-gold{background:linear-gradient(135deg,#d97706,#b45309)}
.b-green{background:linear-gradient(135deg,#059669,#0d9488)}
.b-blue{background:linear-gradient(135deg,#0284c7,#6366f1)}
.b-orange{background:linear-gradient(135deg,#ea580c,#dc2626)}

/* ── DIVIDER ── */
.div{border:none;height:1px;background:var(--b);margin:18px 0}
.sec{font-family:var(--fm);font-size:10px;color:var(--mu);letter-spacing:2px;
     text-transform:uppercase;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.sec::after{content:'';flex:1;height:1px;background:var(--b)}

/* ── RESULT CARDS ── */
.rg{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:8px}
.rg3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:8px}
.rg2{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:10px}
.rc{background:rgba(7,9,15,.75);border:1px solid var(--b);border-radius:12px;
    padding:16px 14px;text-align:center;transition:transform .2s;
    position:relative;overflow:hidden}
.rc::before{content:'';position:absolute;inset:0;opacity:0;transition:opacity .3s;border-radius:inherit}
.rc:hover{transform:translateY(-2px)}.rc:hover::before{opacity:1}
.cg{border-color:rgba(251,191,36,.4)}.cg::before{background:radial-gradient(ellipse at 50% 0,rgba(251,191,36,.08),transparent 70%)}
.cm{border-color:rgba(16,185,129,.4)}.cm::before{background:radial-gradient(ellipse at 50% 0,rgba(16,185,129,.08),transparent 70%)}
.cb2{border-color:rgba(56,189,248,.4)}.cb2::before{background:radial-gradient(ellipse at 50% 0,rgba(56,189,248,.08),transparent 70%)}
.cr{border-color:rgba(244,63,94,.4)}.cr::before{background:radial-gradient(ellipse at 50% 0,rgba(244,63,94,.08),transparent 70%)}
.cv{border-color:rgba(139,92,246,.4)}.cv::before{background:radial-gradient(ellipse at 50% 0,rgba(139,92,246,.08),transparent 70%)}
.co{border-color:rgba(249,115,22,.4)}.co::before{background:radial-gradient(ellipse at 50% 0,rgba(249,115,22,.08),transparent 70%)}
.rl{font-family:var(--fm);font-size:10px;color:var(--mu);letter-spacing:1px;text-transform:uppercase}
.rv{font-family:var(--ff);font-size:28px;font-weight:700;line-height:1;margin:7px 0 3px}
.ru{font-family:var(--fm);font-size:10px;color:var(--mu)}
.tg{color:var(--ac)}.tm{color:var(--gr)}.tb{color:var(--bl)}.tr{color:var(--re)}.tv{color:var(--vi)}.to{color:var(--or)}

/* ── BANNERS ── */
.warn,.okb,.errb,.infob{margin-top:12px;padding:13px 16px;border-radius:10px;
                         font-size:13px;display:flex;gap:9px;align-items:flex-start;line-height:1.5}
.warn{background:rgba(249,115,22,.08);border:1px solid rgba(249,115,22,.3);color:#fdba74}
.okb{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.3);color:#34d399}
.errb{background:rgba(244,63,94,.08);border:1px solid rgba(244,63,94,.3);color:#fb7185}
.infob{background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.3);color:#fde68a}

/* ── PLOTLY CONTAINER ── */
.plot-box{background:rgba(7,9,15,.8);border:1px solid var(--b);border-radius:14px;
          padding:16px;margin-top:14px;min-height:360px}
.plot-box .js-plotly-plot{border-radius:10px}

/* ── FLOW PATTERN BADGE ── */
.pattern-badge{display:inline-flex;align-items:center;gap:7px;padding:8px 18px;
               border-radius:20px;font-family:var(--fm);font-size:12px;letter-spacing:1px;
               margin-bottom:12px;border:1px solid}
.pat-seg{background:rgba(56,189,248,.1);border-color:rgba(56,189,248,.4);color:var(--bl)}
.pat-int{background:rgba(251,191,36,.1);border-color:rgba(251,191,36,.4);color:var(--ac)}
.pat-dis{background:rgba(16,185,129,.1);border-color:rgba(16,185,129,.4);color:var(--gr)}
.pat-tr{background:rgba(139,92,246,.1);border-color:rgba(139,92,246,.4);color:var(--vi)}

/* ── UNITS TOGGLE ── */
.unit-row{display:flex;gap:6px;margin-bottom:16px;flex-wrap:wrap}
.unit-chip{padding:5px 14px;border-radius:20px;border:1px solid var(--b);background:transparent;
           color:var(--mu);font-family:var(--fm);font-size:11px;cursor:pointer;transition:all .2s;min-height:34px}
.unit-chip.on{background:rgba(56,189,248,.12);border-color:rgba(56,189,248,.4);color:var(--bl)}

/* ── RESPONSIVE ── */
@media(max-width:900px){
  main{padding:16px 16px 28px}
  .desk-tabs{margin:16px 16px 0}
  .g3,.g4{grid-template-columns:1fr 1fr}
  .rg{grid-template-columns:1fr 1fr}
}
@media(max-width:640px){
  :root{--bot-h:60px}
  nav{padding:0 14px}
  main{padding:12px 12px 20px}
  .desk-tabs{display:none}
  .mob-nav{display:block}
  body{padding-bottom:calc(var(--bot-h) + env(safe-area-inset-bottom))}
  .g3,.g4,.g2{grid-template-columns:1fr}
  .rg,.rg3{grid-template-columns:1fr 1fr}
  .rg2{grid-template-columns:1fr 1fr}
  input[type=number],input[type=text],select{font-size:16px;min-height:50px}
  .btn{min-height:52px}
  .rv{font-size:24px}
  .plot-box{min-height:280px;padding:10px}
}
@media(max-width:360px){
  .rg,.rg3{grid-template-columns:1fr 1fr}
  .rv{font-size:20px}
}
</style>
</head>
<body>

<!-- ═══ NAV ═══ -->
<nav>
  <div class="logo">
    <div class="logo-drop">
      <svg viewBox="0 0 32 32" fill="none">
        <path d="M16 2 C16 2, 26 14, 26 20 C26 25.5 21.5 30 16 30 C10.5 30 6 25.5 6 20 C6 14 16 2 16 2Z"
              fill="url(#dropGrad)" opacity="0.9"/>
        <defs>
          <linearGradient id="dropGrad" x1="6" y1="2" x2="26" y2="30" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stop-color="#fbbf24"/>
            <stop offset="100%" stop-color="#f59e0b"/>
          </linearGradient>
        </defs>
      </svg>
    </div>
    PETRO<span>CALC</span>
  </div>
  <div class="nav-r">
    <button class="lbtn on" onclick="setLang('ru',this)">RU</button>
    <button class="lbtn" onclick="setLang('en',this)">EN</button>
    <div class="ver" style="display:none" id="ver-badge">v1.0</div>
  </div>
</nav>

<!-- ═══ DESKTOP TABS ═══ -->
<div class="desk-tabs">
  <button class="dtb on" onclick="showTab('bb',this)">
    🌊 <span data-ru="Многофазный поток" data-en="Multiphase Flow">Многофазный поток</span>
  </button>
  <button class="dtb" onclick="showTab('scale',this)">
    🧪 <span data-ru="Солеотложение" data-en="Scale Index">Солеотложение</span>
  </button>
  <button class="dtb" onclick="showTab('esp',this)">
    ⚙️ <span data-ru="Деградация ЭЦН" data-en="ESP Degradation">Деградация ЭЦН</span>
  </button>
</div>

<!-- ═══ MOBILE BOTTOM NAV ═══ -->
<div class="mob-nav">
  <div class="mob-nav-inner">
    <button class="mnb on" onclick="showTab('bb',this)" data-tab="bb">
      <div class="mni">🌊</div>
      <span data-ru="Поток" data-en="Flow">Поток</span>
    </button>
    <button class="mnb" onclick="showTab('scale',this)" data-tab="scale">
      <div class="mni">🧪</div>
      <span data-ru="Соль" data-en="Scale">Соль</span>
    </button>
    <button class="mnb" onclick="showTab('esp',this)" data-tab="esp">
      <div class="mni">⚙️</div>
      <span data-ru="ЭЦН" data-en="ESP">ЭЦН</span>
    </button>
  </div>
</div>

<main>

<!-- ═══════════════════════════════════════
     TAB 1 — BEGGS & BRILL
═══════════════════════════════════════ -->
<div id="tab-bb" class="tc on">
<div class="card">
  <div class="ch">
    <div class="ci">🌊</div>
    <div>
      <div class="ct" data-ru="МНОГОФАЗНЫЙ ПОТОК — БЕГГЗ И БРИЛЛ" data-en="MULTIPHASE FLOW — BEGGS &amp; BRILL">МНОГОФАЗНЫЙ ПОТОК — БЕГГЗ И БРИЛЛ</div>
      <div class="cs" data-ru="Расчёт потерь давления · ГЖС · Профиль по глубине" data-en="Pressure drop · Gas-liquid mixture · Depth profile">Расчёт потерь давления · ГЖС · Профиль по глубине</div>
    </div>
  </div>
  <div class="cb">

    <!-- UNITS TOGGLE -->
    <div class="unit-row">
      <span style="font-family:var(--fm);font-size:10px;color:var(--mu);letter-spacing:1px;align-self:center">ЕДИНИЦЫ:</span>
      <button class="unit-chip on" onclick="setUnits('si',this)" id="u-si">СИ (м³/сут, Па)</button>
      <button class="unit-chip" onclick="setUnits('field',this)" id="u-field">Промысл. (т/сут, атм)</button>
    </div>

    <p class="sec" data-ru="Флюид и трубопровод" data-en="Fluid & pipe">Флюид и трубопровод</p>
    <div class="g3">
      <div class="fg">
        <label class="fl" data-ru="Дебит жидкости" data-en="Liquid flow rate">Дебит жидкости</label>
        <input type="number" id="bb-ql" value="200" step="10" min="1" inputmode="decimal">
        <span class="fu" id="bb-ql-u">м³/сут</span>
      </div>
      <div class="fg">
        <label class="fl" data-ru="Дебит газа" data-en="Gas flow rate">Дебит газа</label>
        <input type="number" id="bb-qg" value="50000" step="1000" min="0" inputmode="decimal">
        <span class="fu" id="bb-qg-u">м³/сут</span>
      </div>
      <div class="fg">
        <label class="fl" data-ru="Внутр. диаметр трубы" data-en="Pipe inner diameter">Внутр. диаметр трубы</label>
        <input type="number" id="bb-d" value="73" step="1" min="10" inputmode="decimal">
        <span class="fu" id="bb-d-u">мм</span>
      </div>
      <div class="fg">
        <label class="fl" data-ru="Плотность жидкости" data-en="Liquid density">Плотность жидкости</label>
        <input type="number" id="bb-rhol" value="850" step="10" min="600" max="1100" inputmode="decimal">
        <span class="fu" id="bb-rhol-u">кг/м³</span>
      </div>
      <div class="fg">
        <label class="fl" data-ru="Плотность газа" data-en="Gas density">Плотность газа</label>
        <input type="number" id="bb-rhog" value="5.0" step="0.5" min="0.5" inputmode="decimal">
        <span class="fu">кг/м³</span>
      </div>
      <div class="fg">
        <label class="fl" data-ru="Вязкость жидкости" data-en="Liquid viscosity">Вязкость жидкости</label>
        <input type="number" id="bb-mul" value="2.0" step="0.5" min="0.1" inputmode="decimal">
        <span class="fu">мПа·с</span>
      </div>
      <div class="fg">
        <label class="fl" data-ru="Поверхностное натяжение" data-en="Surface tension">Поверхностное натяжение</label>
        <input type="number" id="bb-sigma" value="25" step="1" min="1" inputmode="decimal">
        <span class="fu">мН/м</span>
      </div>
      <div class="fg">
        <label class="fl" data-ru="Угол наклона, °" data-en="Inclination angle, °">Угол наклона, °</label>
        <input type="number" id="bb-theta" value="90" step="5" min="-90" max="90" inputmode="decimal">
        <span class="fu">° (90 = вертикаль)</span>
      </div>
      <div class="fg">
        <label class="fl" data-ru="Глубина скважины" data-en="Well depth">Глубина скважины</label>
        <input type="number" id="bb-depth" value="2000" step="100" min="100" inputmode="decimal">
        <span class="fu" id="bb-depth-u">м</span>
      </div>
    </div>

    <hr class="div">
    <button type="button" onclick="calcBB()" class="btn b-gold">🌊 РАССЧИТАТЬ ПАДЕНИЕ ДАВЛЕНИЯ</button>

    <div id="res-bb"></div>
  </div>
</div>
</div>

<!-- ═══════════════════════════════════════
     TAB 2 — SCALE INDEX
═══════════════════════════════════════ -->
<div id="tab-scale" class="tc">
<div class="card">
  <div class="ch">
    <div class="ci">🧪</div>
    <div>
      <div class="ct" data-ru="ИНДЕКС СОЛЕОТЛОЖЕНИЯ STIFF-DAVIS" data-en="STIFF-DAVIS SCALE INDEX">ИНДЕКС СОЛЕОТЛОЖЕНИЯ STIFF-DAVIS</div>
      <div class="cs" data-ru="CaCO₃ · Ионный состав воды · Профиль стабильности" data-en="CaCO₃ · Ion composition · Stability profile">CaCO₃ · Ионный состав воды · Профиль стабильности</div>
    </div>
  </div>
  <div class="cb">

    <p class="sec" data-ru="Ионный состав попутно-добываемой воды (мг/л)" data-en="Produced water ion composition (mg/L)">Ионный состав попутно-добываемой воды (мг/л)</p>
    <div class="g3">
      <div class="fg">
        <label class="fl">Ca²⁺ (Кальций)</label>
        <input type="number" id="sc-ca" value="850" step="10" min="0" inputmode="decimal">
        <span class="fu">мг/л</span>
      </div>
      <div class="fg">
        <label class="fl">Mg²⁺ (Магний)</label>
        <input type="number" id="sc-mg" value="120" step="10" min="0" inputmode="decimal">
        <span class="fu">мг/л</span>
      </div>
      <div class="fg">
        <label class="fl">Na⁺ (Натрий)</label>
        <input type="number" id="sc-na" value="5000" step="100" min="0" inputmode="decimal">
        <span class="fu">мг/л</span>
      </div>
      <div class="fg">
        <label class="fl">HCO₃⁻ (Гидрокарбонат)</label>
        <input type="number" id="sc-hco3" value="400" step="10" min="0" inputmode="decimal">
        <span class="fu">мг/л</span>
      </div>
      <div class="fg">
        <label class="fl">SO₄²⁻ (Сульфат)</label>
        <input type="number" id="sc-so4" value="200" step="10" min="0" inputmode="decimal">
        <span class="fu">мг/л</span>
      </div>
      <div class="fg">
        <label class="fl">Cl⁻ (Хлорид)</label>
        <input type="number" id="sc-cl" value="8000" step="100" min="0" inputmode="decimal">
        <span class="fu">мг/л</span>
      </div>
    </div>

    <hr class="div">
    <p class="sec" data-ru="Пластовые условия" data-en="Reservoir conditions">Пластовые условия</p>
    <div class="g3">
      <div class="fg">
        <label class="fl" data-ru="Температура пласта" data-en="Reservoir temperature">Температура пласта</label>
        <input type="number" id="sc-temp" value="80" step="5" min="10" max="200" inputmode="decimal">
        <span class="fu">°C</span>
      </div>
      <div class="fg">
        <label class="fl" data-ru="Давление" data-en="Pressure">Давление</label>
        <input type="number" id="sc-pres" value="150" step="10" min="1" inputmode="decimal">
        <span class="fu" id="sc-pres-u">атм</span>
      </div>
    </div>

    <hr class="div">
    <button type="button" onclick="calcScale()" class="btn b-green">🧪 РАССЧИТАТЬ ИНДЕКС СОЛЕОТЛОЖЕНИЯ</button>

    <div id="res-scale"></div>
  </div>
</div>
</div>

<!-- ═══════════════════════════════════════
     TAB 3 — ESP DEGRADATION
═══════════════════════════════════════ -->
<div id="tab-esp" class="tc">
<div class="card">
  <div class="ch">
    <div class="ci">⚙️</div>
    <div>
      <div class="ct" data-ru="ДЕГРАДАЦИЯ ЭЦН" data-en="ESP DEGRADATION">ДЕГРАДАЦИЯ ЭЦН</div>
      <div class="cs" data-ru="Паспортная vs реальная кривая · ГФ · Вязкость · Песок" data-en="Nameplate vs actual curve · GOR · Viscosity · Sand">Паспортная vs реальная кривая · ГФ · Вязкость · Песок</div>
    </div>
  </div>
  <div class="cb">

    <p class="sec" data-ru="Паспортная кривая насоса (5 точек Q–H)" data-en="Nameplate pump curve (5 Q–H points)">Паспортная кривая насоса (5 точек Q–H)</p>
    <div style="overflow-x:auto">
      <table style="width:100%;border-collapse:collapse;font-size:13px;min-width:400px">
        <thead>
          <tr>
            <th style="font-family:var(--fm);font-size:10px;color:var(--ac);letter-spacing:1px;text-transform:uppercase;padding:8px 10px;text-align:left;border-bottom:1px solid var(--b)">
              <span data-ru="Q, м³/сут" data-en="Q, m³/day">Q, м³/сут</span>
            </th>
            <th style="font-family:var(--fm);font-size:10px;color:var(--ac);letter-spacing:1px;text-transform:uppercase;padding:8px 10px;text-align:left;border-bottom:1px solid var(--b)">
              <span data-ru="H, м (напор)" data-en="H, m (head)">H, м (напор)</span>
            </th>
          </tr>
        </thead>
        <tbody id="esp-table-body"></tbody>
      </table>
    </div>

    <hr class="div">
    <p class="sec" data-ru="Условия эксплуатации" data-en="Operating conditions">Условия эксплуатации</p>
    <div class="g3">
      <div class="fg">
        <label class="fl" data-ru="Газовый фактор (ГФ)" data-en="Gas-Oil Ratio">Газовый фактор (ГФ)</label>
        <input type="number" id="esp-gor" value="100" step="10" min="0" inputmode="decimal">
        <span class="fu">м³/м³</span>
      </div>
      <div class="fg">
        <label class="fl" data-ru="Обводнённость" data-en="Water cut">Обводнённость</label>
        <input type="number" id="esp-bsw" value="30" step="5" min="0" max="100" inputmode="decimal">
        <span class="fu">%</span>
      </div>
      <div class="fg">
        <label class="fl" data-ru="Содержание песка" data-en="Sand content">Содержание песка</label>
        <input type="number" id="esp-sand" value="50" step="10" min="0" inputmode="decimal">
        <span class="fu">ppm</span>
      </div>
      <div class="fg">
        <label class="fl" data-ru="Вязкость флюида" data-en="Fluid viscosity">Вязкость флюида</label>
        <input type="number" id="esp-visc" value="5" step="1" min="1" inputmode="decimal">
        <span class="fu">сПз (cP)</span>
      </div>
    </div>

    <hr class="div">
    <button type="button" onclick="calcESP()" class="btn b-orange">⚙️ ПОСТРОИТЬ КРИВЫЕ ЭЦН</button>

    <div id="res-esp"></div>
  </div>
</div>
</div>

</main>

<!-- ═══════════════════════════════════════
     JAVASCRIPT
═══════════════════════════════════════ -->
<script>
// ── CHART.JS HELPERS ──────────────────────────────────────────
const CHART_DEFAULTS = {
  gridColor: 'rgba(251,191,36,0.08)',
  textColor: '#94a3b8',
  fontFamily: 'JetBrains Mono, monospace',
};
const chartInstances = {};

function destroyChart(id) {
  if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; }
}

function makeChartOptions(xLabel, yLabel, extraOpts={}) {
  return {
    responsive: true,
    animation: { duration: 400 },
    plugins: {
      legend: { labels: { color: CHART_DEFAULTS.textColor, font: { size: 11, family: CHART_DEFAULTS.fontFamily } } },
      tooltip: { backgroundColor: '#111720', titleColor: '#e2e8f0', bodyColor: '#94a3b8', borderColor: 'rgba(251,191,36,.3)', borderWidth: 1 },
    },
    scales: {
      x: { title: { display: true, text: xLabel, color: CHART_DEFAULTS.textColor, font: { size: 10 } },
           grid: { color: CHART_DEFAULTS.gridColor }, ticks: { color: CHART_DEFAULTS.textColor, font: { size: 10 } } },
      y: { title: { display: true, text: yLabel, color: CHART_DEFAULTS.textColor, font: { size: 10 } },
           grid: { color: CHART_DEFAULTS.gridColor }, ticks: { color: CHART_DEFAULTS.textColor, font: { size: 10 } },
           ...( extraOpts.reverseY ? { reverse: true } : {} ) },
    },
    ...extraOpts,
  };
}

// ── LANGUAGE ─────────────────────────────────────────────────
let LANG = 'ru';
function setLang(l, btn) {
  LANG = l;
  document.querySelectorAll('.lbtn').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  document.querySelectorAll('[data-ru]').forEach(el => {
    el.textContent = el.getAttribute('data-' + l) || el.textContent;
  });
}

// ── TABS ─────────────────────────────────────────────────────
function showTab(id, btn) {
  document.querySelectorAll('.tc').forEach(el => el.classList.remove('on'));
  document.querySelectorAll('.dtb, .mnb').forEach(el => el.classList.remove('on'));
  document.getElementById('tab-' + id).classList.add('on');
  document.querySelectorAll('.dtb').forEach(b => {
    if (b.getAttribute('onclick') && b.getAttribute('onclick').includes("'" + id + "'")) b.classList.add('on');
  });
  document.querySelectorAll('.mnb').forEach(b => {
    if (b.dataset.tab === id) b.classList.add('on');
  });
  if (window.innerWidth <= 640) window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── UNITS ─────────────────────────────────────────────────────
let UNITS = 'si';
function setUnits(u, btn) {
  UNITS = u;
  document.querySelectorAll('.unit-chip').forEach(b => b.classList.remove('on'));
  btn.classList.add('on');
  if (u === 'si') {
    document.getElementById('bb-ql-u').textContent   = 'м³/сут';
    document.getElementById('bb-qg-u').textContent   = 'м³/сут';
    document.getElementById('bb-d-u').textContent    = 'мм';
    document.getElementById('bb-depth-u').textContent= 'м';
  } else {
    document.getElementById('bb-ql-u').textContent   = 'т/сут';
    document.getElementById('bb-qg-u').textContent   = 'тыс. м³/сут';
    document.getElementById('bb-d-u').textContent    = 'дюйм';
    document.getElementById('bb-depth-u').textContent= 'м';
  }
}

// ── FLOW PATTERN COLORS ───────────────────────────────────────
const PAT_CLS = {
  'Segregated':  'pat-seg',
  'Intermittent':'pat-int',
  'Distributed': 'pat-dis',
  'Transition':  'pat-tr',
};
const PAT_RU = {
  'Segregated':  'Расслоённый',
  'Intermittent':'Прерывистый (пробковый)',
  'Distributed': 'Дисперсный',
  'Transition':  'Переходный',
};

// ── BEGGS & BRILL ─────────────────────────────────────────────
async function calcBB() {
  let ql    = parseFloat(document.getElementById('bb-ql').value);
  let qg    = parseFloat(document.getElementById('bb-qg').value);
  let d_mm  = parseFloat(document.getElementById('bb-d').value);
  let rhol  = parseFloat(document.getElementById('bb-rhol').value);
  let rhog  = parseFloat(document.getElementById('bb-rhog').value);
  let mul   = parseFloat(document.getElementById('bb-mul').value) / 1000; // mPa·s → Pa·s
  let sigma = parseFloat(document.getElementById('bb-sigma').value) / 1000; // mN/m → N/m
  let theta = parseFloat(document.getElementById('bb-theta').value);
  let depth = parseFloat(document.getElementById('bb-depth').value);

  // Field units conversion
  if (UNITS === 'field') {
    ql = ql / rhol * 1000;  // t/day → m3/day (approx)
    qg = qg * 1000;          // Mm3/day → m3/day
    d_mm = d_mm * 25.4;      // inch → mm
  }

  const body = new URLSearchParams({
    q_l: ql, q_g: qg, d_mm, rho_l: rhol, rho_g: rhog,
    mu_l: mul * 1000, sigma: sigma * 1000,
    theta, depth
  });
  const res = await fetch('/api/beggs_brill', { method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body });
  const d = await res.json();

  // ── Result cards ──
  const patRu  = PAT_RU[d.pattern] || d.pattern;
  const patCls = PAT_CLS[d.pattern] || 'pat-tr';

  document.getElementById('res-bb').innerHTML = `
    <div class="pattern-badge ${patCls}">
      <span>🌊</span> <span>Режим течения: <strong>${patRu}</strong></span>
    </div>
    <p class="sec">Результаты</p>
    <div class="rg">
      <div class="rc cg"><div class="rl">Перепад давления</div><div class="rv tg">${d.dp_total_bar}</div><div class="ru">бар</div></div>
      <div class="rc cm"><div class="rl">Удержание жидкости H_L</div><div class="rv tm">${d.holdup_Hl}</div><div class="ru">доля</div></div>
      <div class="rc cb2"><div class="rl">Скор. жидкости v_sl</div><div class="rv tb">${d.v_sl}</div><div class="ru">м/с</div></div>
      <div class="rc co"><div class="rl">Скор. смеси v_m</div><div class="rv to">${d.v_m}</div><div class="ru">м/с</div></div>
    </div>
    <div class="rg" style="margin-top:10px">
      <div class="rc"><div class="rl">Плотность смеси</div><div class="rv">${d.rho_mixture}</div><div class="ru">кг/м³</div></div>
      <div class="rc"><div class="rl">Re</div><div class="rv">${Number(d.Re).toLocaleString()}</div><div class="ru"></div></div>
      <div class="rc"><div class="rl">Гравитац. потери</div><div class="rv">${d.dpdz_elevation_Pa_m}</div><div class="ru">Па/м</div></div>
      <div class="rc"><div class="rl">Потери трения</div><div class="rv">${d.dpdz_friction_Pa_m}</div><div class="ru">Па/м</div></div>
    </div>
    <div class="plot-box" id="plot-bb"><canvas id="canvas-bb"></canvas></div>`;

  // ── Chart.js pressure profile ──
  destroyChart('bb');
  const ctx = document.getElementById('canvas-bb').getContext('2d');
  chartInstances['bb'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: d.pressures,
      datasets: [{
        label: 'Давление (бар)',
        data: d.depths.map((y, i) => ({ x: d.pressures[i], y })),
        borderColor: '#fbbf24', backgroundColor: 'rgba(251,191,36,0.07)',
        borderWidth: 2.5, tension: 0.4, fill: true,
        pointRadius: 2, pointBackgroundColor: '#fbbf24',
      }]
    },
    options: {
      ...makeChartOptions('Давление, бар', 'Глубина, м', { reverseY: true }),
      plugins: {
        ...makeChartOptions('','').plugins,
        title: { display: true, text: 'Профиль давления по глубине', color: '#fbbf24', font: { size: 13, family: 'JetBrains Mono, monospace' } },
      },
      parsing: { xAxisKey: 'x', yAxisKey: 'y' },
    }
  });
}

// ── SCALE INDEX ───────────────────────────────────────────────
async function calcScale() {
  const body = new URLSearchParams({
    ca2:  document.getElementById('sc-ca').value,
    mg2:  document.getElementById('sc-mg').value,
    na:   document.getElementById('sc-na').value,
    hco3: document.getElementById('sc-hco3').value,
    so4:  document.getElementById('sc-so4').value,
    cl:   document.getElementById('sc-cl').value,
    temp_c:       document.getElementById('sc-temp').value,
    pressure_atm: document.getElementById('sc-pres').value,
  });
  const res = await fetch('/api/scale', { method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body });
  const d = await res.json();

  const siColor  = d.SI > 0.5 ? 'cr' : (d.SI > 0 ? 'co' : (d.SI > -0.5 ? 'cm' : 'cb2'));
  const siText   = d.SI > 0.5 ? 'tr'  : (d.SI > 0 ? 'to' : (d.SI > -0.5 ? 'tm' : 'tb'));
  const depthStr = d.scale_depth_m ? `Начало солеотложения ≈ ${d.scale_depth_m} м` : 'Солеотложение не прогнозируется';
  const bannerCls= d.SI > 0.5 ? 'errb' : (d.SI > 0 ? 'warn' : 'okb');

  document.getElementById('res-scale').innerHTML = `
    <p class="sec" style="margin-top:18px">Результаты</p>
    <div class="rg3">
      <div class="rc ${siColor}"><div class="rl">Индекс Stiff-Davis (SI)</div><div class="rv ${siText}">${d.SI}</div><div class="ru">SI &gt; 0 → осадок</div></div>
      <div class="rc"><div class="rl">pH фактический</div><div class="rv">${d.pH_actual}</div><div class="ru"></div></div>
      <div class="rc"><div class="rl">pH насыщения</div><div class="rv">${d.pH_saturation}</div><div class="ru"></div></div>
    </div>
    <div class="rg2" style="margin-top:10px">
      <div class="rc"><div class="rl">Ионная сила</div><div class="rv">${d.ionic_strength}</div><div class="ru">моль/л</div></div>
      <div class="rc cg"><div class="rl">Глубина осадка</div><div class="rv tg">${d.scale_depth_m || '—'}</div><div class="ru">м</div></div>
    </div>
    <div class="${bannerCls}" style="margin-top:10px"><span>🧪</span><div><strong>${d.verdict}</strong><br>${depthStr}</div></div>
    <div class="plot-box" id="plot-scale"><canvas id="canvas-scale"></canvas></div>`;

  // ── Chart.js SI profile ──
  destroyChart('scale');
  const ctxS = document.getElementById('canvas-scale').getContext('2d');
  chartInstances['scale'] = new Chart(ctxS, {
    type: 'line',
    data: {
      labels: d.temps,
      datasets: [
        {
          label: 'Индекс SI',
          data: d.si_profile,
          borderColor: '#fbbf24', backgroundColor: d.SI > 0 ? 'rgba(244,63,94,0.07)' : 'rgba(16,185,129,0.07)',
          borderWidth: 2.5, tension: 0.4, fill: true, pointRadius: 3,
          pointBackgroundColor: d.si_profile.map(v => v > 0 ? '#f43f5e' : '#10b981'),
        },
        {
          label: 'Граница SI=0',
          data: d.temps.map(() => 0),
          borderColor: '#64748b', borderWidth: 1.5, borderDash: [6, 4],
          pointRadius: 0, fill: false,
        }
      ]
    },
    options: {
      ...makeChartOptions('Температура, °C', 'Индекс SI'),
      plugins: {
        ...makeChartOptions('','').plugins,
        title: { display: true, text: 'Термодинамическая стабильность CaCO₃', color: '#fbbf24', font: { size: 12, family: 'JetBrains Mono, monospace' } },
      }
    }
  });
}

// ── ESP DEGRADATION ───────────────────────────────────────────
// Default pump curve (5 points)
const DEFAULT_Q = [0, 100, 200, 300, 400];
const DEFAULT_H = [1500, 1450, 1300, 1050, 700];

function initESPTable() {
  const tb = document.getElementById('esp-table-body');
  tb.innerHTML = DEFAULT_Q.map((q, i) => `
    <tr>
      <td style="padding:6px 10px;border-bottom:1px solid rgba(251,191,36,.05)">
        <input type="number" class="esp-q" value="${q}" style="background:transparent;border:1px solid var(--b);border-radius:6px;color:var(--tx);font-family:var(--fm);font-size:13px;padding:6px 10px;width:100%;min-height:38px" inputmode="decimal">
      </td>
      <td style="padding:6px 10px;border-bottom:1px solid rgba(251,191,36,.05)">
        <input type="number" class="esp-h" value="${DEFAULT_H[i]}" style="background:transparent;border:1px solid var(--b);border-radius:6px;color:var(--tx);font-family:var(--fm);font-size:13px;padding:6px 10px;width:100%;min-height:38px" inputmode="decimal">
      </td>
    </tr>`).join('');
}
initESPTable();

async function calcESP() {
  const qs = [...document.querySelectorAll('.esp-q')].map(i => parseFloat(i.value) || 0);
  const hs = [...document.querySelectorAll('.esp-h')].map(i => parseFloat(i.value) || 0);
  const body = new URLSearchParams({
    q_pts:    JSON.stringify(qs),
    h_pts:    JSON.stringify(hs),
    gor:      document.getElementById('esp-gor').value,
    bsw:      parseFloat(document.getElementById('esp-bsw').value) / 100,
    sand_ppm: document.getElementById('esp-sand').value,
    visc_cp:  document.getElementById('esp-visc').value,
  });
  const res = await fetch('/api/esp', { method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body });
  const d = await res.json();

  const kTotal_pct = Math.round((1 - d.k_total) * 100);

  document.getElementById('res-esp').innerHTML = `
    <p class="sec" style="margin-top:18px">Коэффициенты деградации</p>
    <div class="rg">
      <div class="rc co"><div class="rl">K газ</div><div class="rv to">${d.k_gas}</div><div class="ru">газовый фактор</div></div>
      <div class="rc cv"><div class="rl">K вязкость</div><div class="rv tv">${d.k_visc}</div><div class="ru">вязкость</div></div>
      <div class="rc cr"><div class="rl">K песок</div><div class="rv tr">${d.k_sand}</div><div class="ru">абразия</div></div>
      <div class="rc ${d.k_total > 0.8 ? 'cm' : 'cr'}">
        <div class="rl">K суммарный</div>
        <div class="rv ${d.k_total > 0.8 ? 'tm' : 'tr'}">${d.k_total}</div>
        <div class="ru">итоговый</div>
      </div>
    </div>
    <div class="rg2" style="margin-top:10px">
      <div class="rc cr"><div class="rl">Потеря подачи</div><div class="rv tr">${d.flow_loss_pct}</div><div class="ru">%</div></div>
      <div class="rc co"><div class="rl">Потеря напора</div><div class="rv to">${d.head_loss_pct}</div><div class="ru">%</div></div>
    </div>
    <div class="${d.k_total > 0.8 ? 'okb' : (d.k_total > 0.6 ? 'warn' : 'errb')}" style="margin-top:10px">
      <span>${d.k_total > 0.8 ? '✅' : (d.k_total > 0.6 ? '⚠️' : '🔴')}</span>
      <div>Суммарная деградация <strong>${kTotal_pct}%</strong>. 
      ${d.k_total > 0.8 ? 'Насос работает в штатном режиме.' : 
        d.k_total > 0.6 ? 'Умеренная деградация. Рекомендуется мониторинг.' : 
        'Критическая деградация. Требуется вмешательство.'}</div>
    </div>
    <div class="plot-box" id="plot-esp"><canvas id="canvas-esp"></canvas></div>`;

  // ── Chart.js Q-H curves ──
  destroyChart('esp');
  const ctxE = document.getElementById('canvas-esp').getContext('2d');
  chartInstances['esp'] = new Chart(ctxE, {
    type: 'line',
    data: {
      labels: d.q_nominal,
      datasets: [
        {
          label: '📋 Паспортная кривая',
          data: d.h_nominal,
          borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.06)',
          borderWidth: 2.5, tension: 0.4, fill: false,
          pointRadius: 5, pointBackgroundColor: '#38bdf8',
        },
        {
          label: '⚠️ Реальная (деградация)',
          data: d.h_degraded,
          borderColor: '#f43f5e', backgroundColor: 'rgba(244,63,94,0.06)',
          borderWidth: 2.5, borderDash: [6, 4], tension: 0.4, fill: true,
          pointRadius: 5, pointBackgroundColor: '#f43f5e',
        }
      ]
    },
    options: {
      ...makeChartOptions('Q, м³/сут', 'H, м (напор)'),
      plugins: {
        ...makeChartOptions('','').plugins,
        title: { display: true, text: 'Кривая Q–H: Паспортная vs Реальная', color: '#fbbf24', font: { size: 13, family: 'JetBrains Mono, monospace' } },
      }
    }
  });
}

// Show ver badge on mobile
if (window.innerWidth <= 640) {
  document.getElementById('ver-badge').style.display = 'block';
}
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML


@app.post("/api/beggs_brill")
async def api_beggs_brill(
    q_l:    float = Form(...),
    q_g:    float = Form(...),
    d_mm:   float = Form(...),
    rho_l:  float = Form(...),
    rho_g:  float = Form(...),
    mu_l:   float = Form(...),   # mPa·s received, converted inside
    sigma:  float = Form(...),   # mN/m received, converted inside
    theta:  float = Form(...),
    depth:  float = Form(...),
):
    result = beggs_brill(
        q_l_m3d=q_l, q_g_m3d=q_g,
        d_m=d_mm / 1000.0,
        rho_l=rho_l, rho_g=rho_g,
        mu_l=mu_l / 1000.0,
        mu_g=1.8e-5,
        sigma=sigma / 1000.0,
        theta_deg=theta,
        depth_m=depth,
    )
    from fastapi.responses import JSONResponse
    return JSONResponse(result)


@app.post("/api/scale")
async def api_scale(
    ca2:          float = Form(...),
    mg2:          float = Form(...),
    na:           float = Form(...),
    hco3:         float = Form(...),
    so4:          float = Form(...),
    cl:           float = Form(...),
    temp_c:       float = Form(...),
    pressure_atm: float = Form(...),
):
    result = stiff_davis_index(ca2, mg2, na, hco3, so4, cl, temp_c, pressure_atm)
    from fastapi.responses import JSONResponse
    return JSONResponse(result)


@app.post("/api/esp")
async def api_esp(
    q_pts:    str   = Form(...),
    h_pts:    str   = Form(...),
    gor:      float = Form(...),
    bsw:      float = Form(...),
    sand_ppm: float = Form(...),
    visc_cp:  float = Form(...),
):
    q_list = json.loads(q_pts)
    h_list = json.loads(h_pts)
    result = esp_degradation(q_list, h_list, gor, bsw, sand_ppm, visc_cp)
    from fastapi.responses import JSONResponse
    return JSONResponse(result)


VIZUAL_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,viewport-fit=cover">
<meta name="theme-color" content="#07090f">
<title>PetroCalc — Визуализация</title>
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{
  --bg:#07090f;--bg2:#0c1018;--bg3:#111720;
  --panel:rgba(11,17,28,0.95);
  --b:rgba(251,191,36,0.12);--bb:rgba(251,191,36,0.35);
  --ac:#fbbf24;--gr:#10b981;--re:#f43f5e;--bl:#38bdf8;--vi:#8b5cf6;--or:#f97316;
  --tx:#e2e8f0;--mu:#64748b;
  --ff:'Rajdhani',sans-serif;--fm:'JetBrains Mono',monospace;--fb:'Inter',sans-serif;
  --nav-h:62px;--bot-h:60px;
}
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--tx);font-family:var(--fb);min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;
  background-image:
    radial-gradient(circle at 20% 20%,rgba(251,191,36,.03) 0%,transparent 50%),
    radial-gradient(circle at 80% 80%,rgba(16,185,129,.03) 0%,transparent 50%),
    linear-gradient(rgba(251,191,36,.02) 1px,transparent 1px),
    linear-gradient(90deg,rgba(251,191,36,.02) 1px,transparent 1px);
  background-size:100% 100%,100% 100%,50px 50px,50px 50px;
  pointer-events:none;z-index:0}

/* NAV */
nav{position:sticky;top:0;z-index:200;height:var(--nav-h);
    background:rgba(7,9,15,.97);backdrop-filter:blur(24px);
    border-bottom:1px solid var(--b);padding:0 24px;
    display:flex;align-items:center;justify-content:space-between}
.logo{font-family:var(--ff);font-size:22px;font-weight:700;letter-spacing:2px;
      display:flex;align-items:center;gap:10px;text-decoration:none;color:var(--tx)}
.bolt{width:28px;height:28px;background:linear-gradient(135deg,var(--ac),var(--vi));
      clip-path:polygon(50% 0%,80% 40%,55% 40%,55% 100%,20% 55%,48% 55%);
      animation:pulse 2s infinite;flex-shrink:0}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
.logo span{color:var(--ac)}
.nav-r{display:flex;align-items:center;gap:8px}
.nav-link{padding:6px 14px;border-radius:20px;border:1px solid var(--b);color:var(--mu);
          font-family:var(--fm);font-size:11px;text-decoration:none;transition:all .2s}
.nav-link:hover{border-color:var(--bb);color:var(--ac)}
.ver{font-family:var(--fm);font-size:10px;border:1px solid var(--bb);color:var(--ac);padding:4px 10px;border-radius:20px}

/* TABS */
.desk-tabs{display:flex;gap:3px;background:var(--bg3);border:1px solid var(--b);
           border-radius:14px;padding:4px;margin:20px 24px 0;flex-wrap:wrap}
.dtb{flex:1;min-width:120px;padding:9px 12px;background:transparent;border:none;
     cursor:pointer;color:var(--mu);font-family:var(--ff);font-size:13px;font-weight:600;
     border-radius:10px;transition:all .2s;display:flex;align-items:center;justify-content:center;gap:6px}
.dtb:hover{color:var(--tx);background:rgba(251,191,36,.06)}
.dtb.on{background:linear-gradient(135deg,rgba(251,191,36,.15),rgba(16,185,129,.1));
        color:var(--ac);border:1px solid var(--bb)}

/* MOBILE BOTTOM NAV */
.mob-nav{display:none;position:fixed;bottom:0;left:0;right:0;z-index:200;
         height:var(--bot-h);background:rgba(7,9,15,.98);
         backdrop-filter:blur(24px);border-top:1px solid var(--b);
         padding-bottom:env(safe-area-inset-bottom)}
.mob-nav-inner{display:flex;height:100%}
.mnb{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
     gap:2px;background:transparent;border:none;cursor:pointer;color:var(--mu);
     font-size:9px;font-family:var(--fb);padding:4px 2px;min-height:44px}
.mnb.on{color:var(--ac)}
.mnb.on .mni{background:rgba(251,191,36,.12);border-radius:9px}
.mni{font-size:19px;padding:3px 10px;transition:background .2s}

/* MAIN */
main{padding:20px 24px 40px;position:relative;z-index:1}
.tc{display:none}.tc.on{display:block;animation:fi .28s ease}
@keyframes fi{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}

/* CARD */
.card{background:var(--panel);border:1px solid var(--b);border-radius:18px;
      backdrop-filter:blur(20px);overflow:hidden;margin-bottom:16px}
.ch{padding:16px 22px;border-bottom:1px solid var(--b);display:flex;align-items:center;gap:12px}
.ci{width:38px;height:38px;background:linear-gradient(135deg,rgba(251,191,36,.2),rgba(16,185,129,.15));
    border:1px solid var(--bb);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:17px;flex-shrink:0}
.ct{font-family:var(--ff);font-size:16px;font-weight:700;letter-spacing:.8px}
.cs{font-size:11px;color:var(--mu);margin-top:2px}
.cb{padding:20px}

/* GRID */
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}

/* CONTROLS */
.fg{display:flex;flex-direction:column;gap:5px}
.fl{font-family:var(--fm);font-size:10px;color:var(--mu);letter-spacing:1.5px;text-transform:uppercase;display:flex;justify-content:space-between}
input[type=range]{width:100%;accent-color:var(--ac);cursor:pointer;height:4px}
.btn-row{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}
.fbtn{padding:7px 14px;border-radius:20px;border:1px solid var(--b);background:transparent;
      color:var(--mu);font-family:var(--fm);font-size:11px;cursor:pointer;transition:all .2s;min-height:36px}
.fbtn:hover{border-color:var(--bb);color:var(--tx)}
.fbtn.on{background:rgba(251,191,36,.12);border-color:var(--bb);color:var(--ac)}

/* STAT CARDS */
.rg{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px}
.rg2{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin-top:10px}
.rc{background:rgba(7,9,15,.7);border:1px solid var(--b);border-radius:12px;
    padding:14px 12px;text-align:center}
.rl{font-family:var(--fm);font-size:9px;color:var(--mu);letter-spacing:1px;text-transform:uppercase}
.rv{font-family:var(--ff);font-size:26px;font-weight:700;line-height:1;margin:6px 0 2px}
.ru{font-family:var(--fm);font-size:10px;color:var(--mu)}
.ca{border-color:rgba(251,191,36,.4)}.ce{border-color:rgba(16,185,129,.4)}
.cr{border-color:rgba(244,63,94,.4)}.cb2{border-color:rgba(56,189,248,.4)}
.cv{border-color:rgba(139,92,246,.4)}.co{border-color:rgba(249,115,22,.4)}
.tg{color:var(--ac)}.tm{color:var(--gr)}.tr{color:var(--re)}.tb{color:var(--bl)}.tv{color:var(--vi)}.to{color:var(--or)}

/* BANNERS */
.infob,.warn,.okb,.errb{margin-top:10px;padding:12px 15px;border-radius:10px;font-size:13px;display:flex;gap:8px;align-items:flex-start;line-height:1.5}
.infob{background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.3);color:#fde68a}
.warn{background:rgba(249,115,22,.08);border:1px solid rgba(249,115,22,.3);color:#fdba74}
.okb{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.3);color:#34d399}
.errb{background:rgba(244,63,94,.08);border:1px solid rgba(244,63,94,.3);color:#fb7185}

/* DIVIDER */
.div{border:none;height:1px;background:var(--b);margin:16px 0}
.sec{font-family:var(--fm);font-size:10px;color:var(--mu);letter-spacing:2px;
     text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:8px}
.sec::after{content:'';flex:1;height:1px;background:var(--b)}

/* WELL SVG */
.well-grid{display:grid;grid-template-columns:260px 1fr;gap:16px;align-items:start}
.well-svg-box{background:rgba(7,9,15,.7);border:1px solid var(--b);border-radius:12px;padding:10px}
.leg-item{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--mu);margin-bottom:7px}
.leg-dot{width:11px;height:11px;border-radius:50%;flex-shrink:0}
.well-params{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}

/* MAP */
.map-box{background:rgba(7,9,15,.7);border:1px solid var(--b);border-radius:12px;padding:10px}
.map-leg{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}
.ml{display:flex;align-items:center;gap:6px;font-size:11px;color:var(--mu)}
.ml-d{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.well-popup{display:none;background:rgba(7,9,15,.95);border:1px solid var(--bb);
            border-radius:12px;padding:14px;margin-top:10px}

/* ESP */
.esp-grid{display:grid;grid-template-columns:1fr;gap:12px}
.chart-box{background:rgba(7,9,15,.7);border:1px solid var(--b);border-radius:12px;padding:14px}
.ctrl-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;
           background:rgba(7,9,15,.7);border:1px solid var(--b);border-radius:12px;padding:14px}
.ctrl-row{display:flex;flex-direction:column;gap:4px}
.ctrl-l{font-family:var(--fm);font-size:10px;color:var(--mu);display:flex;justify-content:space-between}

/* FLOW */
@keyframes flow-move{to{stroke-dashoffset:-24}}
@keyframes pulse-glow{0%,100%{opacity:.6}50%{opacity:1}}

/* RESPONSIVE */
@media(max-width:900px){
  main{padding:14px 14px 26px}
  .desk-tabs{margin:14px 14px 0}
  .well-grid{grid-template-columns:1fr}
  .g3,.g4{grid-template-columns:1fr 1fr}
  .rg{grid-template-columns:1fr 1fr}
}
@media(max-width:640px){
  :root{--bot-h:58px}
  nav{padding:0 14px}
  main{padding:10px 10px 18px}
  .desk-tabs{display:none}
  .mob-nav{display:block}
  body{padding-bottom:calc(var(--bot-h) + env(safe-area-inset-bottom))}
  .g2,.g3,.g4{grid-template-columns:1fr}
  .rg{grid-template-columns:1fr 1fr}
  .ctrl-grid{grid-template-columns:1fr}
  .rv{font-size:22px}
}
</style>
</head>
<body>

<!-- NAV -->
<nav>
  <a href="/" class="logo"><div class="bolt"></div>PETRO<span>CALC</span></a>
  <div class="nav-r">
    <a href="/" class="nav-link">← Калькуляторы</a>
    <div class="ver">VIZ 1.0</div>
  </div>
</nav>

<!-- DESKTOP TABS -->
<div class="desk-tabs">
  <button class="dtb on" onclick="showTab('well',this)">🛢️ Схема скважины</button>
  <button class="dtb" onclick="showTab('flow',this)">🌊 Поток флюида</button>
  <button class="dtb" onclick="showTab('map',this)">🗺️ Карта месторождения</button>
  <button class="dtb" onclick="showTab('esp',this)">⚙️ Кривые ЭЦН</button>
</div>

<!-- MOBILE BOTTOM NAV -->
<div class="mob-nav">
  <div class="mob-nav-inner">
    <button class="mnb on" onclick="showTab('well',this)" data-tab="well">
      <div class="mni">🛢️</div><span>Скважина</span>
    </button>
    <button class="mnb" onclick="showTab('flow',this)" data-tab="flow">
      <div class="mni">🌊</div><span>Поток</span>
    </button>
    <button class="mnb" onclick="showTab('map',this)" data-tab="map">
      <div class="mni">🗺️</div><span>Карта</span>
    </button>
    <button class="mnb" onclick="showTab('esp',this)" data-tab="esp">
      <div class="mni">⚙️</div><span>ЭЦН</span>
    </button>
  </div>
</div>

<main>

<!-- ════════════════════ WELL SCHEMA ════════════════════ -->
<div id="tab-well" class="tc on">
<div class="card">
  <div class="ch"><div class="ci">🛢️</div>
    <div><div class="ct">СХЕМА СКВАЖИНЫ С ЭЦН</div>
         <div class="cs">Профиль · Зоны пластов · Перфорации · Насос</div></div>
  </div>
  <div class="cb">
    <p class="sec">Параметры скважины</p>
    <div class="g3" style="margin-bottom:16px">
      <div class="fg">
        <label class="fl"><span>Глубина забоя, м</span><span id="w-depth-v">2000</span></label>
        <input type="range" id="w-depth" min="500" max="5000" value="2000" step="100" oninput="updateWell()">
      </div>
      <div class="fg">
        <label class="fl"><span>Глубина спуска ЭЦН, м</span><span id="w-esp-v">900</span></label>
        <input type="range" id="w-esp" min="100" max="1800" value="900" step="50" oninput="updateWell()">
      </div>
      <div class="fg">
        <label class="fl"><span>Кровля нефт. пласта, м</span><span id="w-oil-v">1500</span></label>
        <input type="range" id="w-oil" min="300" max="4000" value="1500" step="50" oninput="updateWell()">
      </div>
    </div>

    <div class="well-grid">
      <div class="well-svg-box">
        <svg id="well-svg" width="100%" viewBox="0 0 240 560" role="img">
          <title>Схема скважины</title>
        </svg>
      </div>
      <div>
        <p class="sec">Легенда</p>
        <div class="leg-item"><div class="leg-dot" style="background:#f59e0b"></div>Нефтяной пласт</div>
        <div class="leg-item"><div class="leg-dot" style="background:#3b82f6"></div>Водоносный пласт</div>
        <div class="leg-item"><div class="leg-dot" style="background:#10b981"></div>Газовая шапка</div>
        <div class="leg-item"><div class="leg-dot" style="background:#8b5cf6"></div>Переходная зона</div>
        <div class="leg-item"><div class="leg-dot" style="background:#1d4ed8"></div>ЭЦН (насос)</div>
        <div class="leg-item"><div class="leg-dot" style="background:#f59e0b;opacity:.5"></div>Перфорации</div>
        <div class="leg-item"><div class="leg-dot" style="background:#f59e0b;opacity:.3;border:1px dashed #f59e0b"></div>Поток флюида ↑</div>

        <hr class="div">
        <p class="sec">Показатели</p>
        <div class="well-params">
          <div class="rc ca"><div class="rl">Дебит</div><div class="rv tg" id="w-stat-q">200</div><div class="ru">м³/сут</div></div>
          <div class="rc ce"><div class="rl">Забой</div><div class="rv tm" id="w-stat-d">2000</div><div class="ru">м</div></div>
          <div class="rc cb2"><div class="rl">ЭЦН</div><div class="rv tb" id="w-stat-esp">900</div><div class="ru">м</div></div>
          <div class="rc co"><div class="rl">Пласт</div><div class="rv to" id="w-stat-oil">1500</div><div class="ru">м кровля</div></div>
        </div>
      </div>
    </div>
  </div>
</div>
</div>

<!-- ════════════════════ FLOW ════════════════════ -->
<div id="tab-flow" class="tc">
<div class="card">
  <div class="ch"><div class="ci">🌊</div>
    <div><div class="ct">АНИМАЦИЯ МНОГОФАЗНОГО ПОТОКА</div>
         <div class="cs">Режимы течения · ГФ · Структура потока в трубе</div></div>
  </div>
  <div class="cb">
    <p class="sec">Режим течения</p>
    <div class="btn-row">
      <button class="fbtn on" onclick="setFlowMode('slug',this)">🔵 Пробковый (slug)</button>
      <button class="fbtn" onclick="setFlowMode('bubble',this)">⚪ Пузырьковый (bubble)</button>
      <button class="fbtn" onclick="setFlowMode('annular',this)">🔘 Кольцевой (annular)</button>
      <button class="fbtn" onclick="setFlowMode('segregated',this)">➖ Расслоённый</button>
    </div>

    <div class="g2" style="margin-bottom:16px">
      <div class="fg">
        <label class="fl"><span>Газовый фактор (ГФ)</span><span id="fl-gor-v">100 м³/м³</span></label>
        <input type="range" id="fl-gor" min="0" max="500" value="100" oninput="updateFlow()">
      </div>
      <div class="fg">
        <label class="fl"><span>Скорость потока</span><span id="fl-spd-v">1.0 м/с</span></label>
        <input type="range" id="fl-spd" min="1" max="10" value="5" oninput="updateFlow()">
      </div>
    </div>

    <div style="background:rgba(7,9,15,.7);border:1px solid var(--b);border-radius:12px;padding:12px;overflow:hidden">
      <svg id="flow-svg" width="100%" viewBox="0 0 600 200" role="img">
        <title>Анимация потока в трубе</title>
        <defs>
          <marker id="af" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          </marker>
        </defs>
        <!-- Pipe -->
        <rect x="10" y="50" width="580" height="100" rx="10" fill="#0f172a" stroke="#334155" stroke-width="1.2"/>
        <rect x="10" y="50" width="580" height="100" rx="10" fill="none" stroke="#475569" stroke-width=".4" opacity=".4"/>
        <!-- Inlet / outlet labels -->
        <text x="10" y="43" font-size="10" fill="#64748b" font-family="JetBrains Mono,monospace">Вход (НКТ)</text>
        <text x="470" y="43" font-size="10" fill="#64748b" font-family="JetBrains Mono,monospace">Выход (сепаратор)</text>
        <!-- Dynamic layers rendered by JS -->
        <g id="fl-layers"></g>
        <g id="fl-particles"></g>
        <!-- Velocity arrow -->
        <line id="fl-arrow" x1="20" y1="100" x2="580" y2="100"
              stroke="#f59e0b" stroke-width=".8" stroke-dasharray="10 6" opacity=".4"
              style="animation:flow-move 1s linear infinite" marker-end="url(#af)"/>
        <!-- Pipe label -->
        <text id="fl-mode-label" x="300" y="107" text-anchor="middle" font-size="11"
              fill="#fbbf24" font-family="JetBrains Mono,monospace" font-weight="600">Пробковый режим</text>
      </svg>
    </div>

    <div class="rg" style="margin-top:12px">
      <div class="rc ca"><div class="rl">Режим</div><div class="rv tg" id="fl-stat-mode" style="font-size:18px">Пробковый</div></div>
      <div class="rc cb2"><div class="rl">Доля газа</div><div class="rv tb" id="fl-stat-gas">9%</div></div>
      <div class="rc cm" style="border-color:rgba(16,185,129,.4)"><div class="rl">Удержание жидк.</div><div class="rv tm" id="fl-stat-hl">0.91</div></div>
      <div class="rc co"><div class="rl">Скорость смеси</div><div class="rv to" id="fl-stat-vm">1.0 м/с</div></div>
    </div>
  </div>
</div>
</div>

<!-- ════════════════════ MAP ════════════════════ -->
<div id="tab-map" class="tc">
<div class="card">
  <div class="ch"><div class="ci">🗺️</div>
    <div><div class="ct">КАРТА МЕСТОРОЖДЕНИЯ</div>
         <div class="cs">Скважины · Изолинии · Трубопроводы · Нажми на скважину</div></div>
  </div>
  <div class="cb">
    <div class="map-box">
      <svg id="map-svg" width="100%" viewBox="0 0 620 400" role="img">
        <title>Карта нефтяного месторождения</title>
        <defs>
          <marker id="am" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          </marker>
        </defs>
        <!-- Depth contours -->
        <ellipse cx="300" cy="200" rx="270" ry="170" fill="none" stroke="#f59e0b" stroke-width=".4" opacity=".1"/>
        <ellipse cx="300" cy="200" rx="220" ry="138" fill="none" stroke="#f59e0b" stroke-width=".5" opacity=".18"/>
        <ellipse cx="300" cy="200" rx="168" ry="105" fill="none" stroke="#f59e0b" stroke-width=".8" opacity=".26"/>
        <ellipse cx="300" cy="200" rx="115" ry="70" fill="none" stroke="#f59e0b" stroke-width="1" opacity=".38"/>
        <ellipse cx="300" cy="200" rx="65" ry="40" fill="none" stroke="#f59e0b" stroke-width="1.2" opacity=".52"/>
        <ellipse cx="300" cy="200" rx="28" ry="17" fill="#f59e0b" opacity=".1"/>
        <!-- Contour depth labels -->
        <text x="522" y="202" font-size="9" fill="#f59e0b" opacity=".45" font-family="JetBrains Mono,monospace">-1200 м</text>
        <text x="470" y="202" font-size="9" fill="#f59e0b" opacity=".55" font-family="JetBrains Mono,monospace">-1500</text>
        <text x="420" y="202" font-size="9" fill="#f59e0b" opacity=".65" font-family="JetBrains Mono,monospace">-1800</text>
        <!-- Field boundary -->
        <ellipse cx="300" cy="200" rx="250" ry="158" fill="none" stroke="#f59e0b" stroke-width=".5" stroke-dasharray="7 5" opacity=".3"/>
        <!-- Pipelines -->
        <path d="M148 108 Q224 128 300 138 Q378 128 458 112" fill="none" stroke="#94a3b8" stroke-width="1.4" opacity=".35"/>
        <path d="M162 302 Q238 282 300 272 Q362 282 438 308" fill="none" stroke="#94a3b8" stroke-width="1.4" opacity=".35"/>
        <line x1="300" y1="138" x2="300" y2="272" stroke="#94a3b8" stroke-width="1" opacity=".25"/>
        <!-- Pipeline arrows -->
        <line x1="202" y1="120" x2="268" y2="185" stroke="#94a3b8" stroke-width=".7" stroke-dasharray="4 3" opacity=".35" marker-end="url(#am)"/>
        <line x1="408" y1="118" x2="342" y2="185" stroke="#94a3b8" stroke-width=".7" stroke-dasharray="4 3" opacity=".35" marker-end="url(#am)"/>
        <line x1="196" y1="288" x2="268" y2="228" stroke="#3b82f6" stroke-width=".7" stroke-dasharray="4 3" opacity=".35" marker-end="url(#am)"/>
        <line x1="414" y1="298" x2="342" y2="228" stroke="#94a3b8" stroke-width=".7" stroke-dasharray="4 3" opacity=".35" marker-end="url(#am)"/>
        <!-- Collection point (UPN) -->
        <rect x="276" y="178" width="48" height="44" rx="8" fill="#1e3a5f" stroke="#3b82f6" stroke-width="1"/>
        <text x="300" y="204" text-anchor="middle" font-size="8" fill="#60a5fa" font-family="JetBrains Mono,monospace" font-weight="600">УПН</text>
        <!-- Wells -->
        <g style="cursor:pointer" onclick="selectWell('Скв. #1','Нефтяная','200 м³/сут','ЭЦН-200','150 атм','#f59e0b')">
          <circle cx="148" cy="108" r="11" fill="#f59e0b" opacity=".9" stroke="#fbbf24" stroke-width="1.5"/>
          <text x="148" y="112" text-anchor="middle" font-size="9" fill="#1c1209" font-family="JetBrains Mono,monospace" font-weight="700">#1</text>
          <text x="148" y="94" text-anchor="middle" font-size="9" fill="#fbbf24" font-family="JetBrains Mono,monospace">200 м³</text>
        </g>
        <g style="cursor:pointer" onclick="selectWell('Скв. #2','Нефтяная','155 м³/сут','ЭЦН-150','142 атм','#f59e0b')">
          <circle cx="458" cy="112" r="11" fill="#f59e0b" opacity=".9" stroke="#fbbf24" stroke-width="1.5"/>
          <text x="458" y="116" text-anchor="middle" font-size="9" fill="#1c1209" font-family="JetBrains Mono,monospace" font-weight="700">#2</text>
          <text x="458" y="98" text-anchor="middle" font-size="9" fill="#fbbf24" font-family="JetBrains Mono,monospace">155 м³</text>
        </g>
        <g style="cursor:pointer" onclick="selectWell('Скв. #3','Нагнетательная','300 м³/сут','ЦНС-300','200 атм','#3b82f6')">
          <circle cx="162" cy="302" r="11" fill="#3b82f6" opacity=".9" stroke="#60a5fa" stroke-width="1.5"/>
          <text x="162" y="306" text-anchor="middle" font-size="9" fill="#fff" font-family="JetBrains Mono,monospace" font-weight="700">#3</text>
          <text x="162" y="288" text-anchor="middle" font-size="9" fill="#60a5fa" font-family="JetBrains Mono,monospace">нагн.</text>
        </g>
        <g style="cursor:pointer" onclick="selectWell('Скв. #4 (ГРП)','Нефтяная после ГРП','385 м³/сут','ЭЦН-400','188 атм','#10b981')">
          <circle cx="438" cy="308" r="13" fill="#10b981" opacity=".9" stroke="#34d399" stroke-width="1.5"/>
          <text x="438" y="312" text-anchor="middle" font-size="9" fill="#fff" font-family="JetBrains Mono,monospace" font-weight="700">#4</text>
          <text x="438" y="293" text-anchor="middle" font-size="9" fill="#34d399" font-family="JetBrains Mono,monospace">385 м³★</text>
        </g>
        <g style="cursor:pointer" onclick="selectWell('Скв. #5','Остановлена','0 м³/сут','—','—','#64748b')">
          <circle cx="218" cy="162" r="9" fill="transparent" stroke="#64748b" stroke-width="1.5" stroke-dasharray="3 2"/>
          <text x="218" y="166" text-anchor="middle" font-size="9" fill="#64748b" font-family="JetBrains Mono,monospace">#5</text>
        </g>
        <g style="cursor:pointer" onclick="selectWell('Скв. #6','Газовая','52 тыс. м³/сут','—','90 атм','#8b5cf6')">
          <circle cx="374" cy="168" r="10" fill="#8b5cf6" opacity=".9" stroke="#a78bfa" stroke-width="1.5"/>
          <text x="374" y="172" text-anchor="middle" font-size="9" fill="#fff" font-family="JetBrains Mono,monospace" font-weight="700">#6</text>
          <text x="374" y="154" text-anchor="middle" font-size="9" fill="#a78bfa" font-family="JetBrains Mono,monospace">газ</text>
        </g>
      </svg>
    </div>

    <!-- Well popup -->
    <div id="well-popup" class="well-popup">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
        <span id="wp-name" style="font-size:14px;font-weight:600;font-family:var(--ff);letter-spacing:.5px"></span>
        <button onclick="document.getElementById('well-popup').style.display='none'"
                style="background:none;border:none;cursor:pointer;color:var(--mu);font-size:18px;line-height:1">×</button>
      </div>
      <div class="g3">
        <div class="rc"><div class="rl">Тип</div><div class="rv" id="wp-type" style="font-size:16px"></div></div>
        <div class="rc ca"><div class="rl">Дебит</div><div class="rv tg" id="wp-q" style="font-size:18px"></div></div>
        <div class="rc cb2"><div class="rl">Давление</div><div class="rv tb" id="wp-p" style="font-size:18px"></div></div>
      </div>
    </div>

    <div class="map-leg">
      <div class="ml"><div class="ml-d" style="background:#f59e0b"></div>Нефтяная</div>
      <div class="ml"><div class="ml-d" style="background:#3b82f6"></div>Нагнетательная</div>
      <div class="ml"><div class="ml-d" style="background:#10b981"></div>После ГРП</div>
      <div class="ml"><div class="ml-d" style="background:#8b5cf6"></div>Газовая</div>
      <div class="ml"><div class="ml-d" style="background:transparent;border:1px dashed #64748b"></div>Остановлена</div>
    </div>
  </div>
</div>
</div>

<!-- ════════════════════ ESP ════════════════════ -->
<div id="tab-esp" class="tc">
<div class="card">
  <div class="ch"><div class="ci">⚙️</div>
    <div><div class="ct">КРИВЫЕ Q–H НАСОСА ЭЦН</div>
         <div class="cs">Паспортная vs реальная · Деградация · Коэффициенты</div></div>
  </div>
  <div class="cb">
    <div class="esp-grid">
      <div class="chart-box">
        <canvas id="esp-chart" height="240"></canvas>
      </div>
      <div class="ctrl-grid">
        <div class="ctrl-row">
          <label class="ctrl-l"><span>Газовый фактор (ГФ)</span><span id="cd-gor">100 м³/м³</span></label>
          <input type="range" min="0" max="400" value="100" oninput="updateESP('gor',this.value)">
        </div>
        <div class="ctrl-row">
          <label class="ctrl-l"><span>Обводнённость</span><span id="cd-bsw">30%</span></label>
          <input type="range" min="0" max="95" value="30" oninput="updateESP('bsw',this.value)">
        </div>
        <div class="ctrl-row">
          <label class="ctrl-l"><span>Вязкость, сПз</span><span id="cd-visc">5</span></label>
          <input type="range" min="1" max="150" value="5" oninput="updateESP('visc',this.value)">
        </div>
        <div class="ctrl-row">
          <label class="ctrl-l"><span>Содержание песка, ppm</span><span id="cd-sand">50</span></label>
          <input type="range" min="0" max="3000" value="50" oninput="updateESP('sand',this.value)">
        </div>
      </div>
      <div class="rg">
        <div class="rc ca"><div class="rl">K суммарный</div><div class="rv tg" id="esp-k">0.92</div></div>
        <div class="rc cr"><div class="rl">Потеря подачи</div><div class="rv tr" id="esp-fl">8%</div></div>
        <div class="rc co"><div class="rl">Потеря напора</div><div class="rv to" id="esp-hl">5%</div></div>
        <div class="rc cb2"><div class="rl">ТОЭ факт.</div><div class="rv tb" id="esp-bep">184 м³/сут</div></div>
      </div>
      <div id="esp-banner" class="okb"><span>✅</span><span>Насос работает в штатном режиме.</span></div>
    </div>
  </div>
</div>
</div>

</main>

<script>
// ── TABS ──────────────────────────────────────
function showTab(id,btn){
  document.querySelectorAll('.tc').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('.dtb,.mnb').forEach(b=>b.classList.remove('on'));
  document.getElementById('tab-'+id).classList.add('on');
  document.querySelectorAll('.dtb').forEach(b=>{if(b.onclick&&b.getAttribute('onclick')&&b.getAttribute('onclick').includes("'"+id+"'"))b.classList.add('on')});
  document.querySelectorAll('.mnb').forEach(b=>{if(b.dataset.tab===id)b.classList.add('on')});
  if(id==='esp')setTimeout(drawESP,60);
  if(window.innerWidth<=640)window.scrollTo({top:0,behavior:'smooth'});
}

// ── WELL SVG ──────────────────────────────────
function updateWell(){
  const depth=parseInt(document.getElementById('w-depth').value);
  const espD=parseInt(document.getElementById('w-esp').value);
  const oilD=parseInt(document.getElementById('w-oil').value);
  document.getElementById('w-depth-v').textContent=depth;
  document.getElementById('w-esp-v').textContent=espD;
  document.getElementById('w-oil-v').textContent=oilD;
  document.getElementById('w-stat-q').textContent=Math.round(150+depth/20);
  document.getElementById('w-stat-d').textContent=depth;
  document.getElementById('w-stat-esp').textContent=espD;
  document.getElementById('w-stat-oil').textContent=oilD;
  drawWell(depth,espD,oilD);
}

function drawWell(depth,espD,oilD){
  const H=520; const scale=H/depth;
  const cx=120; const tubeW=40; const casingW=78;
  const top=30; const pipeH=depth*scale;
  const espY=top+espD*scale; const oilY=top+oilD*scale;
  const waterY=top+(oilD*0.75)*scale; const gasY=top+(oilD*1.18)*scale;
  const botY=top+pipeH;

  let svg=`
  <defs>
    <marker id="aw" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  <!-- Surface -->
  <rect x="0" y="0" width="240" height="${top}" rx="0" fill="#7a6648" opacity=".35"/>
  <text x="120" y="20" text-anchor="middle" font-size="10" fill="#a89070" font-family="JetBrains Mono,monospace">Поверхность (0 м)</text>
  <!-- Casing -->
  <rect x="${cx-casingW/2}" y="${top}" width="${casingW}" height="${pipeH}" rx="5"
        fill="none" stroke="#555" stroke-width="2" opacity=".5"/>
  <!-- Tubing -->
  <rect x="${cx-tubeW/2}" y="${top}" width="${tubeW}" height="${Math.min(espY-top+30,pipeH)}" rx="3"
        fill="none" stroke="#888" stroke-width="1.2"/>`;

  // Water zone
  if(waterY<botY){
    const wH=Math.min(oilY-waterY,botY-waterY);
    svg+=`<rect x="${cx-casingW/2+2}" y="${waterY}" width="${casingW-4}" height="${Math.max(wH,4)}"
               fill="#3b82f6" opacity=".18"/>`;
  }
  // Transition
  const trY=Math.min(oilY,botY); const trH=Math.min(20,botY-trY);
  if(trH>2) svg+=`<rect x="${cx-casingW/2+2}" y="${trY-trH}" width="${casingW-4}" height="${trH}" fill="#8b5cf6" opacity=".15"/>`;
  // Oil zone
  const oilH=Math.min(depth*0.2*scale, botY-oilY);
  if(oilH>4) svg+=`<rect x="${cx-casingW/2+2}" y="${oilY}" width="${casingW-4}" height="${oilH}" fill="#f59e0b" opacity=".22"/>`;
  // Gas cap
  const gasH=Math.min(depth*0.12*scale, botY-(oilY+oilH));
  if(gasH>4) svg+=`<rect x="${cx-casingW/2+2}" y="${oilY+oilH}" width="${casingW-4}" height="${gasH}" fill="#10b981" opacity=".16"/>`;

  // Perforations
  const pfY=oilY+5; const pfCount=Math.min(6,Math.floor(oilH/18));
  for(let i=0;i<pfCount;i++){
    const py=pfY+i*18;
    svg+=`<line x1="${cx-casingW/2}" y1="${py}" x2="${cx-casingW/2-12}" y2="${py}" stroke="#f59e0b" stroke-width="1.5"/>
          <circle cx="${cx-casingW/2-12}" cy="${py}" r="2.5" fill="#f59e0b"/>`;
  }
  if(pfCount>0) svg+=`<text x="${cx-casingW/2-20}" y="${pfY+pfCount*9}" text-anchor="end" font-size="9" fill="#f59e0b" font-family="JetBrains Mono,monospace">Перф.</text>`;

  // ESP
  const espBoxH=50;
  svg+=`<rect x="${cx-16}" y="${espY-espBoxH/2}" width="32" height="${espBoxH}" rx="5"
             fill="#1d4ed8" opacity=".85" stroke="#2563eb" stroke-width="1"/>
        <text x="${cx}" y="${espY-5}" text-anchor="middle" font-size="8" fill="#bfdbfe" font-family="JetBrains Mono,monospace" font-weight="600">ЭЦН</text>
        <text x="${cx}" y="${espY+8}" text-anchor="middle" font-size="7" fill="#93c5fd" font-family="JetBrains Mono,monospace">${espD}м</text>`;

  // Flow arrow
  svg+=`<line x1="${cx}" y1="${espY-espBoxH/2-5}" x2="${cx}" y2="${top+5}"
             stroke="#f59e0b" stroke-width="1.4" stroke-dasharray="6 4" opacity=".55"
             style="animation:flow-move 1.3s linear infinite" marker-end="url(#aw)"/>`;

  // Depth ruler
  svg+=`<line x1="72" y1="${top}" x2="72" y2="${botY}" stroke="#334155" stroke-width=".5"/>`;
  const ticks=[0,0.25,0.5,0.75,1.0];
  ticks.forEach(t=>{
    const y=top+t*pipeH; const d=Math.round(t*depth);
    svg+=`<line x1="68" y1="${y}" x2="72" y2="${y}" stroke="#475569" stroke-width=".8"/>
          <text x="64" y="${y+4}" text-anchor="end" font-size="8" fill="#475569" font-family="JetBrains Mono,monospace">${d}</text>`;
  });
  svg+=`<text x="58" y="${top+pipeH/2}" text-anchor="middle" font-size="8" fill="#475569"
             font-family="JetBrains Mono,monospace"
             transform="rotate(-90,58,${top+pipeH/2})">глубина, м</text>`;

  // Bottom plug
  svg+=`<rect x="${cx-casingW/2}" y="${botY}" width="${casingW}" height="7" rx="3" fill="#444" opacity=".5"/>`;
  svg+=`<text x="${cx}" y="${botY+20}" text-anchor="middle" font-size="8" fill="#475569"
             font-family="JetBrains Mono,monospace">${depth} м (забой)</text>`;

  document.getElementById('well-svg').innerHTML=svg;
  document.getElementById('well-svg').setAttribute('viewBox',`0 0 240 ${botY+30}`);
}
updateWell();

// ── FLOW ANIMATION ────────────────────────────
let flowMode='slug';
function setFlowMode(m,btn){
  flowMode=m;
  document.querySelectorAll('.fbtn').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  updateFlow();
}
function updateFlow(){
  const gor=parseInt(document.getElementById('fl-gor').value);
  const spd=parseFloat(document.getElementById('fl-spd').value)/5;
  document.getElementById('fl-gor-v').textContent=gor+' м³/м³';
  document.getElementById('fl-spd-v').textContent=(spd*2).toFixed(1)+' м/с';
  const fGas=Math.round(gor/(gor+1000)*100);
  const Hl=(1-fGas/100).toFixed(2);
  document.getElementById('fl-stat-gas').textContent=fGas+'%';
  document.getElementById('fl-stat-hl').textContent=Hl;
  document.getElementById('fl-stat-vm').textContent=(spd*2).toFixed(1)+' м/с';
  const modeNames={slug:'Пробковый',bubble:'Пузырьковый',annular:'Кольцевой',segregated:'Расслоённый'};
  document.getElementById('fl-stat-mode').textContent=modeNames[flowMode];
  document.getElementById('fl-mode-label').textContent=modeNames[flowMode]+' режим';
  renderFlowAnim(gor,spd);
}

function renderFlowAnim(gor,spd){
  const layers=document.getElementById('fl-layers');
  const parts=document.getElementById('fl-particles');
  const dur=(2/spd).toFixed(2);
  const liqH=Math.max(20,90-gor/6);
  const liqY=55+(100-liqH);
  layers.innerHTML=`<rect x="12" y="${liqY}" width="576" height="${liqH}" rx="3" fill="#3b82f6" opacity=".25"/>
    <line x1="12" y1="${liqY}" x2="588" y2="${liqY}" stroke="#3b82f6" stroke-width=".5" opacity=".4" stroke-dasharray="14 8" style="animation:flow-move ${dur}s linear infinite"/>`;
  let p='';
  if(flowMode==='bubble'){
    for(let i=0;i<14;i++){
      const x=40+i*40; const y=68+Math.sin(i)*22; const r=3+i%3;
      p+=`<circle cx="${x}" cy="${y}" r="${r}" fill="none" stroke="#10b981" stroke-width=".8" opacity=".65" style="animation:flow-move ${(parseFloat(dur)+i*0.15).toFixed(2)}s linear infinite"/>`;
    }
  } else if(flowMode==='slug'){
    const sc=3+Math.floor(gor/80);
    for(let i=0;i<Math.min(sc,5);i++){
      const x=40+i*108; const w=55+gor/15;
      p+=`<rect x="${x}" y="53" width="${w}" height="94" rx="5" fill="#10b981" opacity=".2" stroke="#10b981" stroke-width=".6" style="animation:flow-move ${(parseFloat(dur)+i*0.25).toFixed(2)}s linear infinite"/>`;
    }
  } else if(flowMode==='annular'){
    p=`<rect x="12" y="53" width="576" height="16" rx="3" fill="#10b981" opacity=".28"/>
       <rect x="12" y="131" width="576" height="16" rx="3" fill="#10b981" opacity=".22"/>
       <rect x="12" y="69" width="576" height="62" rx="0" fill="#3b82f6" opacity=".15"/>`;
  } else {
    p=`<rect x="12" y="53" width="576" height="46" rx="3" fill="#10b981" opacity=".2"/>
       <rect x="12" y="101" width="576" height="46" rx="3" fill="#3b82f6" opacity=".22"/>`;
  }
  parts.innerHTML=p;
}
updateFlow();

// ── MAP ───────────────────────────────────────
function selectWell(name,type,q,pump,p,color){
  const popup=document.getElementById('well-popup');
  popup.style.display='block';
  document.getElementById('wp-name').textContent=name+' — '+pump;
  document.getElementById('wp-name').style.color=color;
  document.getElementById('wp-type').textContent=type;
  document.getElementById('wp-q').textContent=q;
  document.getElementById('wp-p').textContent=p;
}

// ── ESP CHART ─────────────────────────────────
let espChart=null;
let espState={gor:100,bsw:30,visc:5,sand:50};
const Q_NOM=[0,50,100,150,200,250,300,350,400];
const H_NOM=[1620,1595,1530,1420,1268,1072,828,508,124];

function espDeg(){
  const s=espState;
  const fG=s.gor/(s.gor+1000);
  const kG=Math.max(0.2,1-0.5*fG-1.5*fG*fG);
  const v=s.visc;
  const kV=v<=1?1:Math.max(0.55,1-0.0012*Math.pow(v-1,0.9));
  const kVH=v<=1?1:Math.max(0.6,1-0.001*Math.pow(v-1,0.85));
  const kS=Math.max(0.5,1-s.sand/500000);
  const kT=kG*kV*kS;
  return{kT,kVH,kG,qD:Q_NOM.map(q=>+(q*kT).toFixed(1)),hD:H_NOM.map(h=>+(h*kVH*kG).toFixed(0))};
}

function drawESP(){
  const d=espDeg();
  const dk=window.matchMedia('(prefers-color-scheme:dark)').matches;
  const gc=dk?'rgba(255,255,255,0.05)':'rgba(0,0,0,0.06)';
  const tc=dk?'#94a3b8':'#64748b';
  if(espChart){espChart.destroy();espChart=null;}
  const ctx=document.getElementById('esp-chart').getContext('2d');
  espChart=new Chart(ctx,{
    type:'line',
    data:{
      labels:Q_NOM,
      datasets:[
        {label:'Паспортная кривая',data:H_NOM,borderColor:'#38bdf8',backgroundColor:'rgba(56,189,248,0.05)',borderWidth:2,tension:0.4,pointRadius:4,pointBackgroundColor:'#38bdf8'},
        {label:'Реальная (деградация)',data:d.hD,borderColor:'#f43f5e',backgroundColor:'rgba(244,63,94,0.05)',borderWidth:2,borderDash:[6,4],tension:0.4,pointRadius:4,pointBackgroundColor:'#f43f5e'}
      ]
    },
    options:{
      responsive:true,
      plugins:{
        legend:{labels:{color:tc,font:{size:11,family:'JetBrains Mono'}}},
        tooltip:{backgroundColor:dk?'#111720':'#fff',titleColor:dk?'#e2e8f0':'#0f172a',bodyColor:tc,borderColor:'rgba(251,191,36,.3)',borderWidth:1}
      },
      scales:{
        x:{title:{display:true,text:'Q, м³/сут',color:tc,font:{size:10}},grid:{color:gc},ticks:{color:tc,font:{size:10}}},
        y:{title:{display:true,text:'H, м (напор)',color:tc,font:{size:10}},grid:{color:gc},ticks:{color:tc,font:{size:10}}}
      }
    }
  });
  const k=+(d.kT).toFixed(3);
  const fl=+(100*(1-d.kT)).toFixed(1);
  const hl=+(100*(1-d.kVH*d.kG)).toFixed(1);
  const bep=+(200*d.kT).toFixed(0);
  document.getElementById('esp-k').textContent=k;
  document.getElementById('esp-fl').textContent=fl+'%';
  document.getElementById('esp-hl').textContent=hl+'%';
  document.getElementById('esp-bep').textContent=bep+' м³/сут';
  document.getElementById('esp-k').style.color=k>0.85?'#10b981':k>0.65?'#f59e0b':'#f43f5e';
  const banner=document.getElementById('esp-banner');
  if(k>0.85){banner.className='okb';banner.innerHTML='<span>✅</span><span>Насос работает в штатном режиме. Деградация минимальна.</span>';}
  else if(k>0.65){banner.className='warn';banner.innerHTML='<span>⚠️</span><span>Умеренная деградация — '+fl+'% потери подачи. Рекомендуется мониторинг.</span>';}
  else{banner.className='errb';banner.innerHTML='<span>🔴</span><span>Критическая деградация — '+fl+'% потери. Требуется вмешательство или замена насоса.</span>';}
}

function updateESP(k,v){
  espState[k]=parseFloat(v);
  if(k==='gor')document.getElementById('cd-gor').textContent=v+' м³/м³';
  if(k==='bsw')document.getElementById('cd-bsw').textContent=v+'%';
  if(k==='visc')document.getElementById('cd-visc').textContent=v+' сПз';
  if(k==='sand')document.getElementById('cd-sand').textContent=v+' ppm';
  drawESP();
}
// init ESP chart on load
setTimeout(drawESP,200);
</script>
</body>
</html>"""


@app.get("/vizual", response_class=HTMLResponse)
async def vizual():
    return VIZUAL_HTML


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
