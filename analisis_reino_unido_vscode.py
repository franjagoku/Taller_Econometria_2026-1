# Taller 1 — Metodología Box-Jenkins: crecimiento anual del PIB del Reino Unido (100 trimestres)
# Econometría II, Facultad de Ciencias Económicas, Universidad Nacional de Colombia
#
# Serie: LORSGPORGBQ659S (FRED / OECD) — "Composite Leading Indicators: Reference Series (GDP)
#        for the United Kingdom, growth rate same period previous year", trimestral.
# Muestra de trabajo: las últimas N_OBS = 100 observaciones (1998 Q4 – 2023 Q3).
#

# %% Importación de paquetes ============================================================

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_arch
from statsmodels.tools.sm_exceptions import ConvergenceWarning, InterpolationWarning
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

# ruptures NO es parte de las librerías básicas: la sección de quiebres es opcional
try:
    import ruptures as rpt
    HAY_RUPTURES = True
except ImportError:
    HAY_RUPTURES = False

warnings.filterwarnings("ignore", category=InterpolationWarning)  # KPSS: p-valor fuera de tabla
warnings.filterwarnings("ignore", category=ConvergenceWarning)    # se revisa la convergencia a mano
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)         # aviso de het_arch en statsmodels 0.15

# %% Parámetros del análisis (lo único que normalmente hay que tocar) ====================

N_OBS          = 100                       # tamaño de muestra pedido
FIN_MUESTRA    = "2023-09-30"              # último dato disponible (2023 Q3)
VENTANA_COVID  = ("2020-03-31", "2022-03-31")  # 2020 Q1 – 2022 Q1 (9 trimestres)
UMBRAL_OUTLIER = 3.5                       # |residuo estandarizado| que dispara una dummy adicional
D_MANUAL       = 0                         # orden de integración elegido (ver Paso 1)
MODELO_FINAL   = None                      # None = regla automática (BIC entre válidos); o p.ej. "AR(6)"
PASOS          = 10                        # horizonte de pronóstico (trimestres)
N_BOOT         = 5000                      # réplicas del bootstrap
SEMILLA        = 42
CORRER_SENSIBILIDAD = True                 # tabla "¿qué pasa con otros tratamientos del COVID?"

# %% Rutas relativas =====================================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:                           # celdas interactivas sin __file__
    BASE_DIR = Path.cwd()

# Carpeta donde está la base de datos (el Excel de FRED). Si no existe, se usa la subcarpeta "datos".
RUTA_DATOS_USUARIO = Path(r"C:\Users\LENOVO\OneDrive - Universidad Nacional de Colombia"
                          r"\Economía\Septimo semestre economía\Econometría II\Poster")
DATA_DIR       = RUTA_DATOS_USUARIO if RUTA_DATOS_USUARIO.exists() else BASE_DIR / "datos"
RESULTADOS_DIR = BASE_DIR / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

CODIGO_SERIE = "LORSGPORGBQ659S"
ruta_xlsx = DATA_DIR / f"{CODIGO_SERIE}.xlsx"
ruta_csv  = DATA_DIR / f"{CODIGO_SERIE}.csv"

# %% Carga de la base de datos ===========================================================

if ruta_xlsx.exists():
    raw = pd.read_excel(ruta_xlsx, sheet_name="Quarterly", parse_dates=["observation_date"])
elif ruta_csv.exists():
    raw = pd.read_csv(ruta_csv, parse_dates=["observation_date"])
else:
    raise FileNotFoundError(f"No encontré {ruta_xlsx.name} (ni .csv) en {DATA_DIR}")

raw = raw.set_index("observation_date")

# FRED fecha cada trimestre en su primer día (1998-10-01). Se lleva al último día del trimestre
# y se fija la frecuencia trimestral (necesaria para SARIMAX y para fechar el pronóstico).
raw.index = raw.index.to_period("Q").to_timestamp(how="end").normalize()
FREQ = pd.offsets.QuarterEnd(startingMonth=12)

serie_completa = pd.to_numeric(raw[CODIGO_SERIE], errors="coerce").dropna().asfreq(FREQ)
assert serie_completa.isna().sum() == 0, "La serie tiene trimestres faltantes"

serie = serie_completa[:FIN_MUESTRA].iloc[-N_OBS:]      # últimas 100 observaciones
PAIS = "Reino Unido"
print("=== Descripción de la serie ===")
print(f"Base completa : {len(serie_completa)} obs ({serie_completa.index[0].date()} — {serie_completa.index[-1].date()})")
print(f"Muestra usada : {len(serie)} obs ({serie.index[0].date()} — {serie.index[-1].date()})")
print(serie.describe().round(3).to_string())

# %% Utilidades ==========================================================================


def crear_dummies(indice, fechas):
    """Dummies de impulso (outlier aditivo): 1 en la fecha, 0 en el resto."""
    cols = {f"d_{pd.Timestamp(f).strftime('%Y%m')}": (indice == pd.Timestamp(f)).astype(float)
            for f in fechas}
    return pd.DataFrame(cols, index=indice)


def ajustar(y, orden, exog=None):
    """Estima por máxima verosimilitud (Kalman) probando varios optimizadores hasta converger."""
    mejor = None
    for metodo, iters in (("bfgs", 1000), ("lbfgs", 1000), ("nm", 5000)):
        mod = SARIMAX(y, exog=exog, order=orden, trend="c",
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(method=metodo, maxiter=iters, disp=False)
        if mejor is None or res.llf > mejor.llf:
            mejor = res
        if res.mle_retvals.get("converged", True):
            return res
    print(f"  ⚠ {orden}: ningún optimizador reportó convergencia; se usa el de mayor log-verosimilitud")
    return mejor


def nombre_modelo(orden):
    p, _, q = orden
    return f"AR({p})" if q == 0 else f"ARMA({p},{q})"


def residuos_validos(res, exog=None):
    """Residuos sin las primeras max(p,q) observaciones y SIN las observaciones con dummy
    (su residuo es ~0 por construcción y deformaría LB, JB, ARCH y el bootstrap)."""
    p, _, q = res.model.order
    e = res.resid.iloc[max(p, q, 1):]
    if exog is not None and exog.shape[1] > 0:
        e = e[exog.loc[e.index].sum(axis=1).values == 0]
    return e


def media_incondicional(res):
    """mu = a0 / (1 - a1 - ... - ap)   (en SARIMAX con trend='c', 'intercept' es a0, no la media)."""
    ar = [v for k, v in res.params.items() if k.startswith("ar.L")]
    return res.params["intercept"] / (1 - sum(ar))


def min_raices(res):
    ar = np.abs(res.arroots).min() if len(res.arroots) else np.inf
    ma = np.abs(res.maroots).min() if len(res.maroots) else np.inf
    return ar, ma


LAGS_LB = [8, 12, 16, 20]


def diagnostico(res, exog, nombre):
    """p-valores de LB (con df ajustados), JB y ARCH. Todos con H0 = 'supuesto cumplido'.
    Aproximación: al excluir los residuos de las obs. con dummy queda un 'hueco' en la serie de
    residuos; las autocorrelaciones se calculan como si fuera contigua (práctica habitual)."""
    p, _, q = res.model.order
    e = residuos_validos(res, exog)
    fila = {"Modelo": nombre}
    # Ljung-Box: Q ~ chi2(h - p - q). Solo tiene sentido si h > p + q.
    for h in LAGS_LB:
        if h > p + q:
            fila[f"LB({h})"] = acorr_ljungbox(e, lags=[h], model_df=p + q, return_df=True)["lb_pvalue"].iloc[0]
        else:
            fila[f"LB({h})"] = np.nan
    fila["JB"]   = jarque_bera(e).pvalue
    fila["A(4)"] = het_arch(e, nlags=4)[1]
    fila["A(8)"] = het_arch(e, nlags=8)[1]
    fila["A(12)"] = het_arch(e, nlags=12)[1]
    return fila


def graficas_diagnostico(nombre, res, exog, archivo):
    """Residuales, FAC de residuales, FACP de residuales² y Q-Q plot."""
    e = residuos_validos(res, exog)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(e, color="black", lw=1)
    ax[0, 0].axhline(0, color="red", lw=0.8, ls="--")
    ax[0, 0].set_title(f"Residuales — {nombre}")
    ax[0, 0].grid(alpha=0.3)
    plot_acf(e, lags=16, alpha=0.05, bartlett_confint=False, ax=ax[0, 1])
    ax[0, 1].set_title("FAC residuales (no autocorrelación)")
    ax[0, 1].set_ylim(-1, 1)
    plot_pacf(e ** 2, lags=16, alpha=0.05, method="ywm", ax=ax[1, 0])
    ax[1, 0].set_title("FACP residuales² (homocedasticidad)")
    ax[1, 0].set_ylim(-1, 1)
    probplot(e, dist="norm", plot=ax[1, 1])
    ax[1, 1].set_title("Q-Q plot residuales (normalidad)")
    ax[1, 1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / archivo, dpi=150)
    plt.show()


def tabla_como_imagen(df, titulo, archivo, ancho=12, escala=(1.15, 1.7)):
    fig, ax = plt.subplots(figsize=(ancho, 0.42 * len(df) + 1.2))
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(*escala)
    for j in range(len(df.columns)):
        t[0, j].set_text_props(fontweight="bold")
        t[0, j].set_facecolor("#D3D3D3")
    plt.title(titulo, fontsize=12, pad=14)
    plt.savefig(RESULTADOS_DIR / archivo, dpi=150, bbox_inches="tight")
    plt.show()


def fmt(x, d=3):
    return "—" if pd.isna(x) else f"{x:.{d}f}"


# %% =====================================================================================
# PASO 0: INSPECCIÓN Y TRATAMIENTO PRELIMINAR DE ATÍPICOS
# ========================================================================================
# Intuición: dos observaciones extremas (2020 Q2 = −22.6 % y 2021 Q2 = +24.4 %) separadas
# EXACTAMENTE 4 trimestres fabrican una autocorrelación parcial grande en el rezago 4 (y 8),
# que no es dinámica de la economía sino un artefacto. Por eso se tratan antes de identificar.

plt.figure(figsize=(11, 5))
plt.plot(serie, color="black", lw=1.2)
plt.axhline(0, color="black", lw=0.7, ls="--")
plt.axvspan(pd.Timestamp(VENTANA_COVID[0]), pd.Timestamp(VENTANA_COVID[1]),
            color="salmon", alpha=0.25, label="Ventana COVID (dummies)")
plt.title(f"Crecimiento anual del PIB — {PAIS} ({serie.index[0].year}–{serie.index[-1].year})")
plt.xlabel("Fecha")
plt.ylabel("Tasa de crecimiento (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "01_serie_original.png", dpi=150)
plt.show()

fechas_covid = list(serie[VENTANA_COVID[0]:VENTANA_COVID[1]].index)
print("Trimestres con |crecimiento| > 7 %:",
      [d.strftime("%Y-Q") + str(d.quarter) for d in serie.index[np.abs(serie) > 7]])
print("Ventana COVID (dummies)           :",
      [d.strftime("%Y-Q") + str(d.quarter) for d in fechas_covid])
print(f"Rango fuera de la ventana: [{serie.drop(fechas_covid).min():.2f}, {serie.drop(fechas_covid).max():.2f}] %")

# %% =====================================================================================
# PASO 1: IDENTIFICACIÓN
# ========================================================================================

# 1a) FAC y FACP. Fila superior: muestra completa (contaminada por COVID).
#     Fila inferior: tramo sin COVID (hasta 2019 Q4), que muestra la dinámica "real".
pre_covid = serie[:"2019-12-31"]
print(f"\nTramo pre-COVID para identificar: {len(pre_covid)} obs ({pre_covid.index[0].date()} — {pre_covid.index[-1].date()})")

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
for fila, (x, etiqueta) in enumerate([(serie, f"muestra completa (n={len(serie)}), con COVID"),
                                      (pre_covid, f"tramo pre-COVID (n={len(pre_covid)})")]):
    plot_acf(x, lags=20, alpha=0.05, bartlett_confint=False, ax=ax[fila, 0])
    ax[fila, 0].set_title(f"FAC — {etiqueta}")
    ax[fila, 0].set_ylim(-1, 1)
    plot_pacf(x, lags=20, alpha=0.05, method="ywm", ax=ax[fila, 1])
    ax[fila, 1].set_title(f"FACP — {etiqueta}")
    ax[fila, 1].set_ylim(-1, 1)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "02_fac_facp.png", dpi=150)
plt.show()

# Reglas de identificación Box-Jenkins:
#   FACP se corta en p y FAC decae           → AR(p)
#   FAC se corta en q y FACP decae           → MA(q)
#   ambas decaen                             → ARMA(p,q)
# Lectura esperada: en el tramo pre-COVID la FAC decae rápido (0.90, 0.66, 0.35, 0.08, ...) y la
# FACP tiene picos en los primeros rezagos (0.90, −0.74, ...) → componente AR de orden bajo/medio.
# El pico del rezago 4 de la muestra completa NO es dinámica: es el par de outliers del COVID.

# 1b) Estacionariedad: ADF (H0: raíz unitaria) y KPSS (H0: estacionaria). Se usan JUNTOS.
#     El ADF automático de Colab elegía 12 rezagos (el máximo) con 100 datos → casi sin potencia,
#     y los outliers COVID lo distorsionan. Aquí: rezago máximo de Schwert (≈4) y selección por BIC.


def pruebas_raiz_unitaria(x, etiqueta):
    maxlag = int(np.ceil(4 * (len(x) / 100) ** 0.25))
    adf = adfuller(x, maxlag=maxlag, autolag="BIC", regression="c")
    kp = kpss(x, regression="c", nlags="auto")
    return {"Muestra": etiqueta, "n": len(x), "ADF stat": adf[0], "ADF p-valor": adf[1],
            "ADF rezagos": adf[2], "KPSS stat": kp[0], "KPSS p-valor": kp[1]}


tabla_raiz = pd.DataFrame([
    pruebas_raiz_unitaria(serie, "completa (con COVID)"),
    pruebas_raiz_unitaria(pre_covid, "pre-COVID"),
])
print("\n=== Raíz unitaria: ADF (H0: raíz unitaria) y KPSS (H0: estacionaria) ===")
print(tabla_raiz.round(3).to_string(index=False))
print("(KPSS: statsmodels reporta 0.10 cuando el p-valor real es mayor)")

# Interpretación para decidir d (sin automatismos):
#   ADF rechaza  & KPSS no rechaza  → I(0)
#   ADF no rechaza & KPSS rechaza   → I(1) → diferenciar
#   resultados mezclados            → evidencia no concluyente: pesan la teoría y las raíces del modelo
# Aquí KPSS no rechaza estacionariedad en ninguna muestra; el ADF pre-COVID tiene poca potencia
# (persistencia alta, 85 datos) y el de la muestra completa está sesgado hacia rechazar por los
# outliers. Una tasa de crecimiento de un PIB (cuyo nivel es I(1)) es I(0) por construcción, y las
# raíces AR estimadas quedan fuera del círculo unitario → se trabaja con d = 0 (sobrediferenciar
# introduciría una raíz unitaria en la parte MA).
d = D_MANUAL
print(f"\nDecisión: d = {d}  (D_MANUAL; ver comentario sobre la lectura conjunta de ADF y KPSS)")

# %% =====================================================================================
# PASO 1b: DUMMIES DE IMPULSO (ventana COVID + atípicos residuales)
# ========================================================================================
# Una dummy de impulso en t equivale a tratar esa observación como faltante (outlier aditivo).
# Tras la ventana COVID se revisan los residuos de un AR(4) provisional y se agrega, de a una,
# la observación más extrema si supera UMBRAL_OUTLIER (detección iterativa tipo Chen–Liu).

fechas_dummy = list(fechas_covid)
P_PROV = 4
for it in range(6):
    D_tmp = crear_dummies(serie.index, fechas_dummy)
    r_tmp = ajustar(serie, (P_PROV, d, 0), D_tmp)
    e_tmp = residuos_validos(r_tmp, D_tmp)
    z = e_tmp / e_tmp.std()
    if np.abs(z).max() > UMBRAL_OUTLIER:
        nuevo = z.abs().idxmax()
        print(f"  Iteración {it + 1}: nuevo atípico {nuevo.date()} ({z[nuevo]:.2f}σ) → se agrega dummy")
        fechas_dummy.append(nuevo)
    else:
        print(f"  Iteración {it + 1}: ningún residuo supera {UMBRAL_OUTLIER}σ → fin")
        break

exog = crear_dummies(serie.index, sorted(fechas_dummy))
print(f"\nDummies finales ({exog.shape[1]}): {list(exog.columns)}")
print(f"Observaciones efectivas: {len(serie) - exog.shape[1]} de {len(serie)}")

# %% =====================================================================================
# PASO 2: ESTIMACIÓN
# ========================================================================================
# Modelos candidatos (con la serie ya tratada):
#   AR(1)  : base
#   AR(2), AR(3), AR(5), AR(6): picos de la FACP en rezagos bajos / 5
#   ARMA(1,1), ARMA(2,1), ARMA(2,2): alternativas mixtas
#   AR(8)  : REFERENCIA — el modelo que salió en Colab; se estima para compararlo con justicia
# Nota: SARIMAX maximiza la verosimilitud EXACTA (filtro de Kalman); las notas de clase desarrollan
# la verosimilitud CONDICIONAL. Son asintóticamente equivalentes; difieren un poco en muestras chicas.

ORDENES = [(1, d, 0), (2, d, 0), (3, d, 0), (4, d, 0), (5, d, 0), (6, d, 0), (8, d, 0),
           (1, d, 1), (2, d, 1), (2, d, 2)]
estimaciones = {}
for orden in ORDENES:
    nombre = nombre_modelo(orden)
    estimaciones[nombre] = ajustar(serie, orden, exog)
    print(f"  estimado {nombre}")

filas = []
for nombre, r in estimaciones.items():
    p, _, q = r.model.order
    ar_names = [k for k in r.params.index if k.startswith("ar.L")]
    ar_ar, ar_ma = min_raices(r)
    filas.append({
        "Modelo": nombre, "AIC": r.aic, "BIC": r.bic,
        "mín|raíz AR|": ar_ar, "mín|raíz MA|": ar_ma,
        "AR signif. 5%": f"{int((r.pvalues[ar_names] < 0.05).sum())}/{p}" if p else "—",
        "μ̂ (media incond.)": media_incondicional(r),
    })
tabla_est = pd.DataFrame(filas)
print("\n=== Resumen de estimaciones (todas con las mismas dummies) ===")
print(tabla_est.round(3).to_string(index=False))

# %% =====================================================================================
# PASO 3: VALIDACIÓN DE SUPUESTOS
# ========================================================================================
# Ljung-Box: H0 = sin autocorrelación; estadístico Q(h) = T(T+2) Σ ρ̂_k²/(T−k) ~ χ²(h − p − q).
# Solo se calcula para h > p + q (en Colab, LB(5) y LB(10) para un AR(8) tenían gl ≤ 0).
# JB: H0 normalidad.  ARCH(m): H0 = no hay heterocedasticidad condicional.

tabla_diag = pd.DataFrame([diagnostico(r, exog, n) for n, r in estimaciones.items()])
print("\n=== Diagnóstico de residuos (p-valores; > 0.05 = supuesto se cumple; — = no aplicable) ===")
print(tabla_diag.round(3).to_string(index=False, na_rep="—"))

resumen = tabla_est.merge(tabla_diag, on="Modelo")
lb_cols = [f"LB({h})" for h in LAGS_LB]
resumen["LB_ok"] = resumen[lb_cols].apply(lambda f: bool((f.dropna() > 0.05).all()), axis=1)
resumen["válido"] = (resumen["mín|raíz AR|"] > 1.01) & (resumen["mín|raíz MA|"] > 1.01) & resumen["LB_ok"]

# %% Selección del modelo final

# Regla: entre los modelos estacionarios, invertibles y sin autocorrelación residual, el de menor
# BIC (penaliza más la complejidad que el AIC; con T pequeño protege contra sobreajuste).
# Si hay empate práctico (<2 puntos de BIC), se prefiere el de menos parámetros (parsimonia).
validos = resumen[resumen["válido"]].copy()
if validos.empty:
    print("\n⚠ Ningún modelo pasa todos los filtros; se elige por BIC entre estacionarios/invertibles.")
    validos = resumen[(resumen["mín|raíz AR|"] > 1.01) & (resumen["mín|raíz MA|"] > 1.01)].copy()
validos["k_params"] = validos["Modelo"].map(lambda n: len(estimaciones[n].params))
bic_min = validos["BIC"].min()
cercanos = validos[validos["BIC"] <= bic_min + 2].sort_values(["k_params", "BIC"])
nombre_regla = cercanos["Modelo"].iloc[0]

nombre_final = MODELO_FINAL or nombre_regla
estimacion_final = estimaciones[nombre_final]
print("\n=== Selección ===")
print(resumen[["Modelo", "AIC", "BIC", "mín|raíz AR|", "mín|raíz MA|", "LB_ok", "válido"]].round(3).to_string(index=False))
print(f"\nMejor AIC entre válidos : {validos.sort_values('AIC')['Modelo'].iloc[0]}")
print(f"Mejor BIC entre válidos : {validos.sort_values('BIC')['Modelo'].iloc[0]}")
print(f"Modelo elegido          : {nombre_final}" + ("  (regla)" if MODELO_FINAL is None else "  (manual)"))

print(f"\n=== Estimación explícita — {nombre_final} ===")
coefs = pd.DataFrame({
    "coef": estimacion_final.params, "error est.": estimacion_final.bse,
    "z": estimacion_final.tvalues, "p-valor": estimacion_final.pvalues,
    "signif. 5%": estimacion_final.pvalues < 0.05,
}).round(4)
print(coefs.to_string())
print(f"μ̂ = a0/(1−Σa_i) = {media_incondicional(estimacion_final):.3f} %")
print(f"σ̂² = {estimacion_final.params['sigma2']:.4f}")
print("Raíces AR (módulo):", np.round(np.abs(estimacion_final.arroots), 3))

# %% Gráficas de diagnóstico: modelo final y AR(8) de referencia

graficas_diagnostico(nombre_final, estimacion_final, exog, "03_diagnostico_modelo_final.png")
if "AR(8)" in estimaciones and nombre_final != "AR(8)":
    graficas_diagnostico("AR(8) (referencia Colab)", estimaciones["AR(8)"], exog, "03b_diagnostico_AR8.png")

tabla_como_imagen(tabla_diag.round(3).fillna("—"), "Diagnóstico de residuos (p-valores)", "04_tabla_diagnostico.png")
tabla_como_imagen(tabla_est.round(3), "Resumen de estimaciones", "04b_tabla_estimaciones.png")
tabla_est.to_csv(RESULTADOS_DIR / "tabla_estimaciones.csv", index=False)
tabla_diag.to_csv(RESULTADOS_DIR / "tabla_diagnostico.csv", index=False)

e_final = residuos_validos(estimacion_final, exog)
print(f"\nNormalidad (JB p = {jarque_bera(e_final).pvalue:.3f}): "
      + ("se cumple" if jarque_bera(e_final).pvalue > 0.05
         else "se rechaza → los intervalos de pronóstico se calculan también por bootstrap"))

# %% =====================================================================================
# ESTABILIDAD: OLS-CUSUM (y, opcional, quiebres en la media)
# ========================================================================================
# OLS-CUSUM (Ploberger–Krämer): S_t = Σ_{i≤t} ê_i / (σ̂ √n). Bajo H0 de parámetros estables se
# comporta como un puente browniano; se rechaza si sup|S_t| > 1.358 (5 %).

n_e = len(e_final)
S = np.cumsum(e_final.values) / (e_final.std(ddof=1) * np.sqrt(n_e))
stat_cusum, p_cusum, _ = breaks_cusumolsresid(e_final.values, ddof=len(estimacion_final.params) - exog.shape[1])
print(f"\nOLS-CUSUM: estadístico = {stat_cusum:.3f}, p-valor = {p_cusum:.3f} → "
      + ("estable (no se rechaza H0)" if p_cusum > 0.05 else "posible inestabilidad"))

plt.figure(figsize=(10, 4))
plt.plot(e_final.index, S, color="steelblue", lw=1.5, label="OLS-CUSUM")
plt.axhline(1.358, color="red", ls="--", lw=1, label="±1.358 (5 %)")
plt.axhline(-1.358, color="red", ls="--", lw=1)
plt.axhline(0, color="black", lw=0.6, ls=":")
plt.title(f"OLS-CUSUM de los residuos — {nombre_final}")
plt.xlabel("Fecha")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "05_cusum.png", dpi=150)
plt.show()

# Quiebres en la media (opcional). 'ruptures' solo implementa el algoritmo de programación
# dinámica de mínimos cuadrados de Bai–Perron; NO entrega pruebas sup-F ni inferencia, y supone
# errores no autocorrelacionados. Úsese como exploración, sobre el tramo pre-COVID.
if HAY_RUPTURES:
    y_bp = pre_covid.values
    n_bp = len(y_bp)
    modelo_bp = rpt.Dynp(model="l2", min_size=12, jump=1).fit(y_bp)
    bic_bp = {}
    for m in range(0, 4):
        bkps = modelo_bp.predict(n_bkps=m) if m > 0 else [n_bp]
        rss = sum(np.sum((y_bp[i:j] - y_bp[i:j].mean()) ** 2) for i, j in zip([0] + bkps[:-1], bkps))
        bic_bp[m] = n_bp * np.log(rss / n_bp) + (m + 1) * np.log(n_bp)
    m_opt = min(bic_bp, key=bic_bp.get)
    print("\nQuiebres en la media (pre-COVID, exploratorio): BIC por m =",
          {int(k): round(float(v), 1) for k, v in bic_bp.items()}, "→ m óptimo =", m_opt)
    if m_opt > 0:
        bk = modelo_bp.predict(n_bkps=m_opt)
        print("  Fechas de quiebre:", [pre_covid.index[b - 1].date() for b in bk[:-1]])
else:
    print("\n(ruptures no está instalado: se omite la exploración de quiebres. Es opcional.)")

# %% =====================================================================================
# PASO 4: PRONÓSTICO
# ========================================================================================
# Pronóstico óptimo (esperanza condicional, mínimo ECM):  E_t y_{t+j}
# Error de pronóstico:  e_t(j) = ψ0 ε_{t+j} + ψ1 ε_{t+j−1} + ... + ψ_{j−1} ε_{t+1},  ψ0 = 1
# Varianza:             Var[e_t(j)] = σ² (ψ0² + ψ1² + ... + ψ_{j−1}²)
# IC al 95 % (normal):  E_t y_{t+j} ± 1.96 · sqrt(Var[e_t(j)])
# Los ψ salen de igualar coeficientes en  φ(L) ψ(L) = θ(L).

idx_pron = pd.date_range(serie.index[-1], periods=PASOS + 1, freq=FREQ)[1:]
exog_fut = pd.DataFrame(0.0, index=idx_pron, columns=exog.columns)   # no se espera otro COVID


def pronosticar(res, pasos, exog_futuro):
    """Pronóstico puntual, pesos ψ, varianza e intervalos analíticos (fórmulas del curso)."""
    par = res.params
    p, _, q = res.model.order
    phi = np.array([par[f"ar.L{i}"] for i in range(1, p + 1)])
    theta = np.array([par[f"ma.L{j}"] for j in range(1, q + 1)])
    psi = arma2ma(np.r_[1.0, -phi], np.r_[1.0, theta], lags=pasos)
    var_e = par["sigma2"] * np.cumsum(psi ** 2)
    fc = res.get_forecast(pasos, exog=exog_futuro)
    puntual = fc.predicted_mean.values
    assert np.allclose(var_e, fc.var_pred_mean, rtol=1e-4), "ψ-pesos y statsmodels no coinciden"
    return puntual, psi, var_e


def bootstrap_trayectorias(res, residuos, pasos, n_boot, exog_futuro, semilla):
    """Bootstrap de residuos: se remuestrean los residuos (centrados, SIN los de obs. con dummy) y se
    propagan por la ecuación ARMA; así el intervalo crece con el horizonte. No exige normalidad,
    pero tampoco incorpora la incertidumbre de estimación de los parámetros."""
    rng = np.random.default_rng(semilla)
    par = res.params
    p, _, q = res.model.order
    c = par["intercept"]
    a = np.array([par[f"ar.L{i}"] for i in range(1, p + 1)])
    b = np.array([par[f"ma.L{j}"] for j in range(1, q + 1)])
    cols = list(exog_futuro.columns)
    beta = par[cols].values
    u_hist = res.model.endog.ravel() - res.model.exog @ beta    # serie sin el efecto de las dummies
    e_hist = res.resid.values
    media_x = exog_futuro.values @ beta
    pool = residuos.values - residuos.values.mean()

    def una(choques):
        u = list(u_hist[-p:]) if p else []
        e = list(e_hist[-q:]) if q else []
        out = np.empty(pasos)
        for h in range(pasos):
            ar_part = np.dot(a, u[::-1][:p]) if p else 0.0
            ma_part = np.dot(b, e[::-1][:q]) if q else 0.0
            u_new = c + ar_part + ma_part + choques[h]
            u.append(u_new)
            e.append(choques[h])
            out[h] = u_new + media_x[h]
        return out

    puntual = una(np.zeros(pasos))
    sims = np.vstack([una(rng.choice(pool, size=pasos, replace=True)) for _ in range(n_boot)])
    return puntual, sims


puntual, psi, var_e = pronosticar(estimacion_final, PASOS, exog_fut)
sd_e = np.sqrt(var_e)
puntual_b, sims = bootstrap_trayectorias(estimacion_final, e_final, PASOS, N_BOOT, exog_fut, SEMILLA)
assert np.allclose(puntual, puntual_b, atol=1e-6), "La recursión del bootstrap no reproduce el pronóstico"

tabla_pron = pd.DataFrame({
    "pronóstico": puntual,
    "ψ_{j-1}": psi,
    "Var[e_t(j)]": var_e,
    "IC95 inf (analítico)": puntual - 1.96 * sd_e,
    "IC95 sup (analítico)": puntual + 1.96 * sd_e,
    "IC95 inf (bootstrap)": np.percentile(sims, 2.5, axis=0),
    "IC95 sup (bootstrap)": np.percentile(sims, 97.5, axis=0),
}, index=idx_pron)
j_max_conv = 80
mu_hat = media_incondicional(estimacion_final)
largo = estimacion_final.get_forecast(
    j_max_conv, exog=pd.DataFrame(0.0, index=pd.date_range(serie.index[-1], periods=j_max_conv + 1, freq=FREQ)[1:],
                                  columns=exog.columns)).predicted_mean.values
TOL_CONV = 0.25
fuera = np.where(np.abs(largo - mu_hat) >= TOL_CONV)[0]       # horizontes (0-indexados) fuera de la banda
j_conv = (fuera[-1] + 2) if len(fuera) else 1                 # primer j desde el cual PERMANECE dentro
print(f"\n=== Pronóstico {PASOS} trimestres — {nombre_final} ===")
print(tabla_pron.round(3).to_string())
print(f"\nMedia incondicional μ̂ = {mu_hat:.3f} %; Var[e_t(∞)] → γ0 (varianza incondicional).")
print(f"Desde j = {j_conv} el pronóstico queda de forma permanente a menos de {TOL_CONV} pp de μ̂ "
      "(con raíces complejas el pronóstico oscila alrededor de μ̂ antes de estabilizarse).")
tabla_pron.round(4).to_csv(RESULTADOS_DIR / "tabla_pronostico.csv")

plt.figure(figsize=(11, 5))
plt.plot(serie.iloc[-40:], color="black", lw=1.2, label="Histórico")
plt.plot(idx_pron, puntual, color="orange", lw=2, ls="--", marker="o", ms=4, label=f"Pronóstico {nombre_final}")
plt.fill_between(idx_pron, tabla_pron["IC95 inf (analítico)"], tabla_pron["IC95 sup (analítico)"],
                 color="orange", alpha=0.25, label="IC 95 % analítico (ψ-pesos)")
plt.plot(idx_pron, tabla_pron["IC95 inf (bootstrap)"], color="firebrick", lw=1, ls=":", label="IC 95 % bootstrap")
plt.plot(idx_pron, tabla_pron["IC95 sup (bootstrap)"], color="firebrick", lw=1, ls=":")
plt.axhline(mu_hat, color="gray", lw=0.8, ls="--", label=f"μ̂ = {mu_hat:.2f} %")
plt.title(f"Pronóstico — PIB {PAIS}, {PASOS} trimestres desde {serie.index[-1].year} Q{serie.index[-1].quarter}")
plt.xlabel("Fecha")
plt.ylabel("Tasa de crecimiento (%)")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "06_pronostico.png", dpi=150)
plt.show()

# %% =====================================================================================
# ¿ES UN PROBLEMA EL AR(8)?  Comparación directa con el modelo final
# ========================================================================================

if "AR(8)" in estimaciones and nombre_final != "AR(8)":
    r8 = estimaciones["AR(8)"]
    print("\n=== Coeficientes del AR(8) de referencia ===")
    print(pd.DataFrame({"coef": r8.params, "z": r8.tvalues, "p-valor": r8.pvalues})
          .loc[[k for k in r8.params.index if k.startswith("ar.L")]].round(3).to_string())
    p8, _, _ = r8.model.order
    print(f"AR(8): {len(r8.params)} parámetros estimados; se pierden {p8} obs. iniciales en el diagnóstico;"
          f" mín|raíz AR| = {min_raices(r8)[0]:.3f}")
    pr8, _, _ = pronosticar(r8, PASOS, exog_fut)
    comp = pd.DataFrame({nombre_final: puntual, "AR(8)": pr8, "diferencia": puntual - pr8}, index=idx_pron)
    print("\n=== Pronósticos: modelo final vs AR(8) ===")
    print(comp.round(3).to_string())

# %% Sensibilidad: ¿el orden elegido depende de cómo se trate el COVID?

if CORRER_SENSIBILIDAD:
    candidatos = [(1, d, 0), (2, d, 0), (3, d, 0), (4, d, 0), (5, d, 0), (6, d, 0), (8, d, 0),
                  (1, d, 1), (2, d, 1), (2, d, 2)]
    tratamientos = {
        "Sin tratamiento": crear_dummies(serie.index, []),
        "2 dummies (2020Q2, 2021Q2; como en Colab)": crear_dummies(serie.index, ["2020-06-30", "2021-06-30"]),
        f"Ventana COVID + atípicos ({exog.shape[1]} dummies)": exog,
    }
    filas_s = []
    for etiqueta, X in tratamientos.items():
        X_ = X if X.shape[1] > 0 else None
        tabla_t = []
        for orden in candidatos:
            r = ajustar(serie, orden, X_)
            ar_m, ma_m = min_raices(r)
            dg = diagnostico(r, X if X.shape[1] > 0 else crear_dummies(serie.index, []), nombre_modelo(orden))
            lb_ok = all(v > 0.05 for k, v in dg.items() if k.startswith("LB") and not pd.isna(v))
            tabla_t.append({"Modelo": nombre_modelo(orden), "AIC": r.aic, "BIC": r.bic,
                            "válido": lb_ok and ar_m > 1.01 and ma_m > 1.01})
        tt = pd.DataFrame(tabla_t)
        v = tt[tt["válido"]] if tt["válido"].any() else tt
        filas_s.append({"Tratamiento": etiqueta,
                        "Mejor AIC (todos)": tt.loc[tt["AIC"].idxmin(), "Modelo"],
                        "Mejor BIC (todos)": tt.loc[tt["BIC"].idxmin(), "Modelo"],
                        "Mejor BIC (válidos)": v.loc[v["BIC"].idxmin(), "Modelo"]})
    sens = pd.DataFrame(filas_s)
    print("\n=== Sensibilidad del modelo elegido al tratamiento del COVID ===")
    print(sens.to_string(index=False))
    sens.to_csv(RESULTADOS_DIR / "tabla_sensibilidad.csv", index=False)

# %% Resumen final

print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print(f"Muestra          : {serie.index[0].date()} — {serie.index[-1].date()} ({len(serie)} trimestres)")
print(f"Orden integración: d = {d}")
print(f"Dummies          : {exog.shape[1]} ({', '.join(exog.columns)})")
print(f"Modelo final     : {nombre_final}  | AIC = {estimacion_final.aic:.1f}  BIC = {estimacion_final.bic:.1f}")
print(f"Media incondic.  : {mu_hat:.3f} %")
print(f"Pronóstico j=1..3: {np.round(puntual[:3], 3)}")
print(f"Resultados en    : {RESULTADOS_DIR}")
