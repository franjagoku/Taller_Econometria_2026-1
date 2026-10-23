# Taller 1 — Metodología Box-Jenkins: crecimiento anual del PIB del Reino Unido (100 trimestres)
# Econometría II, Facultad de Ciencias Económicas, Universidad Nacional de Colombia
#
# Serie: LORSGPORGBQ659S (FRED / OECD) — "Composite Leading Indicators: Reference Series (GDP)
#        for the United Kingdom, growth rate same period previous year", trimestral.
# Muestra de trabajo: las últimas N_OBS = 100 observaciones (1998 Q4 – 2023 Q3).
#

# %% Importación de paquetes ============================================================

import itertools, warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import jarque_bera, probplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_arch
from statsmodels.tools.sm_exceptions import ConvergenceWarning, InterpolationWarning
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

import statsmodels.stats.stattools as smt


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

N_OBS          = 200                       # tamaño de muestra pedido
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

# %% PASO 2: ESTIMACIÓN DE MODELOS SARIMA CON DUMMIES

warnings.filterwarnings("ignore")

s = 4
d, D = 0, 0

p_rng = q_rng = range(0, 5)
P_rng = Q_rng = range(0, 2)

resultados, modelos = [], {}

# Verificar que las dummies correspondan a las observaciones de la serie
exog = exog.reindex(serie.index).fillna(0)

print(f"Número de dummies incluidas: {exog.shape[1]}")
print("Dummies:", list(exog.columns))

for p, q, P, Q in itertools.product(
    p_rng, q_rng, P_rng, Q_rng
):
    orden = (p, d, q)
    est = (P, D, Q, s)

    try:
        m = SARIMAX(
            serie,
            exog=exog,  # Incluir las dummies como variables exógenas
            order=orden,
            seasonal_order=est,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False
    ).fit(disp=False)


        clave = (orden, est)
        modelos[clave] = m

        resultados.append({
            "orden": orden,
            "estacional": est,
            "AIC": m.aic,
            "BIC": m.bic
        })

    except Exception as e:
        print(f"Falló {orden} x {est}: {e}")

tabla = (
    pd.DataFrame(resultados)
    .sort_values("BIC")
    .reset_index(drop=True)
)

# %% PASO 3: SELECCIÓN DE LOS 10 MEJORES MODELOS POR BIC

TOP_N = 10

# Los modelos ya fueron estimados en el Paso 2 para poder calcular el BIC.
top10 = tabla.nsmallest(TOP_N, "AIC").copy()

print("\n=== Top 10 modelos según AIC ===")
print(top10.to_string(index=False))

# %% PASO 5: PRUEBAS DE DIAGNÓSTICO DE LOS 10 MODELOS
def diagnosticos(modelo, orden, est, lags=(8, 12, 16), arch_lags=(4, 8, 12)):
    k = orden[1] + est[1] * est[3]
    resid = modelo.resid.iloc[max(k, 1):]

    gl = orden[0] + orden[2] + est[0] + est[2]
    lb = acorr_ljungbox(resid, lags=list(lags), model_df=gl, return_df=True)
    jb = smt.jarque_bera(resid)

    salida = {
        "LB_p(8)": lb["lb_pvalue"].iloc[0],
        "LB_p(12)": lb["lb_pvalue"].iloc[1],
        "LB_p(16)": lb["lb_pvalue"].iloc[2],
        "JB_p": jb[1],
        "Asimetría": stats.skew(resid),
        "Curtosis": stats.kurtosis(resid, fisher=False),
    }
    for n in arch_lags:
        salida[f"ARCH_p({n})"] = het_arch(resid, nlags=n)[1]
    return salida

filas = []

for i, (_, r) in enumerate(top10.iterrows(), start=1):
    clave = (r["orden"], r["estacional"])
    modelo = modelos[clave]

    diag = diagnosticos(modelo, *clave)

    filas.append({
        "Ranking BIC": i,
        "orden": clave[0],
        "estacional": clave[1],
        "AIC": modelo.aic,
        "BIC": modelo.bic,
        **diag
    })

comparacion = pd.DataFrame(filas)

print("\n=== Diagnósticos de los 10 mejores modelos ===")

cols_p = [
    c for c in comparacion.columns
    if c.endswith(")") or c == "JB_p"
]
cols_p = [c for c in cols_p if "_p" in c]

formato = {c: "{:.3f}".format for c in cols_p}
formato.update({
    "AIC": "{:.2f}".format,
    "BIC": "{:.2f}".format,
    "Asimetría": "{:.3f}".format,
    "Curtosis": "{:.3f}".format
})

print(comparacion.to_string(index=False, formatters=formato))

# %% Selección del modelo final según Ljung-Box y BIC

# Criterio Ljung-Box: no rechazar la ausencia de autocorrelación
criterio_lb = (
    (comparacion["LB_p(8)"] < 0.05) &
    (comparacion["LB_p(12)"] < 0.05) &
    (comparacion["LB_p(16)"] < 0.05)
)

comparacion["Cumple_LB"] = criterio_lb

# Solo se consideran los modelos que cumplen Ljung-Box
candidatos = comparacion.loc[
    comparacion["Cumple_LB"]
].sort_values("BIC")

print("\n=== Modelos que cumplen Ljung-Box, ordenados por BIC ===")
print(comparacion.to_string(index=False, formatters=formato))

if not candidatos.empty:
    # El modelo con menor BIC entre los que cumplen Ljung-Box
    mejor = candidatos.iloc[0]

    print("\n=== MODELO FINAL SELECCIONADO ===")
    print(f"Orden: {mejor['orden']}")
    print(f"Estacional: {mejor['estacional']}")
    print(f"BIC: {mejor['BIC']:.3f}")

    print("Ljung-Box:")
    print(f"  p(8):  {mejor['LB_p(8)']:.4f}")
    print(f"  p(12): {mejor['LB_p(12)']:.4f}")
    print(f"  p(16): {mejor['LB_p(16)']:.4f}")

    clave_final = (mejor["orden"], mejor["estacional"])
    MODELO_FINAL = modelos[clave_final]

else:
    print(
        "\nNinguno de los modelos cumple todos los criterios "
        "de Ljung-Box. No se seleccionó un modelo final."
    )
    MODELO_FINAL = None

# %% Estimación e impresión de parámetros del modelo seleccionado

if MODELO_FINAL is None:
    raise RuntimeError(
        "No se seleccionó un modelo. Revisa los criterios de Ljung-Box."
    )

estimacion_final = MODELO_FINAL

orden_final = estimacion_final.model.order
estacional_final = estimacion_final.model.seasonal_order

nombre_final = (
    f"SARIMA{orden_final}x{estacional_final}"
)

print(print(f"\n=== MODELO FINAL: SARIMA{MODELO_FINAL.model.order} x {MODELO_FINAL.model.seasonal_order} ==="))
print("Modelo:", nombre_final)
print(f"AIC: {estimacion_final.aic:.3f}")
print(f"BIC: {estimacion_final.bic:.3f}")

# Tabla de parámetros estimados, incluidas las dummies
tabla_est = pd.DataFrame({
    "Coeficiente": estimacion_final.params,
    "Error estándar": estimacion_final.bse,
    "Estadístico z": estimacion_final.tvalues,
    "p-valor": estimacion_final.pvalues,
    "Significativo al 5%": estimacion_final.pvalues < 0.05
})

print(f"\n=== PARÁMETROS DEL MODELO  SARIMA{MODELO_FINAL.model.order} x {MODELO_FINAL.model.seasonal_order}===")
print(tabla_est.round(4).to_string())

# Tabla de diagnósticos de los modelos evaluados
tabla_diag = comparacion.copy()

# Guardar las tablas
tabla_est.to_csv(
    RESULTADOS_DIR / "tabla_estimaciones.csv"
)
tabla_diag.to_csv(
    RESULTADOS_DIR / "tabla_diagnostico.csv",
    index=False
)


# %% GRÁFICAS DE DIAGNÓSTICO DEL MODELO FINAL

graficas_diagnostico(
    nombre_final,
    estimacion_final,
    exog,
    "03_diagnostico_modelo_final.png"
)

tabla_como_imagen(
    tabla_diag.round(3).fillna("—"),
    "Diagnóstico de residuos",
    "04_tabla_diagnostico.png"
)

tabla_como_imagen(
    tabla_est.round(3),
    "Parámetros estimados del modelo final",
    "04b_tabla_estimaciones.png"
)

e_final = residuos_validos(estimacion_final, exog)

jb_p = jarque_bera(e_final).pvalue
print(f"\nNormalidad: JB p-valor = {jb_p:.4f}")
print(
    "No se rechaza normalidad al 5%."
    if jb_p > 0.05
    else "Se rechaza normalidad al 5%."
)


# %% =====================================================================================
# ESTABILIDAD: OLS-CUSUM (y, opcional, quiebres en la media)
# ========================================================================================
# OLS-CUSUM (Ploberger–Krämer): S_t = Σ_{i≤t} ê_i / (σ̂ √n). Bajo H0 de parámetros estables se
# comporta como un puente browniano; se rechaza si sup|S_t| > 1.358 (5 %).


# Obtener residuos del modelo final
residuos = pd.Series(estimacion_final.resid).replace(
    [np.inf, -np.inf], np.nan
).dropna()

# Estandarizar los residuos
desv = residuos.std(ddof=1)

if len(residuos) < 10 or not np.isfinite(desv) or desv == 0:
    print("No hay suficientes residuos válidos para calcular el CUSUM.")
else:
    z = (residuos - residuos.mean()) / desv

    # CUSUM centrado: puente acumulado
    n = len(z)
    cusum = z.cumsum()
    cusum = cusum - (np.arange(1, n + 1) / n) * z.sum()

    # Bandas aproximadas de referencia al 5 %
    t = np.arange(1, n + 1) / n
    limite = 1.358 * np.sqrt(n * t * (1 - t))

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        residuos.index, cusum,
        label="CUSUM de residuos estandarizados"
    )
    ax.plot(
        residuos.index, limite,
        linestyle="--", color="red",
        label="Banda superior de referencia"
    )
    ax.plot(
        residuos.index, -limite,
        linestyle="--", color="red",
        label="Banda inferior de referencia"
    )
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_title(f"Diagnóstico CUSUM: {nombre_final}")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Suma acumulada")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        RESULTADOS_DIR / "cusum_modelo_final.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)

    print(
        "CUSUM guardado. La gráfica es un diagnóstico exploratorio; "
        "las bandas no constituyen por sí solas una prueba formal "
        "de estabilidad de parámetros para SARIMAX."
    )
# %% =====================================================================================

# %% PASO 4: PRONÓSTICO DEL SARIMA SELECCIONADO

idx_pron = pd.date_range(
    start=serie.index[-1],
    periods=PASOS + 1,
    freq=FREQ
)[1:]

# Para el horizonte futuro se supone que las dummies de impulso valen cero.
exog_fut = pd.DataFrame(
    0.0,
    index=idx_pron,
    columns=exog.columns
)

# Pronóstico del modelo SARIMA final
fc = estimacion_final.get_forecast(
    steps=PASOS,
    exog=exog_fut
)

puntual = np.asarray(fc.predicted_mean)
ic = fc.conf_int(alpha=0.05)

tabla_pron = pd.DataFrame({
    "Pronóstico": puntual,
    "IC95 inferior": np.asarray(ic.iloc[:, 0]),
    "IC95 superior": np.asarray(ic.iloc[:, 1])
}, index=idx_pron)

print(f"\n=== PRONÓSTICO: {nombre_final} ===")
print(tabla_pron.round(4).to_string())

tabla_pron.to_csv(
    RESULTADOS_DIR / "tabla_pronostico.csv"
)

# Gráfico del pronóstico
plt.figure(figsize=(11, 5))

plt.plot(
    serie.iloc[-40:],
    color="black",
    lw=1.2,
    label="Histórico"
)

plt.plot(
    idx_pron,
    puntual,
    linestyle="--",
    marker="o",
    label=f"Pronóstico {nombre_final}"
)

plt.fill_between(
    idx_pron,
    np.asarray(ic.iloc[:, 0]),
    np.asarray(ic.iloc[:, 1]),
    alpha=0.25,
    label="Intervalo de confianza del 95%"
)

plt.title(
    f"Pronóstico del PIB — {PAIS}, "
    f"{PASOS} trimestres"
)
plt.xlabel("Fecha")
plt.ylabel("Tasa de crecimiento (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTADOS_DIR / "06_pronostico.png",
    dpi=150
)
plt.show()

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
print(f"Pronóstico j=1..3: {np.round(puntual[:3], 3)}")
print(f"Resultados en    : {RESULTADOS_DIR}")

# %%
