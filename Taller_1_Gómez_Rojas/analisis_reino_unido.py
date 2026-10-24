# Taller 1 Poster — Análisis de la tasa de crecimiento del PIB de Reino Unido (1998–2023)
# Autor: Lucía Rojas Gordillo y Francisco Javier Gómez
# Grupo 3: Jurgen Toloza
# Econometría II, Facultad de Ciencias Económicas, Universidad Nacional de Colombia
#
# Serie: LORSGPORGBQ659S (FRED / OECD) — "Composite Leading Indicators: Reference Series (GDP)
#        for the United Kingdom, growth rate same period previous year", trimestral.
# Muestra: últimas 100 observaciones (1998 Q4 – 2023 Q3).

# %% Importación de paquetes ============================

# Trabajar con rutas relativas en python
from pathlib import Path
import warnings

# Módulos de numpy, pandas, matplotlib, scipy y ruptures
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
from scipy.stats import jarque_bera, probplot

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid, het_arch
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

warnings.filterwarnings("ignore")   # avisos de convergencia y de tablas del KPSS

# %% Rutas relativas ============================

# BASE_DIR apunta a la carpeta raíz del proyecto, independientemente de dónde
# esté guardada en el computador.
BASE_DIR       = Path(__file__).resolve().parent
DATA_DIR       = BASE_DIR / "datos"
RESULTADOS_DIR = BASE_DIR / "resultados"
RESULTADOS_DIR.mkdir(exist_ok=True)

ruta_datos = DATA_DIR / "LORSGPORGBQ659S.xlsx"


# %% Carga de la base de datos ============================

raw = pd.read_excel(
    ruta_datos,
    sheet_name="Quarterly",
    parse_dates=["observation_date"]
)

raw = raw.set_index("observation_date")

# FRED fecha cada trimestre en su primer día; se lleva al último día del trimestre
raw.index = raw.index.to_period("Q").to_timestamp(how="end").normalize()
FREQ = pd.offsets.QuarterEnd(startingMonth=12)

serie_completa = pd.to_numeric(raw["LORSGPORGBQ659S"], errors="coerce").dropna().asfreq(FREQ)

# Se trabaja con las últimas 100 observaciones disponibles (1998 Q4 – 2023 Q3).
serie = serie_completa[:"2023-09-30"].iloc[-100:]

print("=== Descripción de la serie ===")
print(f"Observaciones : {len(serie)}")
print(f"Periodo       : {serie.index[0].date()} — {serie.index[-1].date()}")
print(serie.describe().round(3).to_string())


# %% Funciones auxiliares ============================

def ajustar(y, orden, exog=None):
    """Estima un ARMA por máxima verosimilitud (SARIMAX). Si BFGS no converge,
    prueba con L-BFGS y Nelder-Mead y se queda con la mayor log-verosimilitud."""
    mejor = None
    for metodo, iters in (("bfgs", 1000), ("lbfgs", 1000), ("nm", 5000)):
        modelo = SARIMAX(y, exog=exog, order=orden, trend="c",
                         enforce_stationarity=False, enforce_invertibility=False)
        res = modelo.fit(method=metodo, maxiter=iters, disp=False)
        if mejor is None or res.llf > mejor.llf:
            mejor = res
        if res.mle_retvals.get("converged", True):
            return res
    return mejor


def residuos_validos(res, exog):
    """Residuos sin las primeras max(p,q) observaciones y sin los trimestres con dummy
    (en esos trimestres el residuo es ~0 por construcción)."""
    p, _, q = res.model.order
    e = res.resid.iloc[max(p, q, 1):]
    return e[exog.loc[e.index].sum(axis=1).values == 0]


def media_incondicional(res):
    """Calcula E[y_t] = a0 / (1 - a1 - ... - ap). En SARIMAX con trend='c',
    'intercept' es a0, no la media."""
    suma_ar = sum(v for k, v in res.params.items() if k.startswith("ar.L"))
    return res.params["intercept"] / (1 - suma_ar)


def se_media_incondicional(res):
    """Error estándar de la media incondicional por el método delta."""
    params = res.params
    parametros_ar = [p for p in params.index if p.startswith("ar.L")]
    denominador = 1 - sum(params[p] for p in parametros_ar)
    gradiente = pd.Series(0.0, index=params.index)
    gradiente["intercept"] = 1 / denominador
    for p in parametros_ar:
        gradiente[p] = params["intercept"] / denominador**2
    return np.sqrt(float(gradiente @ res.cov_params() @ gradiente))


def tabla_como_imagen(df, titulo, archivo, ancho=11):
    """Guarda una tabla de pandas como imagen para el póster."""
    fig, ax = plt.subplots(figsize=(ancho, 0.45 * len(df) + 1.3))
    ax.axis("off")
    tabla_img = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center"
    )
    tabla_img.auto_set_font_size(False)
    tabla_img.set_fontsize(9)
    tabla_img.scale(1.2, 1.8)
    # Encabezado en negrilla y gris claro
    for j in range(len(df.columns)):
        tabla_img[0, j].set_text_props(fontweight="bold")
        tabla_img[0, j].set_facecolor("#D3D3D3")
    plt.title(titulo, fontsize=12, pad=20)
    plt.savefig(RESULTADOS_DIR / archivo, dpi=150, bbox_inches="tight")
    plt.show()


# %% =========================
# PASO 1: IDENTIFICACIÓN
# ============================

# Ventana COVID: 2020 Q1 – 2022 Q1. El −22.6 % (2020 Q2) y el +24.4 % (2021 Q2) son el mismo
# choque visto dos veces (efecto base de una tasa anual) y, al estar separados 4 trimestres,
# crean un pico espurio en el rezago 4 de la FACP. Por eso se tratan antes de identificar.
VENTANA_COVID = ("2020-03-31", "2022-03-31")

# Gráfica de la serie de tiempo
plt.figure(figsize=(11, 5))
plt.plot(serie, color="black", lw=1.2)
plt.axhline(0, color="black", lw=0.7, ls="--")
plt.axvspan(pd.Timestamp(VENTANA_COVID[0]), pd.Timestamp(VENTANA_COVID[1]),
            color="salmon", alpha=0.25, label="Ventana COVID (dummies)")
plt.title("Crecimiento anual del PIB — Reino Unido (1998–2023)")
plt.xlabel("Fecha")
plt.ylabel("Tasa de crecimiento (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "01_serie_original.png", dpi=150)
plt.show()

# %% Dummies de impulso para la ventana COVID

# Una dummy de impulso en t equivale a tratar esa observación como atípica (outlier aditivo).
# El modelo se estima con las 100 observaciones; las dummies no eliminan datos.
fechas_covid = serie[VENTANA_COVID[0]:VENTANA_COVID[1]].index
exog = pd.DataFrame({f"d_{f.strftime('%Y%m')}": (serie.index == f).astype(float)
                     for f in fechas_covid}, index=serie.index)
print(f"\nDummies ({exog.shape[1]}): {list(exog.columns)}")

# Serie corregida por las dummies: es la serie sin el efecto de las 9 dummies de una regresión
# de y sobre una constante y las dummies. En la práctica, los 9 trimestres COVID toman el valor
# de la media del resto de la muestra. Sigue teniendo 100 observaciones y solo se usa para leer
# la FAC y la FACP sin la distorsión del COVID.
serie_corregida = serie.copy()
serie_corregida[fechas_covid] = serie.drop(fechas_covid).mean()

# %% FAC y FACP: serie original y serie corregida por las dummies (ambas con 100 observaciones)

# Reglas de identificación Box-Jenkins:
# - FACP se corta en rezago p, FAC decae gradualmente → proceso AR(p)
# - FAC se corta en rezago q, FACP decae gradualmente → proceso MA(q)
# - Ambas decaen gradualmente                         → proceso ARMA(p,q)
#
# Serie original: la FACP tiene picos en 1, 4 y 8. El del rezago 4 es espurio: el −22.6 %
# de 2020 Q2 y el +24.4 % de 2021 Q2 están separados exactamente 4 trimestres.
# Serie corregida por las dummies: la FAC decae gradualmente (0.88, 0.63, 0.33, 0.07) y la
# FACP tiene picos significativos en los rezagos 1, 2 y 5 y se corta después del 5 → AR(5).
# Se proponen además AR(2) como base, AR(4) y AR(6) como órdenes vecinos y ARMA(2,2) como
# modelo mixto alternativo.
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
for fila, (x, etiqueta) in enumerate([(serie, "serie original"),
                                      (serie_corregida, "serie corregida por dummies COVID")]):
    plot_acf(x, lags=20, alpha=0.05, bartlett_confint=False, ax=ax[fila, 0])
    ax[fila, 0].set_title(f"FAC — {etiqueta}")
    ax[fila, 0].set_ylim(-1, 1)
    plot_pacf(x, lags=20, alpha=0.05, method="ywm", ax=ax[fila, 1])
    ax[fila, 1].set_title(f"FACP — {etiqueta}")
    ax[fila, 1].set_ylim(-1, 1)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "02_fac_facp.png", dpi=150)
plt.show()

# %% Test ADF y test KPSS

# ADF  — H0: la serie tiene raíz unitaria. p-valor < 0.05 → la serie es estacionaria.
# KPSS — H0: la serie es estacionaria.     p-valor > 0.05 → la serie es estacionaria.
# Se usan juntas. En el ADF el rezago máximo es el de Schwert (≈ 4 con 100 datos) y se elige por BIC.
filas_raiz = []
for x, etiqueta in [(serie, "100 observaciones")]:
    maxlag = int(np.ceil(4 * (len(x) / 100) ** 0.25))
    adf_result  = adfuller(x, maxlag=maxlag, autolag="BIC", regression="c")
    kpss_result = kpss(x, regression="c", nlags="auto")

    print(f"\n=== Test ADF — muestra {etiqueta} ===")
    print("Estadístico ADF:", round(adf_result[0], 4))
    print("p-valor:", round(adf_result[1], 4))
    print("Rezagos usados:", adf_result[2])
    print("Observaciones:", adf_result[3])
    print("Valores críticos:")
    for nivel, valor in adf_result[4].items():
        print(f"  {nivel}: {valor:.4f}")
    if adf_result[1] < 0.05:
        print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
    else:
        print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

    print(f"\n=== Test KPSS — muestra {etiqueta} ===")
    print("Estadístico KPSS:", round(kpss_result[0], 4))
    print("p-valor:", round(kpss_result[1], 4),
          "(statsmodels reporta 0.10 cuando el p-valor real es mayor)" if kpss_result[1] >= 0.1 else "")
    print("Valores críticos:")
    for nivel, valor in kpss_result[3].items():
        print(f"  {nivel}: {valor:.4f}")
    if kpss_result[1] > 0.05:
        print("KPSS: No rechazamos H0. Según el test, la serie es estacionaria.")
    else:
        print("KPSS: Rechazamos H0. Según el test, la serie no es estacionaria.")

    filas_raiz.append({
        "Muestra": etiqueta, "n": len(x),
        "ADF estadístico": round(adf_result[0], 3), "ADF p-valor": round(adf_result[1], 3),
        "KPSS estadístico": round(kpss_result[0], 3), "KPSS p-valor": round(kpss_result[1], 3),
    })

tabla_raiz = pd.DataFrame(filas_raiz)
print("\n=== Tabla de raíz unitaria ===")
print(tabla_raiz.to_string(index=False))
tabla_como_imagen(tabla_raiz, "Pruebas de raíz unitaria: ADF y KPSS", "02a_tabla_raiz_unitaria.png")

# Decisión: d = 0. El ADF rechaza raíz unitaria y el KPSS no rechaza estacionariedad: ambas
# pruebas coinciden en que la serie es estacionaria (una tasa de crecimiento es I(0)).
d = 0
print(f"\nDecisión: d = {d}")

# %% =========================
# PASO 2: ESTIMACIÓN
# ============================

# Modelos candidatos sugeridos por la FAC y FACP de la serie corregida:
# AR(2)    : picos de la FACP en los rezagos 1 y 2 → AR de orden bajo como base
# AR(4)    : orden vecino inferior
# AR(5)    : la FACP se corta después del rezago 5
# AR(6)    : orden vecino superior, para comparar
# ARMA(2,2): modelo mixto alternativo
modelos_candidatos = {
    "AR(2)"    : (2, d, 0),
    "AR(4)"    : (4, d, 0),
    "AR(5)"    : (5, d, 0),
    "AR(6)"    : (6, d, 0),
    "ARMA(2,2)": (2, d, 2),
}
nombres_modelos = list(modelos_candidatos.keys())

# Diccionario que almacenará las estimaciones de cada modelo
estimaciones = {}

for nombre, orden in modelos_candidatos.items():
    estimacion_modelo = ajustar(serie, orden, exog)
    estimaciones[nombre] = estimacion_modelo

    print("\n", nombre)
    print(estimacion_modelo.summary())
    print(f"Media incondicional implícita c / (1 - suma AR): {media_incondicional(estimacion_modelo):.3f}")

# %% Tabla resumen de modelos estimados

# Coeficientes con su error estándar entre paréntesis (sin las dummies, que son las mismas en
# todos los modelos), media incondicional, AIC y BIC.
parametros_tabla = ["ar.L1", "ar.L2", "ar.L3", "ar.L4", "ar.L5", "ar.L6", "ma.L1", "ma.L2", "intercept"]
etiquetas_tabla  = {"ar.L1": "a1", "ar.L2": "a2", "ar.L3": "a3", "ar.L4": "a4", "ar.L5": "a5",
                    "ar.L6": "a6", "ma.L1": "b1", "ma.L2": "b2", "intercept": "intercepto SARIMAX"}

tabla_publicacion = []
for parametro in parametros_tabla:
    fila_coef = {"Parámetro": etiquetas_tabla[parametro]}
    fila_se   = {"Parámetro": ""}
    for nombre in nombres_modelos:
        res = estimaciones[nombre]
        if parametro in res.params.index:
            fila_coef[nombre] = f"{res.params[parametro]:.3f}"
            fila_se[nombre]   = f"({res.bse[parametro]:.3f})"
        else:
            fila_coef[nombre] = ""
            fila_se[nombre]   = ""
    tabla_publicacion += [fila_coef, fila_se]

fila_mu = {"Parámetro": "media incondicional"}
fila_mu_se = {"Parámetro": ""}
fila_aic = {"Parámetro": "AIC"}
fila_bic = {"Parámetro": "BIC"}
for nombre in nombres_modelos:
    res = estimaciones[nombre]
    fila_mu[nombre]    = f"{media_incondicional(res):.3f}"
    fila_mu_se[nombre] = f"({se_media_incondicional(res):.3f})"
    fila_aic[nombre]   = f"{res.aic:.1f}"
    fila_bic[nombre]   = f"{res.bic:.1f}"
tabla_publicacion += [fila_mu, fila_mu_se, fila_aic, fila_bic]

tabla_publicacion_df = pd.DataFrame(tabla_publicacion)
print("\nTabla resumen de modelos estimados")
print(tabla_publicacion_df.to_string(index=False))
print("\nErrores estándar entre paréntesis. Todos los modelos incluyen las 9 dummies COVID.")
tabla_como_imagen(tabla_publicacion_df, "Tabla resumen de modelos estimados", "02b_tabla_modelos.png")


# %% =========================
# PASO 3: VALIDACIÓN DE SUPUESTOS
# ============================

# Ljung-Box (H0: no autocorrelación) con gl = h − p − q.
# Jarque-Bera (H0: normalidad). ARCH (H0: no hay efectos ARCH).
# p-valor > 0.05 → el supuesto se cumple.
# Un modelo es válido si es estacionario (raíces AR > 1), invertible (raíces MA > 1)
# y no deja autocorrelación en los residuos.
tabla_diagnostico = []
for nombre, res in estimaciones.items():
    p, _, q = res.model.order
    e = residuos_validos(res, exog)
    lb = acorr_ljungbox(e, lags=[8, 12, 16, 20], model_df=p + q, return_df=True)["lb_pvalue"]
    raiz_ar = np.abs(res.arroots).min()
    raiz_ma = np.abs(res.maroots).min() if q > 0 else np.inf
    tabla_diagnostico.append({
        "Modelo": nombre,
        "mín|raíz AR|": raiz_ar, "mín|raíz MA|": raiz_ma,
        "LB(8)": lb[8], "LB(12)": lb[12], "LB(16)": lb[16], "LB(20)": lb[20],
        "JB": jarque_bera(e).pvalue,
        "A(1)": het_arch(e, nlags=1)[1], "A(5)": het_arch(e, nlags=5)[1],
        "válido": raiz_ar > 1.01 and raiz_ma > 1.01 and bool((lb > 0.05).all()),
    })

tabla_diagnostico_df = pd.DataFrame(tabla_diagnostico)
print("\n=== Tabla de diagnóstico (p-valores) ===")
print("JB=Jarque-Bera | A=ARCH | LB=Ljung-Box")
print("p-valor > 0.05 indica que el supuesto se cumple")
print(tabla_diagnostico_df.round(3).to_string(index=False))
tabla_como_imagen(tabla_diagnostico_df.round(3).astype(str), "Tabla de diagnóstico (p-valores)",
                  "03_tabla_diagnostico.png", ancho=13)

# %% Selección del modelo final

# Se elige el de menor BIC entre los modelos válidos.
nombre_final = tabla_diagnostico_df[tabla_diagnostico_df["válido"]].merge(
    pd.DataFrame({"Modelo": nombres_modelos, "BIC": [estimaciones[n].bic for n in nombres_modelos]}),
    on="Modelo").sort_values("BIC")["Modelo"].iloc[0]
estimacion_final = estimaciones[nombre_final]
print(f"\nModelo elegido: {nombre_final}")

# %% Tabla explícita del modelo final

print(f"\n=== Estimación explícita del modelo final {nombre_final} ===")
coefs = pd.DataFrame({
    "coef"            : estimacion_final.params.round(4),
    "error estándar"  : estimacion_final.bse.round(4),
    "z-estadístico"   : estimacion_final.tvalues.round(4),
    "p-valor"         : estimacion_final.pvalues.round(4),
    "significativo 5%": estimacion_final.pvalues < 0.05,
})
print(coefs.to_string())
mu_hat = media_incondicional(estimacion_final)
print(f"μ̂ = a0/(1−Σa_i) = {mu_hat:.3f} %")
print(f"σ̂² = {estimacion_final.params['sigma2']:.4f}")
print("Raíces AR (módulo):", np.round(np.abs(estimacion_final.arroots), 3))
tabla_como_imagen(coefs.reset_index().rename(columns={"index": "Parámetro"}).astype(str),
                  f"Estimación del modelo final {nombre_final}", "03b_tabla_modelo_final.png")

e_final = residuos_validos(estimacion_final, exog)
z_max = np.abs(e_final / e_final.std()).max()
print(f"Mayor residuo estandarizado fuera de la ventana COVID: {z_max:.2f}σ "
      + ("(< 3.5σ → no hay más atípicos)" if z_max < 3.5 else "(> 3.5σ → quedan atípicos fuera de la ventana COVID)"))

# %% Gráficas de diagnóstico del modelo final

fig, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0, 0].plot(e_final, color="black", lw=1)
ax[0, 0].axhline(0, color="red", lw=0.8, ls="--")
ax[0, 0].set_title(f"Residuales — {nombre_final}")
ax[0, 0].grid(alpha=0.3)
plot_acf(e_final, lags=16, alpha=0.05, bartlett_confint=False, ax=ax[0, 1])
ax[0, 1].set_title("FAC residuales (no autocorrelación)")
ax[0, 1].set_ylim(-1, 1)
plot_pacf(e_final ** 2, lags=16, alpha=0.05, method="ywm", ax=ax[1, 0])
ax[1, 0].set_title("FACP residuales² (homocedasticidad)")
ax[1, 0].set_ylim(-1, 1)
probplot(e_final, dist="norm", plot=ax[1, 1])
ax[1, 1].set_title("Q-Q plot residuales (normalidad)")
ax[1, 1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "03c_diagnostico_modelo_final.png", dpi=150)
plt.show()

# %% Estabilidad: OLS-CUSUM

# H0: parámetros estables. Se rechaza si la trayectoria cruza ±1.358 (5 %).
S = np.cumsum(e_final.values) / (e_final.std(ddof=1) * np.sqrt(len(e_final)))
stat_cusum, p_cusum, _ = breaks_cusumolsresid(e_final.values,
                                              ddof=len(estimacion_final.params) - exog.shape[1])
if p_cusum > 0.05:
    print(f"\n  ✓ CUSUM dentro de bandas en toda la muestra (p-valor = {p_cusum:.3f}).")
else:
    print(f"\n  ⚠ CUSUM supera las bandas (p-valor = {p_cusum:.3f}).")

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
plt.savefig(RESULTADOS_DIR / "04_cusum.png", dpi=150)
plt.show()

# %% Tabla de pruebas del modelo final (para el póster)

p_f, _, q_f = estimacion_final.model.order
lb20  = acorr_ljungbox(e_final, lags=[20], model_df=p_f + q_f, return_df=True)
arch5 = het_arch(e_final, nlags=5)
jb    = jarque_bera(e_final)

tabla_pruebas = pd.DataFrame([
    {"Prueba": "Ljung-Box (20)", "Estadístico": lb20.loc[20, "lb_stat"], "p-valor": lb20.loc[20, "lb_pvalue"],
     "Conclusión": "no autocorrelación" if lb20.loc[20, "lb_pvalue"] > 0.05 else "autocorrelación"},
    {"Prueba": "ARCH (5)", "Estadístico": arch5[0], "p-valor": arch5[1],
     "Conclusión": "homocedasticidad" if arch5[1] > 0.05 else "heterocedasticidad"},
    {"Prueba": "Jarque-Bera", "Estadístico": jb.statistic, "p-valor": jb.pvalue,
     "Conclusión": "normalidad" if jb.pvalue > 0.05 else "no normalidad"},
    {"Prueba": "OLS-CUSUM", "Estadístico": stat_cusum, "p-valor": p_cusum,
     "Conclusión": "parámetros estables" if p_cusum > 0.05 else "inestabilidad"},
])
tabla_pruebas[["Estadístico", "p-valor"]] = tabla_pruebas[["Estadístico", "p-valor"]].round(3)
print(f"\n=== Pruebas sobre los residuos del {nombre_final} ===")
print(tabla_pruebas.to_string(index=False))
tabla_como_imagen(tabla_pruebas, f"Validación del modelo {nombre_final}", "03d_tabla_pruebas_modelo_final.png")

# %% Quiebres en la media: test de Bai-Perron

# Bai & Perron (1998, 2003) detectan m quiebres desconocidos en la media. ruptures aplica su
# programación dinámica y el número de quiebres se elige por BIC: BIC = n·ln(RSS/n) + k·ln(n).
print("\n" + "=" * 60)
print("TEST DE BAI-PERRON [ruptures]")
print("=" * 60)

y_bp = serie.values
n_bp = len(y_bp)
M_max = 5           # máximo de quiebres a considerar
modelo_bp = rpt.Dynp(model="l2", jump=1).fit(y_bp)
bic_bp = {}
for m in range(0, M_max + 1):
    bkps = modelo_bp.predict(n_bkps=m) if m > 0 else [n_bp]
    rss = sum(np.sum((y_bp[i:j] - y_bp[i:j].mean()) ** 2) for i, j in zip([0] + bkps[:-1], bkps))
    bic_bp[m] = n_bp * np.log(rss / n_bp) + (m + 1) * np.log(n_bp)
    print(f"  m={m} quiebres  →  RSS={rss:.4f}  BIC={bic_bp[m]:.3f}")

m_opt = min(bic_bp, key=bic_bp.get)
bk = modelo_bp.predict(n_bkps=m_opt) if m_opt > 0 else [n_bp]
fechas_quiebre = [serie.index[b - 1].date() for b in bk[:-1]]
print(f"\n  ✓ Número óptimo de quiebres (BIC): m = {m_opt}")
print(f"  Fechas de quiebre estimadas: {[str(f) for f in fechas_quiebre]}")

segmentos = list(zip([0] + bk[:-1], bk))
tabla_bp = pd.DataFrame([{
    "Régimen": k + 1,
    "Inicio": str(serie.index[i].date()),
    "Fin": str(serie.index[j - 1].date()),
    "Obs.": j - i,
    "Media (%)": round(y_bp[i:j].mean(), 4),
    "Desv. Est. (%)": round(y_bp[i:j].std(), 4),
} for k, (i, j) in enumerate(segmentos)])
print("\n" + tabla_bp.to_string(index=False))
tabla_como_imagen(tabla_bp, f"Bai-Perron — regímenes estimados (m = {m_opt})", "04c_tabla_bai_perron.png")

plt.figure(figsize=(11, 5))
plt.plot(serie, color="black", lw=1.2, label="Serie original")
for k, (i, j) in enumerate(segmentos):
    plt.hlines(y_bp[i:j].mean(), serie.index[i], serie.index[j - 1], colors=f"C{k}", lw=2,
               label=f"Media régimen {k + 1}: {y_bp[i:j].mean():.2f}%")
for fecha in fechas_quiebre:
    plt.axvline(pd.Timestamp(fecha), color="red", lw=1.2, ls="--", alpha=0.8)
plt.axhline(0, color="black", lw=0.6, ls=":")
plt.title(f"Test de Bai-Perron — Reino Unido (1998–2023)\nQuiebres óptimos (BIC): m = {m_opt}")
plt.xlabel("Fecha")
plt.ylabel("Tasa de crecimiento (%)")
plt.legend(fontsize=8, loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "04b_bai_perron.png", dpi=150)
plt.show()


# %% =========================
# PASO 4: PRONÓSTICO
# ============================

# Pronóstico óptimo: E_t y_{t+j}. Error: e_t(j) = Σ_{i<j} ψ_i ε_{t+j−i}, con ψ_0 = 1.
# Var[e_t(j)] = σ² (ψ_0² + … + ψ_{j−1}²);  IC 95 % = pronóstico ± 1.96·sqrt(Var).
# Fuera de muestra las dummies valen 0 (no se espera otro COVID).
pasos  = 10
n_boot = 5000

idx_pron = pd.date_range(serie.index[-1], periods=pasos + 1, freq=FREQ)[1:]
exog_fut = pd.DataFrame(0.0, index=idx_pron, columns=exog.columns)

pronostico = estimacion_final.get_forecast(pasos, exog=exog_fut)
puntual    = pronostico.predicted_mean.values

# Pesos ψ y varianza del error de pronóstico (fórmulas del curso)
phi   = np.array([estimacion_final.params[f"ar.L{i}"] for i in range(1, p_f + 1)])
theta = np.array([estimacion_final.params[f"ma.L{j}"] for j in range(1, q_f + 1)])
psi   = arma2ma(np.r_[1.0, -phi], np.r_[1.0, theta], lags=pasos)
var_e = estimacion_final.params["sigma2"] * np.cumsum(psi ** 2)

# %% Intervalos por bootstrapping de residuales

# Se remuestrean los residuos (centrados) y se propagan por la ecuación ARMA, así el intervalo
# crece con el horizonte. No asume normalidad.
rng    = np.random.default_rng(42)
par    = estimacion_final.params
c      = par["intercept"]
beta   = par[list(exog.columns)].values
u_hist = estimacion_final.model.endog.ravel() - estimacion_final.model.exog @ beta   # serie sin las dummies
e_hist = estimacion_final.resid.values
pool   = e_final.values - e_final.values.mean()


def trayectoria(choques):
    u = list(u_hist[-p_f:])
    e = list(e_hist[-q_f:]) if q_f else []
    salida = np.empty(pasos)
    for h in range(pasos):
        u_nuevo = c + np.dot(phi, u[::-1][:p_f]) + (np.dot(theta, e[::-1][:q_f]) if q_f else 0.0) + choques[h]
        u.append(u_nuevo)
        e.append(choques[h])
        salida[h] = u_nuevo
    return salida


boot_matrix = np.vstack([trayectoria(rng.choice(pool, size=pasos, replace=True)) for _ in range(n_boot)])

tabla_pronostico = pd.DataFrame({
    "pronostico_puntual"  : puntual,
    "ψ_{j-1}"             : psi,
    "Var[e_t(j)]"         : var_e,
    "IC_inf_95_analitico" : puntual - 1.96 * np.sqrt(var_e),
    "IC_sup_95_analitico" : puntual + 1.96 * np.sqrt(var_e),
    "IC_inf_95_bootstrap" : np.percentile(boot_matrix, 2.5, axis=0),
    "IC_sup_95_bootstrap" : np.percentile(boot_matrix, 97.5, axis=0),
}, index=idx_pron).round(4)

print(f"\n=== Pronóstico 10 trimestres adelante — {nombre_final} ===")
print(tabla_pronostico.to_string())
tabla_como_imagen(tabla_pronostico.round(3).reset_index(names="Trimestre").astype(str),
                  f"Pronóstico 10 trimestres adelante — {nombre_final}", "05b_tabla_pronostico.png", ancho=13)

# %% Gráfica del pronóstico

plt.figure(figsize=(11, 5))
plt.plot(serie.iloc[-40:], color="black", lw=1.2, label="Datos históricos")
plt.plot(idx_pron, puntual, color="orange", lw=2, ls="--", marker="o", ms=4, label=f"Pronóstico {nombre_final}")
plt.fill_between(idx_pron, tabla_pronostico["IC_inf_95_analitico"], tabla_pronostico["IC_sup_95_analitico"],
                 color="orange", alpha=0.25, label="IC 95 % analítico (ψ-pesos)")
plt.plot(idx_pron, tabla_pronostico["IC_inf_95_bootstrap"], color="firebrick", lw=1, ls=":", label="IC 95 % bootstrap")
plt.plot(idx_pron, tabla_pronostico["IC_sup_95_bootstrap"], color="firebrick", lw=1, ls=":")
plt.axhline(mu_hat, color="gray", lw=0.8, ls="--", label=f"μ̂ = {mu_hat:.2f} %")
plt.title(f"Pronóstico — PIB Reino Unido, {pasos} trimestres desde 2023 Q3")
plt.xlabel("Fecha")
plt.ylabel("Tasa de crecimiento (%)")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTADOS_DIR / "05_pronostico.png", dpi=150)
plt.show()

print("\n=== Interpretación del pronóstico ===")
print(
    f"El modelo {nombre_final} proyecta que la tasa de crecimiento anual del PIB del Reino Unido "
    f"se situará en promedio en {puntual.mean():.2f}% durante los 10 trimestres siguientes a 2023 Q3 "
    f"y converge hacia su media incondicional de {mu_hat:.2f}%. Los intervalos al 95% se construyeron "
    f"de forma analítica (ψ-pesos) y por bootstrapping de residuales, que no requiere asumir normalidad."
)
