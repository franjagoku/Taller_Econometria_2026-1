#taller_econometria_2_poster
# %% Importación de paquetes ============================

# Trabajar con rutas relativas en python 
from pathlib import Path

# Módulos de numpy, pandas, matplotlib y scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import jarque_bera
import statsmodels.stats.stattools as smt

#Iteradores para bucles eficientes
import itertools, warnings

# %% Cargar bases de datos en python usando rutas relativas =========================

# Obtener la ruta del directorio raíz
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR

# Rutas de las bases de datos 
ruta_exp = DATA_DIR / "LORSGPORUSQ659S.csv" # Base de datos de exportaciones

# %% =========================
# PRIMERA SERIE: GDP US
# ============================

# Base de datos con la serie importada a python
expo_base = pd.read_csv(
    ruta_exp,
)

#ver el tipo de objeto de la base de datos (Pandaas.DataFrame)
print(type(expo_base))

#ver las primeras y últimas observaciones de la base de datos
print(expo_base.head()) #primeras 5 observaciones
print(expo_base.tail()) #segundas 5 observaciones

# %% Creación de la serie de tiempo de "GDP US"

# La nueva serie de tiempo se va a llamar "expo_serie" y va a tener valores numéricos
expo_serie = expo_base["LORSGPORUSQ659S"].copy()
expo_serie = pd.to_numeric(expo_serie, errors="coerce")
expo_serie = expo_serie.dropna() # Borrar missing values

# Ver el principio y final de la serie de tiempo
print(expo_serie.head())
print(expo_serie.tail())

# # El tipo de objeto de la base de datos ahora es un Pandas.Series
print(type(expo_serie))

# Ver algunas estadísticas descriptivas de la serie de tiempo 
print(expo_serie.describe())

# %% =========================
# Paso 1: Identificación
# ============================

# Gráfica de la serie de tiempo "GDP US"
plt.figure(figsize=(10, 5))
plt.plot(expo_serie)
plt.title("GDP US, 1948-2023")
plt.xlabel("Tiempo")
plt.ylabel("GDP US (% cambio trimestral)")
plt.grid(True)
plt.show()

# %% FAC y FACP de la serie original

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    expo_serie,
    lags=50,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC GDP US")

plot_pacf(
    expo_serie,
    lags=50,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de la serie original")

plt.tight_layout()
plt.show()

# %% Tests de Raíz Unitaria 

# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(expo_serie, regression="c", autolag="BIC") # Test de Dickey Fuller con intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2]) 
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result = kpss(expo_serie, regression="c", nlags="auto") #test KPSS con intercepto y tendencia determinística

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result[0])
print("p-valor:", kpss_result[1])
print("Rezagos usados:", kpss_result[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"{nivel}: {valor}")

if kpss_result[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

# Nota: Según los resultados de la prueba ADF y KPSS con tendencia determinística,
# la serie de tiempo del crecimiento del GNB US es estacionaria.

# %% Prebas de modelos SARIMA (p,d,q)(P,D,Q)s
warnings.filterwarnings("ignore")

s = 4                      # periodo estacional cada 4 rezagos (trimestral)
d, D = 0, 0                 # según las pruebas ADF/KPSS
p_rng = q_rng = range(0, 3)
P_rng = Q_rng = range(0, 2)

resultados, modelos = [], {}
for p, q, P, Q in itertools.product(p_rng, q_rng, P_rng, Q_rng):
    orden, est = (p, d, q), (P, D, Q, s)
    try:
        m = SARIMAX(expo_serie, order=orden, seasonal_order=est,
                    enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False)
        modelos[(orden, est)] = m
        resultados.append({"orden": orden, "estacional": est,
                           "AIC": m.aic, "BIC": m.bic})
    except Exception as e:
        print(f"Falló {orden}x{est}: {e}")

tabla = pd.DataFrame(resultados).sort_values("AIC").reset_index(drop=True)
print(tabla.head(10))

# %% Selección del mejor modelo SARIMA
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
for _, r in tabla.head(5).iterrows():      # cambia 5 por los que quieras
    clave = (r["orden"], r["estacional"])
    diag = diagnosticos(modelos[clave], *clave)
    filas.append({**r.to_dict(), **diag})

comparacion = pd.DataFrame(filas)

cols_p = [c for c in comparacion.columns if "_p" in c]

formato = {c: "{:.4f}".format for c in cols_p}
formato.update({"AIC": "{:.2f}".format, "BIC": "{:.2f}".format,
                "Asimetría": "{:.3f}".format, "Curtosis": "{:.3f}".format})

comparacion.style.format(formato)

comparacion.drop(columns=["Asimetría", "Curtosis"]).style.format(formato)
# %%
