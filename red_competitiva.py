"""Proyecto 2 - Redes Neuronales Estáticas
Persona 2: Modelo no supervisado (Red Competitiva).
Ejecutar desde la carpeta del proyecto (donde están datos/ e img/)."""

# ----------------------------------------------------------------------

try:
    from IPython.display import display
except ImportError:
    display = print


# ----------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

carpeta_tratados = Path("datos/tratados")
carpeta_img = Path("img")
carpeta_modelos = Path("modelos")
carpeta_modelos.mkdir(parents=True, exist_ok=True)

semilla = 42
col_objetivo = "quiebra"

# Datos tratados por la Persona 1 (winsorización + Z-score)
entrenamiento = pd.read_csv(carpeta_tratados / "entrenamiento_normalizado.csv", index_col="id_empresa")
prueba = pd.read_csv(carpeta_tratados / "prueba_normalizado.csv", index_col="id_empresa")
ranking = pd.read_csv(carpeta_tratados / "ranking_variables.csv", index_col="variable")

print(f"Entrenamiento: {entrenamiento.shape[0]} empresas | Prueba: {prueba.shape[0]} empresas")

# ----------------------------------------------------------------------

# Selección: mayor separación entre clases y correlación < 0.8 con las ya elegidas
umbral_corr = 0.8
n_variables = 10
corr = entrenamiento.drop(columns=col_objetivo).corr().abs()

variables_rc = []
for v in ranking.index:
    if all(corr.loc[v, s] < umbral_corr for s in variables_rc):
        variables_rc.append(v)
    if len(variables_rc) == n_variables:
        break

nombres_es = {
    "Net Income to Stockholder's Equity": "Utilidad neta / capital contable (ROE)",
    "Borrowing dependency": "Dependencia de préstamos",
    "Retained Earnings to Total Assets": "Utilidades retenidas / activo total",
    "Current Liabilities/Equity": "Pasivo circulante / capital",
    "Current Liability to Equity": "Pasivo circulante / capital",   # Columna idéntica en la base
    "Continuous interest rate (after tax)": "Tasa de interés continua (desp. de imp.)",
    "Current Liability to Current Assets": "Pasivo circulante / activo circulante",
    "Net profit before tax/Paid-in capital": "Utilidad antes de imp. / capital pagado",
    "Non-industry income and expenditure/revenue": "Ingresos y gastos no operativos / ingresos",
    "Total income/Total expense": "Ingreso total / gasto total",
    "Working Capital/Equity": "Capital de trabajo / capital",
}
familia = {
    "Net Income to Stockholder's Equity": "Rentabilidad",
    "Borrowing dependency": "Endeudamiento",
    "Retained Earnings to Total Assets": "Rentabilidad",
    "Current Liabilities/Equity": "Endeudamiento",
    "Current Liability to Equity": "Endeudamiento",
    "Continuous interest rate (after tax)": "Rentabilidad",
    "Current Liability to Current Assets": "Liquidez",
    "Net profit before tax/Paid-in capital": "Rentabilidad",
    "Non-industry income and expenditure/revenue": "Rentabilidad",
    "Total income/Total expense": "Rentabilidad",
    "Working Capital/Equity": "Liquidez",
}

tabla_variables = pd.DataFrame({
    "Variable (base original)": variables_rc,
    "Nombre en español": [nombres_es.get(v, v) for v in variables_rc],
    "Familia": [familia.get(v, "") for v in variables_rc],
    "Diferencia de medias z (quiebra − sana)": ranking.loc[variables_rc, "diferencia_medias_z"].round(3).values,
}, index=range(1, len(variables_rc) + 1))
print(f"Correlación máxima entre variables elegidas: "
      f"{corr.loc[variables_rc, variables_rc].where(~np.eye(len(variables_rc), dtype=bool)).max().max():.3f}")
print(tabla_variables)

# ----------------------------------------------------------------------

# Matrices de la forma usada en clase: (variables, empresas)
X_ent = entrenamiento[variables_rc].to_numpy().T
y_ent = entrenamiento[col_objetivo].to_numpy()
X_pru = prueba[variables_rc].to_numpy().T
y_pru = prueba[col_objetivo].to_numpy()
print(f"X entrenamiento: {X_ent.shape} | X prueba: {X_pru.shape}")

# ----------------------------------------------------------------------

def entrenar_red_competitiva(data, nn, alpha=0.05, epocas=30, semilla=42, init="muestras"):
    """Red competitiva (algoritmo de clase, RedCompetitiva_1.py).
    data: matriz (variables, empresas). Regresa W de tamaño (variables, nn)."""
    rng = np.random.default_rng(semilla)

    # Inicialización de pesos
    if init == "muestras":   # nn empresas elegidas al azar
        W = data[:, rng.choice(data.shape[1], nn, replace=False)].copy()
    else:                    # Igual que en clase: valores aleatorios entre 0 y 1
        W = rng.random((data.shape[0], nn))

    for nepoc in range(epocas):
        alpha_t = alpha * (1 - nepoc / epocas)          # Tasa de aprendizaje decreciente
        for k in rng.permutation(data.shape[1]):        # Empresas en orden aleatorio
            # Competencia: distancia euclidiana cuadrada entre la empresa k y todos los pesos
            muestra_actual = data[:, k].reshape(-1, 1)
            distancias = np.sum((muestra_actual - W) ** 2, axis=0)
            ind = np.argmin(distancias)                 # Neurona ganadora

            # Aprendizaje: solo se actualizan los pesos de la neurona ganadora
            W[:, ind] = W[:, ind] + alpha_t * (data[:, k] - W[:, ind])
    return W


def neurona_ganadora(data, W):
    """Índice de la neurona ganadora para cada empresa (columna) de data."""
    distancias = np.sum((data[:, :, None] - W[:, None, :]) ** 2, axis=0)   # (empresas, nn)
    return np.argmin(distancias, axis=1)


def error_cuantizacion(data, W):
    """Distancia cuadrada promedio de cada empresa a su neurona ganadora."""
    g = neurona_ganadora(data, W)
    return np.mean(np.sum((data - W[:, g]) ** 2, axis=0))

# ----------------------------------------------------------------------

nn_max = 12
conteo_ganadoras = {}
for init, nombre in [("rand", "Pesos iniciales entre 0 y 1 (como en clase)"),
                     ("muestras", "Pesos iniciales = empresas al azar")]:
    W_tmp = entrenar_red_competitiva(X_ent, nn_max, init=init, semilla=semilla)
    conteo_ganadoras[nombre] = np.bincount(neurona_ganadora(X_ent, W_tmp), minlength=nn_max)

tabla_conteo = pd.DataFrame(conteo_ganadoras, index=[f"N{i + 1}" for i in range(nn_max)]).T
tabla_conteo["Neuronas activas"] = (tabla_conteo > 0).sum(axis=1)
tabla_conteo["Neuronas con ≥ 5 % de las empresas"] = (tabla_conteo.iloc[:, :nn_max] >= 0.05 * X_ent.shape[1]).sum(axis=1)
print(tabla_conteo)

# ----------------------------------------------------------------------

from sklearn.metrics import adjusted_rand_score

rango_nn = range(2, 11)
semillas = range(5)
filas = []
for nn in rango_nn:
    grupos_semillas, errores = [], []
    for s in semillas:
        W_tmp = entrenar_red_competitiva(X_ent, nn, semilla=s)
        grupos_semillas.append(neurona_ganadora(X_ent, W_tmp))
        errores.append(error_cuantizacion(X_ent, W_tmp))
    ari = np.mean([adjusted_rand_score(grupos_semillas[0], g) for g in grupos_semillas[1:]])
    # Concentración de quiebras: % de las quiebras que caen en el grupo con mayor tasa de quiebra
    g0 = grupos_semillas[0]
    tasas = np.bincount(g0, weights=y_ent, minlength=nn) / np.maximum(np.bincount(g0, minlength=nn), 1)
    captura = y_ent[g0 == np.argmax(tasas)].sum() / y_ent.sum()
    filas.append({"Neuronas": nn, "Error de cuantización": np.mean(errores),
                  "Estabilidad (ARI)": ari, "Tasa máx. de quiebra en un grupo": tasas.max(),
                  "% de quiebras en ese grupo": captura})

tabla_nn = pd.DataFrame(filas).set_index("Neuronas")

fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
axs[0].plot(tabla_nn.index, tabla_nn["Error de cuantización"], "o-")
axs[0].set_title("Codo de Jambú: error de cuantización")
axs[0].set_xlabel("Número de neuronas $n_n$")
axs[0].set_ylabel("Distancia cuadrada promedio")
axs[1].plot(tabla_nn.index, tabla_nn["Estabilidad (ARI)"], "s-", color="tab:green")
axs[1].set_title("Estabilidad de los grupos entre 5 semillas")
axs[1].set_xlabel("Número de neuronas $n_n$")
axs[1].set_ylabel("ARI promedio")
axs[1].set_ylim(0, 1.05)
for ax in axs:
    ax.axvline(3, color="tab:red", ls="--", alpha=0.6)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(carpeta_img / "rc_numero_neuronas.png", dpi=150)
plt.show()

print(tabla_nn.round(4))

# ----------------------------------------------------------------------

nn = 3
alpha = 0.05
epocas = 30

W = entrenar_red_competitiva(X_ent, nn, alpha=alpha, epocas=epocas, semilla=semilla)
grupo_ent = neurona_ganadora(X_ent, W)
grupo_pru = neurona_ganadora(X_pru, W)

print(f"Neuronas: {nn} | α₀ = {alpha} | Épocas: {epocas}")
print(f"Error de cuantización: entrenamiento = {error_cuantizacion(X_ent, W):.3f} | "
      f"prueba = {error_cuantizacion(X_pru, W):.3f}")

# ----------------------------------------------------------------------

def tabla_grupos(grupos, y):
    t = pd.crosstab(grupos, y).reindex(index=range(nn), columns=[0, 1], fill_value=0)
    t.columns = ["Sanas", "En quiebra"]
    t["Total"] = t.sum(axis=1)
    t["Tasa de quiebra"] = t["En quiebra"] / t["Total"]
    t["% de todas las quiebras"] = t["En quiebra"] / t["En quiebra"].sum()
    return t

dist_ent = tabla_grupos(grupo_ent, y_ent)

# Nombrar los grupos de menor a mayor tasa de quiebra (zona segura, gris y de peligro)
orden = dist_ent["Tasa de quiebra"].sort_values().index
zonas = ["Zona segura", "Zona gris", "Zona de peligro"]
nombre_grupo = {neurona: f"N{neurona + 1} · {zona}" for neurona, zona in zip(orden, zonas)}

dist_ent = dist_ent.loc[orden]
dist_pru = tabla_grupos(grupo_pru, y_pru).loc[orden]
for t in (dist_ent, dist_pru):
    t.index = [nombre_grupo[i] for i in t.index]

formato = {"Tasa de quiebra": "{:.2%}", "% de todas las quiebras": "{:.1%}"}
print(f"Tasa general de quiebra en entrenamiento: {y_ent.mean():.2%}")
print("\nEntrenamiento:"); print(dist_ent.round(4))
print("\nPrueba:"); print(dist_pru.round(4))

# ----------------------------------------------------------------------

# Gráfica: composición de cada grupo y reparto de las quiebras
etiquetas = list(dist_ent.index)
fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))

axs[0].bar(etiquetas, dist_ent["Sanas"], color="tab:blue", label="Sanas")
axs[0].bar(etiquetas, dist_ent["En quiebra"], bottom=dist_ent["Sanas"], color="tab:red", label="En quiebra")
for i, (tot, tasa) in enumerate(zip(dist_ent["Total"], dist_ent["Tasa de quiebra"])):
    axs[0].text(i, tot, f"{tot} empresas\n{tasa:.1%} en quiebra", ha="center", va="bottom")
axs[0].set_ylim(0, dist_ent["Total"].max() * 1.2)
axs[0].set_ylabel("Número de empresas")
axs[0].set_title("Empresas que ganó cada neurona (entrenamiento)")
axs[0].legend()

axs[1].bar(etiquetas, dist_ent["% de todas las quiebras"] * 100, color="tab:red")
for i, v in enumerate(dist_ent["% de todas las quiebras"] * 100):
    axs[1].text(i, v, f"{v:.1f} %", ha="center", va="bottom")
axs[1].set_ylim(0, 105)
axs[1].set_ylabel("% de las empresas en quiebra")
axs[1].set_title("¿En qué grupo cayeron las empresas en quiebra?")
for ax in axs:
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(carpeta_img / "rc_distribucion_etiquetas.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------

# Perfil financiero de cada grupo: pesos finales (centro de cada neurona) en unidades z
perfil = pd.DataFrame(W[:, orden].T, index=[nombre_grupo[i] for i in orden],
                      columns=[nombres_es.get(v, v) for v in variables_rc])

fig, ax = plt.subplots(figsize=(13, 3.6))
im = ax.imshow(perfil.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
ax.set_xticks(range(perfil.shape[1]), perfil.columns, rotation=40, ha="right")
ax.set_yticks(range(perfil.shape[0]), perfil.index)
for i in range(perfil.shape[0]):
    for j in range(perfil.shape[1]):
        ax.text(j, i, f"{perfil.iat[i, j]:.2f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, label="Peso (z-score)")
ax.set_title("Pesos finales $\\mathbf{w}_i$ de cada neurona (perfil del grupo)")
plt.tight_layout()
plt.savefig(carpeta_img / "rc_perfil_grupos.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------

# Vista en 2 dimensiones (componentes principales) de los grupos formados
Xc = X_ent - X_ent.mean(axis=1, keepdims=True)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
P = U[:, :2].T @ Xc                       # Proyección de las empresas
PW = U[:, :2].T @ (W - X_ent.mean(axis=1, keepdims=True))   # Proyección de los pesos
var_exp = S[:2] ** 2 / np.sum(S ** 2)

colores = {orden[0]: "tab:green", orden[1]: "tab:orange", orden[2]: "tab:purple"}
plt.figure(figsize=(9, 6.5))
for g in orden:
    m = grupo_ent == g
    plt.scatter(P[0, m], P[1, m], s=8, alpha=0.35, color=colores[g], label=nombre_grupo[g])
q = y_ent == 1
plt.scatter(P[0, q], P[1, q], s=22, facecolors="none", edgecolors="red", linewidths=0.9, label="Empresa en quiebra (real)")
plt.scatter(PW[0], PW[1], marker="*", s=400, color="black", label="Pesos finales (neuronas)")
for g in orden:
    plt.text(PW[0, g] + 0.25, PW[1, g] + 0.25, f"N{g + 1}", fontsize=14, fontweight="bold")
plt.xlabel(f"Componente 1 ({var_exp[0]:.0%} de la varianza)")
plt.ylabel(f"Componente 2 ({var_exp[1]:.0%} de la varianza)")
plt.title("Agrupación final - Red Competitiva (entrenamiento)")
plt.legend(fontsize=9, loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(carpeta_img / "rc_grupos_pca.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------

def matriz_confusion(y_real, y_pred):
    """[[VN, FP], [FN, VP]] contando directamente (clase positiva = quiebra)."""
    vn = np.sum((y_real == 0) & (y_pred == 0)); fp = np.sum((y_real == 0) & (y_pred == 1))
    fn = np.sum((y_real == 1) & (y_pred == 0)); vp = np.sum((y_real == 1) & (y_pred == 1))
    return np.array([[vn, fp], [fn, vp]])

tasa_general = y_ent.mean()
tasa_grupo = np.bincount(grupo_ent, weights=y_ent, minlength=nn) / np.bincount(grupo_ent, minlength=nn)
neurona_riesgo = (tasa_grupo > tasa_general).astype(int)       # 1 = grupo de riesgo
print("Neuronas etiquetadas como riesgo de quiebra:",
      [nombre_grupo[i] for i in range(nn) if neurona_riesgo[i] == 1])

def medidas(y_real, y_pred):
    vn, fp, fn, vp = matriz_confusion(y_real, y_pred).ravel()
    precision = vp / (vp + fp) if vp + fp > 0 else 0.0
    recall = vp / (vp + fn)
    return {"Exactitud": (vp + vn) / len(y_real), "Precisión": precision, "Recall": recall,
            "Especificidad": vn / (vn + fp),
            "F1": 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0}

pred_ent = neurona_riesgo[grupo_ent]
pred_pru = neurona_riesgo[grupo_pru]

desempeno_rc = pd.DataFrame({
    "Red competitiva (entrenamiento)": medidas(y_ent, pred_ent),
    "Red competitiva (prueba)": medidas(y_pru, pred_pru),
    "Referencia: 'todas sanas' (prueba)": medidas(y_pru, np.zeros_like(y_pru)),
}).T
print(desempeno_rc.round(4))

# ----------------------------------------------------------------------

# Matriz de confusión en prueba
mc = matriz_confusion(y_pru, pred_pru)
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.imshow(mc, cmap="Blues")
textos = [["VN", "FP"], ["FN", "VP"]]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{textos[i][j]}\n{mc[i, j]}", ha="center", va="center", fontsize=14,
                color="white" if mc[i, j] > mc.max() / 2 else "black")
ax.set_xticks([0, 1], ["Sana (0)", "Riesgo (1)"])
ax.set_yticks([0, 1], ["Sana (0)", "Quiebra (1)"])
ax.set_xlabel("Predicción de la red")
ax.set_ylabel("Etiqueta real")
ax.set_title("Matriz de confusión - Red competitiva (prueba)")
plt.tight_layout()
plt.savefig(carpeta_img / "rc_matriz_confusion.png", dpi=150)
plt.show()

# ----------------------------------------------------------------------

filas = []
for n in [2, 3, 4, 5, 6, 8, 10]:
    W_tmp = entrenar_red_competitiva(X_ent, n, alpha=alpha, epocas=epocas, semilla=semilla)
    g_ent = neurona_ganadora(X_ent, W_tmp)
    tasas = np.bincount(g_ent, weights=y_ent, minlength=n) / np.maximum(np.bincount(g_ent, minlength=n), 1)
    riesgo = (tasas > tasa_general).astype(int)
    filas.append({"Neuronas": n, **medidas(y_pru, riesgo[neurona_ganadora(X_pru, W_tmp)])})

comparacion_nn = pd.DataFrame(filas).set_index("Neuronas")
print(comparacion_nn.round(4))

# ----------------------------------------------------------------------

pesos_finales = pd.DataFrame(W, index=variables_rc, columns=[f"N{i + 1}" for i in range(nn)])
pesos_finales.index.name = "variable"
pesos_finales.to_csv(carpeta_modelos / "red_competitiva_pesos.csv")

resumen_neuronas = pd.DataFrame({
    "neurona": [f"N{i + 1}" for i in range(nn)],
    "zona": [nombre_grupo[i].split(" · ")[1] for i in range(nn)],
    "tasa_quiebra_entrenamiento": tasa_grupo,
    "riesgo_quiebra": neurona_riesgo,
})
resumen_neuronas.to_csv(carpeta_modelos / "red_competitiva_neuronas.csv", index=False)

print("Pesos finales estabilizados (Wf):")
display(pesos_finales.round(3))
display(resumen_neuronas)


def clasificar_red_competitiva(df_normalizado):
    """Clasifica empresas ya normalizadas (mismo tratamiento de la Persona 1).
    Regresa la neurona ganadora, su zona, el vector one-hot y la predicción de riesgo."""
    x = df_normalizado[variables_rc].to_numpy().T
    distancias = np.sum((x[:, :, None] - W[:, None, :]) ** 2, axis=0)
    g = np.argmin(distancias, axis=1)
    salida = np.eye(nn, dtype=int)[g]                 # One-hot (como en clase)
    return pd.DataFrame({"neurona_ganadora": [f"N{i + 1}" for i in g],
                         "zona": [nombre_grupo[i].split(" · ")[1] for i in g],
                         "salida_one_hot": [list(s) for s in salida],
                         "riesgo_quiebra_rc": neurona_riesgo[g]},
                        index=df_normalizado.index)


# Ejemplo de uso con las 5 empresas reservadas (la Persona 4 hace el análisis completo)
nuevas = pd.read_csv(carpeta_tratados / "empresas_nuevas_normalizado.csv", index_col="id_empresa")
print(clasificar_red_competitiva(nuevas))
