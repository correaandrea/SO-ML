"""
entrenar_modelo.py
Entrena un árbol de decisión con los datos recolectados y guarda el modelo.
Proyecto Final - Sistemas Operativos - Universidad de Antioquia

Uso:
    python3 entrenar_modelo.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # sin interfaz gráfica, guarda a archivo
import matplotlib.pyplot as plt


# ─── Configuración ────────────────────────────────────────────────────────────

DATASET       = os.path.expanduser("~/so-proyecto/datos/dataset.csv")
MODELO_OUT    = os.path.expanduser("~/so-proyecto/modelo/arbol.pkl")
FIGURA_ARBOL  = os.path.expanduser("~/so-proyecto/modelo/arbol_visual.png")
FIGURA_MATRIZ = os.path.expanduser("~/so-proyecto/modelo/matriz_confusion.png")

FEATURES  = ["cpu_percent", "memory_percent", "num_processes"]
TARGET    = "etiqueta"
MAX_DEPTH = 3
ORDEN_CLASES = ["ligera", "moderada", "pesada"]


# ─── Funciones ────────────────────────────────────────────────────────────────

def cargar_datos():
    print("  Cargando dataset...")
    df = pd.read_csv(DATASET)
    print(f"  Total de muestras : {len(df)}")
    print(f"  Distribución de clases:")
    for clase, n in df[TARGET].value_counts().items():
        print(f"    {clase:<10}: {n}")
    return df


def preparar_datos(df):
    X = df[FEATURES].values
    y = df[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Entrenamiento : {len(X_train)} muestras")
    print(f"  Prueba        : {len(X_test)} muestras")
    return X_train, X_test, y_train, y_test


def entrenar(X_train, y_train):
    print(f"\n  Entrenando árbol de decisión (max_depth={MAX_DEPTH})...")
    modelo = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        random_state=42,
        class_weight="balanced"   # compensa leves desbalances
    )
    modelo.fit(X_train, y_train)
    return modelo


def evaluar(modelo, X_train, X_test, y_train, y_test):
    print("\n  ── Evaluación ───────────────────────────────")

    # Exactitud en entrenamiento y prueba
    acc_train = modelo.score(X_train, y_train)
    acc_test  = modelo.score(X_test,  y_test)
    print(f"  Exactitud entrenamiento : {acc_train*100:.1f}%")
    print(f"  Exactitud prueba        : {acc_test*100:.1f}%")

    # Validación cruzada (5 folds) sobre todo el conjunto
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    cv_scores = cross_val_score(modelo, X_all, y_all, cv=5, scoring="accuracy")
    print(f"  Validación cruzada (5-fold):")
    print(f"    Media    : {cv_scores.mean()*100:.1f}%")
    print(f"    Desv.est.: {cv_scores.std()*100:.1f}%")
    print(f"    Por fold : {[f'{s*100:.1f}%' for s in cv_scores]}")

    # Reporte por clase
    y_pred = modelo.predict(X_test)
    print(f"\n  ── Reporte por clase ────────────────────────")
    print(classification_report(y_test, y_pred,
                                 target_names=ORDEN_CLASES,
                                 zero_division=0))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=ORDEN_CLASES)
    print(f"  ── Matriz de confusión ──────────────────────")
    print(f"  {'':>10}  {'ligera':>8}  {'moderada':>8}  {'pesada':>8}")
    for i, fila in enumerate(cm):
        print(f"  {ORDEN_CLASES[i]:>10}  {fila[0]:>8}  {fila[1]:>8}  {fila[2]:>8}")

    return cm, y_pred


def guardar_modelo(modelo):
    os.makedirs(os.path.dirname(MODELO_OUT), exist_ok=True)
    with open(MODELO_OUT, "wb") as f:
        pickle.dump(modelo, f)
    print(f"\n  ✓ Modelo guardado en: {MODELO_OUT}")


def visualizar_arbol(modelo):
    print(f"\n  Generando visualización del árbol...")

    # Versión texto
    reglas = export_text(modelo, feature_names=FEATURES)
    print("\n  ── Reglas del árbol ─────────────────────────")
    print(reglas)

    # Versión gráfica
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        modelo,
        feature_names=FEATURES,
        class_names=modelo.classes_,
        filled=True,
        rounded=True,
        fontsize=11,
        ax=ax
    )
    ax.set_title("Árbol de Decisión — Clasificador de Carga del Sistema", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURA_ARBOL, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Imagen del árbol guardada en: {FIGURA_ARBOL}")


def visualizar_confusion(cm):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(ORDEN_CLASES)),
        yticks=range(len(ORDEN_CLASES)),
        xticklabels=ORDEN_CLASES,
        yticklabels=ORDEN_CLASES,
        ylabel="Real",
        xlabel="Predicho",
        title="Matriz de Confusión"
    )
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(FIGURA_MATRIZ, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Matriz de confusión guardada en: {FIGURA_MATRIZ}")


def probar_modelo(modelo):
    """Prueba rápida con valores representativos de cada clase."""
    print("\n  ── Prueba rápida con valores típicos ────────")
    casos = [
        ([5.0,  30.0, 180], "ligera   (esperado)"),
        ([55.0, 55.0, 220], "moderada (esperado)"),
        ([95.0, 85.0, 280], "pesada   (esperado)"),
    ]
    for valores, descripcion in casos:
        pred = modelo.predict([valores])[0]
        prob = modelo.predict_proba([valores])[0]
        confianza = max(prob) * 100
        print(f"  CPU:{valores[0]:5.1f}% MEM:{valores[1]:5.1f}% PROCS:{valores[2]} "
              f"→ {pred:<10} ({confianza:.0f}% confianza)  [{descripcion}]")


# ─── Entrada principal ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  ══════════════════════════════════════════")
    print("   Entrenamiento del Modelo — Proyecto SO")
    print("  ══════════════════════════════════════════\n")

    df                              = cargar_datos()
    X_train, X_test, y_train, y_test = preparar_datos(df)
    modelo                          = entrenar(X_train, y_train)
    cm, _                           = evaluar(modelo, X_train, X_test, y_train, y_test)

    guardar_modelo(modelo)
    visualizar_arbol(modelo)
    visualizar_confusion(cm)
    probar_modelo(modelo)

    print("\n  ══════════════════════════════════════════")
    print("   Entrenamiento completado exitosamente.")
    print("  ══════════════════════════════════════════\n")
