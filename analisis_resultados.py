"""
analisis_resultados.py
Análisis estadístico de los resultados del experimento.
Genera tablas, gráficas y la prueba t de Student lista para el informe.

Proyecto Final - Sistemas Operativos - Universidad de Antioquia

Uso:
    python3 analisis_resultados.py

Salida (en carpeta resultados/):
    - tabla_estadisticos.csv       Estadísticos descriptivos por grupo
    - grafica_tiempos.png          Boxplot de tiempos reales
    - grafica_cpu.png              Boxplot de CPU promedio
    - grafica_barras_comparacion.png  Barras con media ± desv. est.
    - informe_estadistico.txt      Reporte completo listo para copiar al informe
"""

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─── Configuración ────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_PATH = os.path.join(BASE_DIR, "datos", "resultados_experimento.csv")
SALIDA_DIR      = os.path.join(BASE_DIR, "resultados")

# Colores para los dos grupos
COLOR_CONTROL    = "#4C72B0"   # azul — sin ML
COLOR_TRATAMIENTO= "#DD8452"   # naranja — con ML
ALPHA            = 0.05        # nivel de significancia para la prueba t


# ─── Carga y validación de datos ─────────────────────────────────────────────

def cargar_datos():
    if not os.path.exists(RESULTADOS_PATH):
        print(f"\n  ERROR: No se encontró el archivo de resultados.")
        print(f"  Ejecuta primero: python3 experimento.py --modo sin_ml --replicas 10")
        print(f"                   python3 experimento.py --modo con_ml  --replicas 10")
        sys.exit(1)

    df = pd.read_csv(RESULTADOS_PATH)

    # Verificar que existan los dos grupos
    modos = df["modo"].unique()
    if "sin_ml" not in modos or "con_ml" not in modos:
        faltantes = [m for m in ["sin_ml", "con_ml"] if m not in modos]
        print(f"\n  ERROR: Faltan datos del grupo: {faltantes}")
        print(f"  Ejecuta el experimento para ese grupo antes de analizar.")
        sys.exit(1)

    control    = df[df["modo"] == "sin_ml"]["tiempo_real_seg"].values
    tratamiento= df[df["modo"] == "con_ml"]["tiempo_real_seg"].values

    print(f"  Datos cargados: {len(control)} réplicas sin_ml | {len(tratamiento)} réplicas con_ml")
    return df, control, tratamiento


# ─── Estadísticos descriptivos ───────────────────────────────────────────────

def calcular_estadisticos(df):
    """Calcula media, desv. est., mín, máx y mediana por grupo y métrica."""
    metricas = ["tiempo_real_seg", "tiempo_cpu_seg", "cpu_promedio", "mem_promedio"]
    labels   = {
        "tiempo_real_seg": "Tiempo real (s)",
        "tiempo_cpu_seg":  "Tiempo CPU (s)",
        "cpu_promedio":    "CPU promedio (%)",
        "mem_promedio":    "Memoria promedio (%)",
    }

    filas = []
    for metrica in metricas:
        for modo in ["sin_ml", "con_ml"]:
            sub = df[df["modo"] == modo][metrica]
            filas.append({
                "metrica":  labels[metrica],
                "modo":     modo,
                "n":        len(sub),
                "media":    round(sub.mean(), 4),
                "std":      round(sub.std(ddof=1), 4),
                "min":      round(sub.min(), 4),
                "max":      round(sub.max(), 4),
                "mediana":  round(sub.median(), 4),
            })

    tabla = pd.DataFrame(filas)
    return tabla


# ─── Prueba t de Student ──────────────────────────────────────────────────────

def prueba_t(control, tratamiento):
    """
    Prueba t de Student de dos muestras independientes (Welch).
    H0: no hay diferencia en el tiempo de ejecución entre grupos.
    H1: hay diferencia significativa.
    """
    t_stat, p_valor = stats.ttest_ind(control, tratamiento, equal_var=False)

    # Cohen's d — tamaño del efecto
    pooled_std = np.sqrt(
        (control.std(ddof=1)**2 + tratamiento.std(ddof=1)**2) / 2
    )
    cohen_d = (control.mean() - tratamiento.mean()) / pooled_std if pooled_std > 0 else 0.0

    # Diferencia relativa
    diferencia_pct = ((control.mean() - tratamiento.mean()) / control.mean()) * 100

    return {
        "t_stat":         round(t_stat, 4),
        "p_valor":        round(p_valor, 6),
        "significativo":  p_valor < ALPHA,
        "cohen_d":        round(cohen_d, 4),
        "diferencia_pct": round(diferencia_pct, 2),
        "media_control":  round(control.mean(), 4),
        "media_tratamiento": round(tratamiento.mean(), 4),
        "std_control":    round(control.std(ddof=1), 4),
        "std_tratamiento":round(tratamiento.std(ddof=1), 4),
    }


# ─── Gráficas ─────────────────────────────────────────────────────────────────

def grafica_boxplot_tiempos(df, ruta):
    fig, ax = plt.subplots(figsize=(7, 5))

    datos_plot = [
        df[df["modo"] == "sin_ml"]["tiempo_real_seg"].values,
        df[df["modo"] == "con_ml"]["tiempo_real_seg"].values,
    ]
    bp = ax.boxplot(
        datos_plot,
        labels=["Sin ML\n(Control)", "Con ML\n(Tratamiento)"],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=5, alpha=0.6),
    )
    bp["boxes"][0].set_facecolor(COLOR_CONTROL)
    bp["boxes"][1].set_facecolor(COLOR_TRATAMIENTO)

    ax.set_ylabel("Tiempo real de ejecución (segundos)", fontsize=11)
    ax.set_title("Distribución del tiempo de ejecución\npor grupo experimental", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Boxplot tiempos guardado en: {ruta}")


def grafica_boxplot_cpu(df, ruta):
    fig, ax = plt.subplots(figsize=(7, 5))

    datos_plot = [
        df[df["modo"] == "sin_ml"]["cpu_promedio"].values,
        df[df["modo"] == "con_ml"]["cpu_promedio"].values,
    ]
    bp = ax.boxplot(
        datos_plot,
        labels=["Sin ML\n(Control)", "Con ML\n(Tratamiento)"],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker="o", markersize=5, alpha=0.6),
    )
    bp["boxes"][0].set_facecolor(COLOR_CONTROL)
    bp["boxes"][1].set_facecolor(COLOR_TRATAMIENTO)

    ax.set_ylabel("CPU promedio durante la réplica (%)", fontsize=11)
    ax.set_title("Uso de CPU promedio\npor grupo experimental", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Boxplot CPU guardado en: {ruta}")


def grafica_barras_comparacion(df, prueba, ruta):
    """Barras con media ± desviación estándar para tiempo real y CPU."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    metricas = [
        ("tiempo_real_seg", "Tiempo real de ejecución (s)"),
        ("cpu_promedio",    "CPU promedio (%)"),
    ]
    colores = [COLOR_CONTROL, COLOR_TRATAMIENTO]
    etiquetas = ["Sin ML", "Con ML"]

    for ax, (metrica, ylabel) in zip(axes, metricas):
        medias = [df[df["modo"] == m][metrica].mean() for m in ["sin_ml", "con_ml"]]
        stds   = [df[df["modo"] == m][metrica].std(ddof=1) for m in ["sin_ml", "con_ml"]]

        bars = ax.bar(etiquetas, medias, color=colores, width=0.5,
                      edgecolor="white", linewidth=1.2)
        ax.errorbar(etiquetas, medias, yerr=stds,
                    fmt="none", color="black", capsize=6, linewidth=2)

        # Anotar valores encima de barras
        for bar, media, std in zip(bars, medias, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                media + std + max(medias) * 0.02,
                f"{media:.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_facecolor("#f9f9f9")
        ax.set_ylim(0, max(medias) * 1.3)

    # Anotación de significancia
    sig_texto = (
        f"Prueba t (Welch): t={prueba['t_stat']}, p={prueba['p_valor']}\n"
        f"{'Diferencia SIGNIFICATIVA' if prueba['significativo'] else 'Sin diferencia significativa'} "
        f"(α={ALPHA}) | d de Cohen = {prueba['cohen_d']}"
    )
    fig.text(0.5, -0.02, sig_texto, ha="center", fontsize=9,
             style="italic", color="#444444")

    fig.suptitle("Comparación de grupos: Sin ML vs. Con ML",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Gráfica de barras guardada en: {ruta}")


def grafica_evolucion_replicas(df, ruta):
    """Línea de tiempo real por réplica, mostrando ambos grupos."""
    fig, ax = plt.subplots(figsize=(9, 4))

    for modo, color, label in [
        ("sin_ml", COLOR_CONTROL,    "Sin ML (Control)"),
        ("con_ml", COLOR_TRATAMIENTO,"Con ML (Tratamiento)"),
    ]:
        sub = df[df["modo"] == modo].sort_values("replica")
        ax.plot(sub["replica"], sub["tiempo_real_seg"],
                marker="o", color=color, label=label, linewidth=2, markersize=6)

    ax.set_xlabel("Número de réplica", fontsize=11)
    ax.set_ylabel("Tiempo real de ejecución (s)", fontsize=11)
    ax.set_title("Evolución del tiempo por réplica", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.5)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Gráfica de evolución guardada en: {ruta}")


# ─── Informe de texto ─────────────────────────────────────────────────────────

def generar_informe_texto(tabla, prueba, ruta):
    """Genera un reporte .txt con todos los resultados, listo para el informe."""
    lineas = [
        "═" * 60,
        "ANÁLISIS ESTADÍSTICO — PROYECTO FINAL SO",
        "Universidad de Antioquia",
        "Sistemas Operativos y Laboratorio",
        "═" * 60,
        "",
        "── 1. Estadísticos Descriptivos ────────────────────────",
        "",
    ]

    # Tabla descriptiva formateada
    for metrica in tabla["metrica"].unique():
        sub = tabla[tabla["metrica"] == metrica]
        lineas.append(f"  {metrica}")
        lineas.append(f"  {'Grupo':<20} {'n':>4} {'Media':>10} {'Desv.Est':>10} "
                      f"{'Mín':>10} {'Máx':>10} {'Mediana':>10}")
        lineas.append(f"  {'─'*74}")
        for _, row in sub.iterrows():
            lineas.append(
                f"  {row['modo']:<20} {int(row['n']):>4} {row['media']:>10.4f} "
                f"{row['std']:>10.4f} {row['min']:>10.4f} {row['max']:>10.4f} "
                f"{row['mediana']:>10.4f}"
            )
        lineas.append("")

    lineas += [
        "── 2. Prueba t de Student (Welch) — Tiempo real ────────",
        "",
        f"  Hipótesis nula (H0): No hay diferencia en el tiempo de",
        f"    ejecución entre el grupo sin ML y con ML.",
        f"  Hipótesis alternativa (H1): Sí hay diferencia.",
        "",
        f"  Media grupo control (sin_ml) : {prueba['media_control']:.4f} s",
        f"  Media grupo tratamiento (con_ml): {prueba['media_tratamiento']:.4f} s",
        f"  Diferencia relativa          : {prueba['diferencia_pct']:+.2f}%",
        "",
        f"  Estadístico t  : {prueba['t_stat']}",
        f"  Valor p        : {prueba['p_valor']}",
        f"  Nivel α        : {ALPHA}",
        f"  Resultado      : {'Se RECHAZA H0 — diferencia estadísticamente significativa.' if prueba['significativo'] else 'No se puede rechazar H0 — diferencia NO significativa.'}",
        "",
        f"  Tamaño del efecto (d de Cohen): {prueba['cohen_d']}",
        f"  Interpretación: {'pequeño (<0.2)' if abs(prueba['cohen_d'])<0.2 else 'mediano (0.2–0.8)' if abs(prueba['cohen_d'])<0.8 else 'grande (>0.8)'}",
        "",
        "── 3. Interpretación ───────────────────────────────────",
        "",
    ]

    if prueba["significativo"]:
        direccion = "reducción" if prueba["diferencia_pct"] > 0 else "aumento"
        lineas += [
            f"  El daemon ML produjo una {direccion} del {abs(prueba['diferencia_pct']):.2f}%",
            f"  en el tiempo de ejecución (p = {prueba['p_valor']} < {ALPHA}).",
            f"  El tamaño del efecto ({abs(prueba['cohen_d']):.2f}) es consistente con",
            f"  un impacto {'pequeño' if abs(prueba['cohen_d'])<0.2 else 'moderado' if abs(prueba['cohen_d'])<0.8 else 'grande'}.",
        ]
    else:
        lineas += [
            f"  La diferencia observada ({prueba['diferencia_pct']:+.2f}%) no es estadísticamente",
            f"  significativa (p = {prueba['p_valor']} ≥ {ALPHA}).",
            f"  El daemon ML no produjo un impacto medible en el tiempo de ejecución",
            f"  bajo las condiciones de este experimento.",
        ]

    lineas += [
        "",
        "═" * 60,
    ]

    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))

    print(f"  ✓ Informe de texto guardado en: {ruta}")

    # También imprime el reporte en pantalla
    print()
    for l in lineas:
        print(" ", l)


# ─── Entrada principal ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  ══════════════════════════════════════════")
    print("   Análisis de Resultados — Proyecto SO")
    print("  ══════════════════════════════════════════\n")

    os.makedirs(SALIDA_DIR, exist_ok=True)

    # 1. Cargar datos
    df, control, tratamiento = cargar_datos()

    # 2. Estadísticos descriptivos
    print("\n  Calculando estadísticos descriptivos...")
    tabla = calcular_estadisticos(df)
    tabla.to_csv(os.path.join(SALIDA_DIR, "tabla_estadisticos.csv"), index=False)
    print(f"  ✓ Tabla guardada en: {SALIDA_DIR}/tabla_estadisticos.csv")

    # 3. Prueba t
    print("\n  Ejecutando prueba t de Student (Welch)...")
    prueba = prueba_t(control, tratamiento)

    # 4. Gráficas
    print("\n  Generando gráficas...")
    grafica_boxplot_tiempos(df, os.path.join(SALIDA_DIR, "grafica_tiempos.png"))
    grafica_boxplot_cpu(df,     os.path.join(SALIDA_DIR, "grafica_cpu.png"))
    grafica_barras_comparacion(df, prueba,
                               os.path.join(SALIDA_DIR, "grafica_barras_comparacion.png"))
    grafica_evolucion_replicas(df,
                               os.path.join(SALIDA_DIR, "grafica_evolucion_replicas.png"))

    # 5. Informe de texto
    print("\n  Generando informe estadístico...")
    generar_informe_texto(tabla, prueba,
                          os.path.join(SALIDA_DIR, "informe_estadistico.txt"))

    print("\n  ══════════════════════════════════════════")
    print("   Análisis completado.")
    print(f"   Archivos en: {SALIDA_DIR}/")
    print("  ══════════════════════════════════════════\n")
