"""
recolector.py
Script de recolección de métricas del sistema para entrenamiento del modelo.
Proyecto Final - Sistemas Operativos - Universidad de Antioquia

Uso:
    python3 recolector.py --etiqueta ligera   --duracion 60
    python3 recolector.py --etiqueta moderada --duracion 60
    python3 recolector.py --etiqueta pesada   --duracion 60
"""

import psutil
import csv
import time
import argparse
import os
from datetime import datetime


# ─── Configuración ────────────────────────────────────────────────────────────

INTERVALO_SEG   = 1       # cada cuántos segundos se toma una muestra
ARCHIVO_SALIDA  = os.path.expanduser("~/so-proyecto/datos/dataset.csv")
ETIQUETAS_VALIDAS = {"ligera", "moderada", "pesada"}


# ─── Funciones ────────────────────────────────────────────────────────────────

def obtener_metricas():
    """Captura una muestra de las tres métricas del sistema."""
    cpu     = psutil.cpu_percent(interval=None)   # % uso de CPU
    memoria = psutil.virtual_memory().percent     # % memoria RAM usada
    procesos = len(psutil.pids())                 # número de procesos activos
    return cpu, memoria, procesos


def inicializar_archivo(ruta):
    """Crea el archivo CSV con encabezado si no existe."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    if not os.path.exists(ruta):
        with open(ruta, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "cpu_percent", "memory_percent",
                             "num_processes", "etiqueta"])
        print(f"  Archivo creado: {ruta}")
    else:
        print(f"  Agregando datos a: {ruta}")


def recolectar(etiqueta, duracion_seg):
    """
    Loop principal de recolección.
    Toma una muestra cada INTERVALO_SEG durante duracion_seg segundos.
    """
    inicializar_archivo(ARCHIVO_SALIDA)

    total_muestras = duracion_seg // INTERVALO_SEG
    muestras_tomadas = 0

    print(f"\n  Etiqueta   : {etiqueta.upper()}")
    print(f"  Duración   : {duracion_seg} segundos")
    print(f"  Muestras   : ~{total_muestras}")
    print(f"  Intervalo  : {INTERVALO_SEG} segundo(s)")
    print("\n  Iniciando recolección... (Ctrl+C para detener antes)\n")

    # Primer cpu_percent siempre da 0.0 — lo descartamos
    psutil.cpu_percent(interval=None)
    time.sleep(INTERVALO_SEG)

    inicio = time.time()

    try:
        with open(ARCHIVO_SALIDA, "a", newline="") as f:
            writer = csv.writer(f)

            while (time.time() - inicio) < duracion_seg:
                ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cpu, mem, procs = obtener_metricas()

                writer.writerow([ts, cpu, mem, procs, etiqueta])
                f.flush()   # escribe al disco inmediatamente

                muestras_tomadas += 1
                elapsed = int(time.time() - inicio)

                # Barra de progreso simple
                progreso = int((elapsed / duracion_seg) * 30)
                barra = "█" * progreso + "░" * (30 - progreso)
                print(f"\r  [{barra}] {elapsed}/{duracion_seg}s  "
                      f"CPU:{cpu:5.1f}%  MEM:{mem:5.1f}%  PROCS:{procs}",
                      end="", flush=True)

                time.sleep(INTERVALO_SEG)

    except KeyboardInterrupt:
        print("\n\n  Recolección detenida manualmente.")

    print(f"\n\n  ✓ Muestras guardadas: {muestras_tomadas}")
    print(f"  ✓ Archivo           : {ARCHIVO_SALIDA}")
    resumen_archivo()


def resumen_archivo():
    """Muestra cuántas muestras hay por etiqueta en el dataset actual."""
    if not os.path.exists(ARCHIVO_SALIDA):
        return

    conteo = {"ligera": 0, "moderada": 0, "pesada": 0}
    total  = 0

    with open(ARCHIVO_SALIDA, "r") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            etiq = fila.get("etiqueta", "")
            if etiq in conteo:
                conteo[etiq] += 1
            total += 1

    print(f"\n  ── Resumen del dataset ──────────────────")
    for etiq, n in conteo.items():
        barra = "▓" * (n // 5)
        print(f"  {etiq:<10}: {n:>4} muestras  {barra}")
    print(f"  {'TOTAL':<10}: {total:>4} muestras")
    print(f"  ─────────────────────────────────────────\n")


# ─── Entrada principal ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recolector de métricas del sistema para entrenamiento ML."
    )
    parser.add_argument(
        "--etiqueta",
        type=str,
        required=False,
	default=None,
        choices=list(ETIQUETAS_VALIDAS),
        help="Nivel de carga durante la recolección: ligera | moderada | pesada"
    )
    parser.add_argument(
        "--duracion",
        type=int,
        default=60,
        help="Duración en segundos (default: 60)"
    )
    parser.add_argument(
        "--resumen",
        action="store_true",
        help="Solo muestra el resumen del dataset actual y sale"
    )

    args = parser.parse_args()

    print("\n  ══════════════════════════════════════════")
    print("   Recolector de Métricas — Proyecto SO")
    print("  ══════════════════════════════════════════")

    if args.resumen:
        resumen_archivo()
    else:
        recolectar(args.etiqueta, args.duracion)
