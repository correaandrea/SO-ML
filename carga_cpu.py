"""
carga_cpu.py
Script de carga CPU-bound para generar estrés controlado.
Proyecto Final - Sistemas Operativos - Universidad de Antioquia

Uso:
    python3 carga_cpu.py --intensidad media  --duracion 60
    python3 carga_cpu.py --intensidad alta   --duracion 60
"""

import argparse
import time
import multiprocessing
import math


# ─── Funciones ────────────────────────────────────────────────────────────────

def trabajador_cpu(duracion_seg, intensidad):
    """
    Proceso hijo que realiza cálculos matemáticos intensivos.
    Calcula números primos con criba de Eratóstenes en bucle
    para mantener el CPU ocupado de forma sostenida.
    """
    limite = 50_000 if intensidad == "alta" else 20_000
    fin = time.time() + duracion_seg

    while time.time() < fin:
        # Criba de Eratóstenes
        criba = [True] * limite
        criba[0] = criba[1] = False
        for i in range(2, int(math.sqrt(limite)) + 1):
            if criba[i]:
                for j in range(i * i, limite, i):
                    criba[j] = False


def lanzar_carga(duracion_seg, intensidad):
    """
    Lanza N procesos paralelos según la intensidad solicitada.
    - media : 1 proceso  (~50% CPU en una VM de 2 núcleos)
    - alta  : 2 procesos (~100% CPU en una VM de 2 núcleos)
    """
    n_procesos = 1 if intensidad == "media" else multiprocessing.cpu_count()

    print(f"\n  ══════════════════════════════════════════")
    print(f"   Carga CPU-bound — Proyecto SO")
    print(f"  ══════════════════════════════════════════")
    print(f"  Intensidad : {intensidad.upper()}")
    print(f"  Procesos   : {n_procesos}")
    print(f"  Duración   : {duracion_seg} segundos")
    print(f"\n  Iniciando carga... (Ctrl+C para detener)\n")

    procesos = []
    for _ in range(n_procesos):
        p = multiprocessing.Process(
            target=trabajador_cpu,
            args=(duracion_seg, intensidad)
        )
        p.start()
        procesos.append(p)

    inicio = time.time()
    try:
        while any(p.is_alive() for p in procesos):
            elapsed = int(time.time() - inicio)
            progreso = int((elapsed / duracion_seg) * 30)
            barra = "█" * progreso + "░" * (30 - progreso)
            print(f"\r  [{barra}] {elapsed}/{duracion_seg}s", end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n  Deteniendo procesos...")
        for p in procesos:
            p.terminate()

    for p in procesos:
        p.join()

    print(f"\n\n  ✓ Carga CPU finalizada.\n")


# ─── Entrada principal ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generador de carga CPU-bound para pruebas de estrés."
    )
    parser.add_argument(
        "--intensidad",
        type=str,
        required=True,
        choices=["media", "alta"],
        help="Nivel de carga: media (~50%% CPU) | alta (~100%% CPU)"
    )
    parser.add_argument(
        "--duracion",
        type=int,
        default=60,
        help="Duración en segundos (default: 60)"
    )

    args = parser.parse_args()
    lanzar_carga(args.duracion, args.intensidad)
