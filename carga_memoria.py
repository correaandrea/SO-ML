"""
carga_memoria.py
Script de carga memory-bound para generar estrés controlado.
Proyecto Final - Sistemas Operativos - Universidad de Antioquia

Uso:
    python3 carga_memoria.py --intensidad media  --duracion 60
    python3 carga_memoria.py --intensidad alta   --duracion 60
"""

import argparse
import time
import multiprocessing
import numpy as np


# ─── Configuración ────────────────────────────────────────────────────────────

# Tamaño de las matrices según intensidad
# En una VM con 4 GB RAM:
#   media : ocupa ~500 MB  (~12% de 4 GB)
#   alta  : ocupa ~1.5 GB  (~37% de 4 GB)
TAMANIO_MATRIZ = {
    "media": 4_000,   # matriz 4000x4000 float64 ≈ 122 MB por proceso
    "alta" : 8_000,   # matriz 8000x8000 float64 ≈ 488 MB por proceso
}
N_PROCESOS = {
    "media": 1,
    "alta" : 3,
}


# ─── Funciones ────────────────────────────────────────────────────────────────

def trabajador_memoria(duracion_seg, tamanio):
    """
    Proceso hijo que realiza operaciones intensivas sobre matrices grandes.
    Aloca, opera y libera memoria de forma continua para mantener
    presión sostenida sobre la RAM.
    """
    fin = time.time() + duracion_seg

    while time.time() < fin:
        # Alocar matriz grande
        A = np.random.rand(tamanio, tamanio).astype(np.float64)
        B = np.random.rand(tamanio, tamanio).astype(np.float64)

        # Operaciones que mantienen los datos en memoria activamente
        C = A + B
        D = np.sum(C)   # evita que el compilador optimice y elimine C

        # Pequeña pausa para no saturar también el CPU
        time.sleep(0.05)

        del A, B, C, D  # liberar explícitamente para el siguiente ciclo


def lanzar_carga(duracion_seg, intensidad):
    """
    Lanza N procesos paralelos de presión de memoria.
    """
    tamanio   = TAMANIO_MATRIZ[intensidad]
    n_procs   = N_PROCESOS[intensidad]
    mem_aprox = (tamanio ** 2 * 8 * 2 * n_procs) / (1024 ** 3)  # GB aprox

    print(f"\n  ══════════════════════════════════════════")
    print(f"   Carga Memory-bound — Proyecto SO")
    print(f"  ══════════════════════════════════════════")
    print(f"  Intensidad  : {intensidad.upper()}")
    print(f"  Procesos    : {n_procs}")
    print(f"  Matriz      : {tamanio}x{tamanio} float64")
    print(f"  RAM aprox.  : ~{mem_aprox:.1f} GB en uso")
    print(f"  Duración    : {duracion_seg} segundos")
    print(f"\n  Iniciando carga... (Ctrl+C para detener)\n")

    procesos = []
    for _ in range(n_procs):
        p = multiprocessing.Process(
            target=trabajador_memoria,
            args=(duracion_seg, tamanio)
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

    print(f"\n\n  ✓ Carga de memoria finalizada.\n")


# ─── Entrada principal ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generador de carga memory-bound para pruebas de estrés."
    )
    parser.add_argument(
        "--intensidad",
        type=str,
        required=True,
        choices=["media", "alta"],
        help="Nivel de carga: media (~500 MB RAM) | alta (~1.5 GB RAM)"
    )
    parser.add_argument(
        "--duracion",
        type=int,
        default=60,
        help="Duración en segundos (default: 60)"
    )

    args = parser.parse_args()
    lanzar_carga(args.duracion, args.intensidad)
