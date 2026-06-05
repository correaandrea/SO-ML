"""
experimento.py
Ejecuta el experimento controlado: N réplicas de carga de trabajo
con y sin el daemon de ML activo, registrando métricas de rendimiento.

Proyecto Final - Sistemas Operativos - Universidad de Antioquia

Uso:
    # Correr las réplicas SIN daemon (grupo control):
    python3 experimento.py --modo sin_ml --replicas 10

    # Correr las réplicas CON daemon (grupo ML):
    # (el daemon debe estar corriendo en otra terminal con: sudo python3 daemon.py)
    python3 experimento.py --modo con_ml --replicas 10

    # Ver resumen del CSV generado:
    python3 experimento.py --resumen

Notas:
    - Cada réplica lanza una carga CPU+memoria simultánea de duración fija.
    - Se mide: tiempo_real, tiempo_cpu, cpu_promedio, memoria_promedio.
    - Los resultados se acumulan en datos/resultados_experimento.csv.
    - Limpiar caché entre modos: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
"""

import os
import sys
import time
import csv
import argparse
import psutil
import multiprocessing
import math
import numpy as np
from datetime import datetime
from threading import Thread


# ─── Configuración ────────────────────────────────────────────────────────────

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RESULTADOS_PATH = os.path.join(BASE_DIR, "datos", "resultados_experimento.csv")

# Duración fija de la carga de trabajo por réplica (segundos)
DURACION_CARGA  = 30

# Pausa entre réplicas para que el sistema se estabilice (segundos)
PAUSA_ENTRE_REPLICAS = 10

# ─── Carga de trabajo (CPU-bound + Memory-bound combinada) ────────────────────

def _worker_cpu(duracion_seg):
    """Trabajador CPU-bound: criba de Eratóstenes en bucle."""
    limite = 50_000
    fin = time.time() + duracion_seg
    while time.time() < fin:
        criba = [True] * limite
        criba[0] = criba[1] = False
        for i in range(2, int(math.sqrt(limite)) + 1):
            if criba[i]:
                for j in range(i * i, limite, i):
                    criba[j] = False


def _worker_memoria(duracion_seg):
    """Trabajador memory-bound: operaciones sobre matrices grandes."""
    tamanio = 4_000   # ~122 MB por proceso
    fin = time.time() + duracion_seg
    while time.time() < fin:
        A = np.random.rand(tamanio, tamanio).astype(np.float64)
        B = np.random.rand(tamanio, tamanio).astype(np.float64)
        C = A + B
        _ = np.sum(C)
        time.sleep(0.05)
        del A, B, C


def lanzar_carga(duracion_seg):
    """
    Lanza la carga combinada (CPU + memoria) en procesos separados.
    Retorna la lista de procesos para poder esperar su finalización.
    """
    n_cpu = multiprocessing.cpu_count()  # un proceso por núcleo
    procesos = []

    # Procesos CPU-bound
    for _ in range(n_cpu):
        p = multiprocessing.Process(target=_worker_cpu, args=(duracion_seg,))
        p.start()
        procesos.append(p)

    # Proceso memory-bound
    p_mem = multiprocessing.Process(target=_worker_memoria, args=(duracion_seg,))
    p_mem.start()
    procesos.append(p_mem)

    return procesos


# ─── Monitor de métricas en tiempo real ───────────────────────────────────────

class MonitorMetricas:
    """
    Hilo secundario que muestrea CPU y memoria cada segundo
    mientras la carga está activa. Permite calcular promedios reales.
    """
    def __init__(self):
        self.muestras_cpu = []
        self.muestras_mem = []
        self._activo = False

    def iniciar(self):
        self._activo = True
        self._hilo = Thread(target=self._loop, daemon=True)
        self._hilo.start()

    def detener(self):
        self._activo = False
        self._hilo.join(timeout=3)

    def _loop(self):
        psutil.cpu_percent(interval=None)  # descarta primera lectura
        while self._activo:
            self.muestras_cpu.append(psutil.cpu_percent(interval=1))
            self.muestras_mem.append(psutil.virtual_memory().percent)

    def promedios(self):
        cpu_avg = sum(self.muestras_cpu) / len(self.muestras_cpu) if self.muestras_cpu else 0.0
        mem_avg = sum(self.muestras_mem) / len(self.muestras_mem) if self.muestras_mem else 0.0
        return round(cpu_avg, 2), round(mem_avg, 2)


# ─── Ejecución de una réplica ────────────────────────────────────────────────

def ejecutar_replica(replica_num, modo, duracion_seg=DURACION_CARGA):
    """
    Ejecuta una réplica completa:
      1. Inicia el monitor de métricas.
      2. Lanza la carga de trabajo combinada.
      3. Espera a que terminen todos los procesos.
      4. Detiene el monitor y calcula estadísticos.

    Retorna un dict con las métricas de la réplica.
    """
    print(f"\n  ┌─ Réplica {replica_num:02d}/{DURACION_CARGA}s "
          f"── modo: {modo.upper()} ───────────────────")

    monitor = MonitorMetricas()
    monitor.iniciar()

    # Tiempo de inicio
    t_inicio_real = time.time()
    t_inicio_cpu  = time.process_time()

    # Lanzar carga
    procesos = lanzar_carga(duracion_seg)

    # Barra de progreso mientras corre
    try:
        while any(p.is_alive() for p in procesos):
            elapsed = int(time.time() - t_inicio_real)
            progreso = int((elapsed / duracion_seg) * 28)
            barra = "█" * progreso + "░" * (28 - progreso)
            print(f"\r  │ [{barra}] {elapsed}/{duracion_seg}s", end="", flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n  Réplica interrumpida. Terminando procesos...")
        for p in procesos:
            p.terminate()
        for p in procesos:
            p.join()
        monitor.detener()
        raise

    for p in procesos:
        p.join()

    # Tiempo de fin
    t_fin_real = time.time()
    t_fin_cpu  = time.process_time()
    monitor.detener()

    tiempo_real = round(t_fin_real - t_inicio_real, 4)
    tiempo_cpu  = round(t_fin_cpu  - t_inicio_cpu,  4)
    cpu_avg, mem_avg = monitor.promedios()

    print(f"\r  │ [{'█' * 28}] {duracion_seg}/{duracion_seg}s", flush=True)
    print(f"  │  Tiempo real : {tiempo_real:.2f}s")
    print(f"  │  Tiempo CPU  : {tiempo_cpu:.2f}s")
    print(f"  │  CPU promedio: {cpu_avg:.1f}%")
    print(f"  │  MEM promedio: {mem_avg:.1f}%")
    print(f"  └──────────────────────────────────────────────")

    return {
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modo":            modo,
        "replica":         replica_num,
        "duracion_seg":    duracion_seg,
        "tiempo_real_seg": tiempo_real,
        "tiempo_cpu_seg":  tiempo_cpu,
        "cpu_promedio":    cpu_avg,
        "mem_promedio":    mem_avg,
    }


# ─── Gestión del archivo de resultados ───────────────────────────────────────

CABECERA = [
    "timestamp", "modo", "replica", "duracion_seg",
    "tiempo_real_seg", "tiempo_cpu_seg", "cpu_promedio", "mem_promedio"
]


def inicializar_csv():
    os.makedirs(os.path.dirname(RESULTADOS_PATH), exist_ok=True)
    if not os.path.exists(RESULTADOS_PATH):
        with open(RESULTADOS_PATH, "w", newline="") as f:
            csv.writer(f).writerow(CABECERA)
        print(f"  Archivo creado: {RESULTADOS_PATH}")
    else:
        print(f"  Agregando resultados a: {RESULTADOS_PATH}")


def guardar_fila(fila_dict):
    with open(RESULTADOS_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CABECERA)
        w.writerow(fila_dict)
        f.flush()


def resumen_csv():
    """Imprime un resumen de los resultados acumulados en el CSV."""
    if not os.path.exists(RESULTADOS_PATH):
        print("  No hay resultados aún. Ejecuta primero el experimento.")
        return

    import pandas as pd
    df = pd.read_csv(RESULTADOS_PATH)

    print(f"\n  ── Resumen de resultados ({'':─<40}")
    print(f"  Total de réplicas registradas: {len(df)}")
    print()

    for modo in df["modo"].unique():
        sub = df[df["modo"] == modo]
        print(f"  Modo: {modo.upper()}  ({len(sub)} réplicas)")
        print(f"    Tiempo real  — media: {sub['tiempo_real_seg'].mean():.2f}s  "
              f"std: {sub['tiempo_real_seg'].std():.2f}s")
        print(f"    Tiempo CPU   — media: {sub['tiempo_cpu_seg'].mean():.2f}s  "
              f"std: {sub['tiempo_cpu_seg'].std():.2f}s")
        print(f"    CPU promedio — media: {sub['cpu_promedio'].mean():.1f}%")
        print(f"    MEM promedio — media: {sub['mem_promedio'].mean():.1f}%")
        print()


# ─── Loop principal del experimento ──────────────────────────────────────────

def correr_experimento(modo, n_replicas):
    inicializar_csv()

    print(f"\n  ══════════════════════════════════════════")
    print(f"   Experimento — modo: {modo.upper()}")
    print(f"  ══════════════════════════════════════════")
    print(f"  Réplicas      : {n_replicas}")
    print(f"  Duración/rep  : {DURACION_CARGA}s")
    print(f"  Pausa entre   : {PAUSA_ENTRE_REPLICAS}s")
    if modo == "con_ml":
        print(f"\n  ⚠  Asegúrate de que el daemon esté corriendo:")
        print(f"     sudo python3 daemon.py")
    print()

    for i in range(1, n_replicas + 1):
        resultado = ejecutar_replica(i, modo)
        guardar_fila(resultado)

        if i < n_replicas:
            print(f"\n  Pausa de {PAUSA_ENTRE_REPLICAS}s antes de la siguiente réplica...\n")
            time.sleep(PAUSA_ENTRE_REPLICAS)

    print(f"\n  ══════════════════════════════════════════")
    print(f"   {n_replicas} réplicas completadas.")
    print(f"   Resultados guardados en: {RESULTADOS_PATH}")
    print(f"  ══════════════════════════════════════════\n")
    resumen_csv()


# ─── Entrada principal ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experimento controlado: carga con y sin daemon ML."
    )
    parser.add_argument(
        "--modo",
        type=str,
        choices=["sin_ml", "con_ml"],
        help="Grupo experimental: sin_ml (control) | con_ml (tratamiento)"
    )
    parser.add_argument(
        "--replicas",
        type=int,
        default=10,
        help="Número de réplicas a ejecutar (default: 10)"
    )
    parser.add_argument(
        "--resumen",
        action="store_true",
        help="Muestra resumen de los resultados acumulados y sale"
    )

    args = parser.parse_args()

    if args.resumen:
        print("\n  ══════════════════════════════════════════")
        print("   Resumen del Experimento — Proyecto SO")
        print("  ══════════════════════════════════════════")
        resumen_csv()
    elif args.modo is None:
        parser.error("Debes especificar --modo sin_ml|con_ml o usar --resumen")
    else:
        correr_experimento(args.modo, args.replicas)
