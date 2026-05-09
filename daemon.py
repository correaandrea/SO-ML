"""
daemon.py
Daemon de ajuste dinámico de prioridades basado en ML.
Proyecto Final - Sistemas Operativos - Universidad de Antioquia

Uso:
    sudo python3 daemon.py              # corre indefinidamente
    sudo python3 daemon.py --duracion 120  # corre por N segundos y para
"""

import os
import sys
import time
import pickle
import argparse
import psutil
from datetime import datetime


# ─── Configuración ────────────────────────────────────────────────────────────

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH    = os.path.join(BASE_DIR, "modelo", "arbol.pkl")
LOG_PATH       = os.path.join(BASE_DIR, "logs", "daemon.log")
INTERVALO_SEG  = 2       # cada cuántos segundos evalúa el sistema
TOP_PROCESOS   = 3       # cuántos procesos pesados interviene por ciclo

# Valores de nice a aplicar según nivel de carga detectado
# nice va de -20 (máxima prioridad) a +19 (mínima prioridad)
# Solo subimos el nice (bajamos prioridad) de procesos pesados
NICE_POR_NIVEL = {
    "ligera":   0,    # no interviene
    "moderada": 5,    # baja un poco la prioridad
    "pesada":  10,    # baja bastante la prioridad
}

# Procesos del sistema que NUNCA deben ser modificados
PROCESOS_PROTEGIDOS = {
    "systemd", "init", "kthreadd", "python3", "daemon.py",
    "bash", "sh", "sshd", "gnome-shell", "Xorg", "Xwayland",
    "dbus-daemon", "NetworkManager", "pulseaudio", "pipewire"
}


# ─── Funciones ────────────────────────────────────────────────────────────────

def log(mensaje, nivel="INFO"):
    """Escribe en pantalla y en el archivo de log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = f"[{ts}] [{nivel}] {mensaje}"
    print(linea, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(linea + "\n")
    except Exception:
        pass


def cargar_modelo():
    """Carga el árbol de decisión serializado."""
    if not os.path.exists(MODELO_PATH):
        print(f"ERROR: No se encontró el modelo en {MODELO_PATH}")
        print("       Ejecuta primero: python3 entrenar_modelo.py")
        sys.exit(1)
    with open(MODELO_PATH, "rb") as f:
        modelo = pickle.load(f)
    return modelo


def obtener_metricas():
    """Captura las tres métricas del sistema."""
    # Primera llamada de cpu_percent siempre da 0.0, por eso usamos interval=1
    cpu     = psutil.cpu_percent(interval=1)
    memoria = psutil.virtual_memory().percent
    procs   = len(psutil.pids())
    return cpu, memoria, procs


def clasificar_sistema(modelo, cpu, memoria, procs):
    """Usa el modelo para clasificar el estado actual del sistema."""
    prediccion = modelo.predict([[cpu, memoria, procs]])[0]
    proba      = modelo.predict_proba([[cpu, memoria, procs]])[0]
    confianza  = max(proba) * 100
    return prediccion, confianza


def obtener_procesos_pesados(n=TOP_PROCESOS):
    """
    Devuelve los N procesos que más CPU están consumiendo,
    excluyendo los protegidos y los del propio sistema.
    """
    candidatos = []

    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'nice', 'username']):
        try:
            info = proc.info
            nombre = info['name'] or ""

            # Saltar procesos protegidos
            if nombre in PROCESOS_PROTEGIDOS:
                continue

            # Saltar procesos del kernel (pid muy bajo o sin usuario)
            if info['pid'] < 100:
                continue

            # Solo considerar procesos con uso real de CPU
            if info['cpu_percent'] is None or info['cpu_percent'] < 1.0:
                continue

            candidatos.append({
                'pid':     info['pid'],
                'nombre':  nombre,
                'cpu':     info['cpu_percent'],
                'nice':    info['nice'] or 0,
            })

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Ordenar por CPU descendente y tomar los top N
    candidatos.sort(key=lambda x: x['cpu'], reverse=True)
    return candidatos[:n]


def aplicar_renice(procesos, nice_valor):
    """
    Aplica renice a los procesos pesados.
    Solo sube el nice (baja prioridad), nunca lo reduce.
    Requiere correr con sudo para modificar procesos de otros usuarios.
    """
    if nice_valor == 0 or not procesos:
        return 0

    modificados = 0
    for proc in procesos:
        pid       = proc['pid']
        nice_actual = proc['nice']

        # Solo subir el nice, nunca bajarlo (no queremos dar más prioridad)
        nuevo_nice = max(nice_actual, nice_valor)

        if nuevo_nice == nice_actual:
            continue  # ya tiene ese nice o mayor, no hace falta cambiar

        try:
            p = psutil.Process(pid)
            p.nice(nuevo_nice)
            modificados += 1
            log(f"  renice PID {pid} ({proc['nombre']}) "
                f"nice {nice_actual} → {nuevo_nice}  CPU:{proc['cpu']:.1f}%")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            log(f"  No se pudo modificar PID {pid} ({proc['nombre']}): {e}", "WARN")

    return modificados


def ciclo(modelo, cpu, memoria, procs):
    """Ejecuta un ciclo completo: clasificar → decidir → actuar."""
    nivel, confianza = clasificar_sistema(modelo, cpu, memoria, procs)
    nice_valor       = NICE_POR_NIVEL[nivel]

    log(f"CPU:{cpu:5.1f}%  MEM:{memoria:5.1f}%  PROCS:{procs}  "
        f"→  {nivel.upper():<10} ({confianza:.0f}% confianza)")

    if nivel == "ligera":
        log("  Sin intervención — sistema estable.")
        return nivel, 0

    # Detectar procesos pesados e intervenir
    pesados = obtener_procesos_pesados()

    if not pesados:
        log("  No se encontraron procesos candidatos para renice.")
        return nivel, 0

    nombres = ', '.join(f"{p['nombre']}({p['pid']})" for p in pesados)
    log(f"  Procesos pesados detectados: {nombres}")

    modificados = aplicar_renice(pesados, nice_valor)
    log(f"  Procesos modificados: {modificados}")

    return nivel, modificados


def loop_principal(modelo, duracion=None):
    """Loop principal del daemon."""
    log("═" * 55)
    log("Daemon ML iniciado.")
    log(f"Modelo     : {MODELO_PATH}")
    log(f"Intervalo  : {INTERVALO_SEG}s")
    log(f"Top procs  : {TOP_PROCESOS}")
    if duracion:
        log(f"Duración   : {duracion}s")
    log("═" * 55)

    inicio   = time.time()
    ciclos   = 0
    conteo   = {"ligera": 0, "moderada": 0, "pesada": 0}

    try:
        while True:
            if duracion and (time.time() - inicio) >= duracion:
                break

            cpu, memoria, procs = obtener_metricas()
            nivel, _ = ciclo(modelo, cpu, memoria, procs)
            conteo[nivel] += 1
            ciclos += 1

            time.sleep(max(0, INTERVALO_SEG - 1))  # cpu_percent ya tardó 1s

    except KeyboardInterrupt:
        log("\nDaemon detenido manualmente (Ctrl+C).")

    # Resumen final
    elapsed = int(time.time() - inicio)
    log("═" * 55)
    log(f"Resumen — {ciclos} ciclos en {elapsed}s")
    log(f"  Ligera  : {conteo['ligera']} ciclos")
    log(f"  Moderada: {conteo['moderada']} ciclos")
    log(f"  Pesada  : {conteo['pesada']} ciclos")
    log("═" * 55)


# ─── Entrada principal ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Daemon de ajuste dinámico de prioridades basado en ML."
    )
    parser.add_argument(
        "--duracion",
        type=int,
        default=None,
        help="Duración en segundos (default: corre indefinidamente)"
    )
    args = parser.parse_args()

    # Crear carpeta de logs
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    # Verificar privilegios (renice requiere sudo)
    if os.geteuid() != 0:
        print("\n  ADVERTENCIA: El daemon funciona mejor con sudo.")
        print("  Sin sudo, solo podrá modificar procesos de tu propio usuario.")
        print("  Ejecuta: sudo python3 daemon.py\n")
        time.sleep(2)

    modelo = cargar_modelo()
    loop_principal(modelo, duracion=args.duracion)
