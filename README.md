# Técnicas de Aprendizaje Automático para Sistemas Operativos

**Curso:** Sistemas Operativos y Laboratorio — Universidad de Antioquia  
**Integrantes:** Andrea Correa Arango, Emanuel Vásquez Yepes

## Descripción

Prototipo en espacio de usuario que usa un árbol de decisión para clasificar el nivel de carga del sistema (ligera, moderada, pesada) y ajustar dinámicamente la prioridad de procesos mediante `renice`, sin modificar el kernel.

---

## Estructura del proyecto

```
so-proyecto/
│   carga_cpu.py                  # Generador de carga CPU-bound
│   carga_memoria.py              # Generador de carga memory-bound
│   daemon.py                     # Daemon ML de ajuste de prioridades
│   entrenar_modelo.py            # Entrenamiento del árbol de decisión
│   experimento.py                # Experimento controlado (réplicas)
│   analisis_resultados.py        # Análisis estadístico y gráficas
│   recolector.py                 # Recolector de métricas del sistema
│   README.md
│
├── datos/
│   ├── dataset.csv               # Generado por recolector.py
│   └── resultados_experimento.csv  # Generado por experimento.py
│
├── modelo/
│   ├── arbol.pkl                 # Generado por entrenar_modelo.py
│   ├── arbol_visual.png          # Generado por entrenar_modelo.py
│   └── matriz_confusion.png      # Generado por entrenar_modelo.py
│
└── resultados/                   # Generado por analisis_resultados.py
    ├── tabla_estadisticos.csv
    ├── grafica_tiempos.png
    ├── grafica_cpu.png
    ├── grafica_barras_comparacion.png
    ├── grafica_evolucion_replicas.png
    └── informe_estadistico.txt
```

---

## Requisitos del sistema

- Ubuntu 22.04 LTS
- Python 3.10+
- `htop` (opcional, para monitorear en tiempo real)

---

## PASO 0 — Configuración inicial del entorno

Estos pasos se hacen **una sola vez** al clonar/descargar el proyecto.

```bash
# 1. Crear el directorio del proyecto y entrar
mkdir ~/so-proyecto
cd ~/so-proyecto

# 2. Copiar todos los archivos .py y el README aquí
#    (si los descargaste en otra carpeta, muévelos)

# 3. Crear las carpetas que los scripts esperan encontrar
mkdir -p datos modelo resultados logs

# 4. Crear el entorno virtual
python3 -m venv venv

# 5. Activarlo
source venv/bin/activate

# 6. Instalar dependencias
pip install psutil scikit-learn numpy pandas matplotlib scipy

# 7. Instalar herramientas del sistema (si no están)
sudo apt install -y htop stress-ng
```

> **Nota:** Cada vez que abras una nueva terminal debes activar el entorno virtual antes de correr cualquier script:
> ```bash
> cd ~/so-proyecto && source venv/bin/activate
> ```

---

## PASO 1 — Recolección de datos

Se hacen **3 rondas** para capturar los tres niveles de carga. Cada ronda genera ~120 muestras. Al finalizar las 3 rondas el archivo `datos/dataset.csv` tendrá ~360 filas listas para entrenar.

### Ronda 1 — Carga LIGERA (sistema en reposo)

Solo el recolector, sin ninguna carga extra.

**Terminal 1:**
```bash
cd ~/so-proyecto && source venv/bin/activate
python3 recolector.py --etiqueta ligera --duracion 120
```

Espera a que termine. No hagas nada más en la VM mientras corre.

---

### Ronda 2 — Carga MODERADA

Necesitas **dos terminales activas al mismo tiempo**.

**Terminal 1 — lanza la carga primero:**
```bash
cd ~/so-proyecto && source venv/bin/activate
python3 carga_cpu.py --intensidad media --duracion 150
```

**Terminal 2 — unos 10 segundos después, arranca el recolector:**
```bash
cd ~/so-proyecto && source venv/bin/activate
python3 recolector.py --etiqueta moderada --duracion 120
```

---

### Ronda 3 — Carga PESADA

Necesitas **tres terminales activas al mismo tiempo**.

**Terminal 1 — carga CPU:**
```bash
cd ~/so-proyecto && source venv/bin/activate
python3 carga_cpu.py --intensidad alta --duracion 150
```

**Terminal 2 — carga memoria (~5 s después):**
```bash
cd ~/so-proyecto && source venv/bin/activate
python3 carga_memoria.py --intensidad alta --duracion 150
```

**Terminal 3 — recolector (~10 s después de la primera carga):**
```bash
cd ~/so-proyecto && source venv/bin/activate
python3 recolector.py --etiqueta pesada --duracion 120
```

---

### Verificar el dataset

```bash
cd ~/so-proyecto && source venv/bin/activate
python3 recolector.py --resumen
```

Deberías ver ~120 muestras por clase antes de continuar.

---

## PASO 2 — Entrenamiento del modelo

```bash
cd ~/so-proyecto && source venv/bin/activate
python3 entrenar_modelo.py
```

El script genera 3 archivos en `modelo/`:

| Archivo | Descripción |
|---|---|
| `arbol.pkl` | Modelo serializado — lo usa el daemon |
| `arbol_visual.png` | Imagen del árbol para el informe |
| `matriz_confusion.png` | Matriz de confusión para el informe |

Al finalizar imprime en pantalla: exactitud, validación cruzada (5-fold), reporte por clase y prueba rápida con valores típicos. **La exactitud de prueba debe ser ≥ 85% antes de continuar.**

---

## PASO 3 — Experimento controlado

El experimento se corre en **dos tandas separadas**: primero el grupo control (sin daemon), luego el grupo de tratamiento (con daemon). Cada tanda ejecuta 10 réplicas de la misma carga de trabajo y guarda las métricas en `datos/resultados_experimento.csv`.

### Tanda A — Grupo control (sin ML)

Solo una terminal necesaria.

```bash
cd ~/so-proyecto && source venv/bin/activate
python3 experimento.py --modo sin_ml --replicas 10
```

Espera a que terminen las 10 réplicas (~7 min con pausa de 10 s entre cada una).

### Limpiar caché antes del grupo con ML

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

### Tanda B — Grupo tratamiento (con ML)

Necesitas **dos terminales**.

**Terminal 1 — arrancar el daemon:**
```bash
cd ~/so-proyecto && source venv/bin/activate
sudo python3 daemon.py
```

**Terminal 2 — correr el experimento:**
```bash
cd ~/so-proyecto && source venv/bin/activate
python3 experimento.py --modo con_ml --replicas 10
```

Cuando terminen las 10 réplicas, detén el daemon con `Ctrl+C` en la Terminal 1.

### Ver resumen rápido

```bash
python3 experimento.py --resumen
```

---

## PASO 4 — Análisis estadístico

```bash
cd ~/so-proyecto && source venv/bin/activate
python3 analisis_resultados.py
```

Genera 6 archivos en `resultados/`:

| Archivo | Descripción |
|---|---|
| `tabla_estadisticos.csv` | Media, desv. est., mín, máx, mediana por grupo |
| `grafica_tiempos.png` | Boxplot de tiempos de ejecución |
| `grafica_cpu.png` | Boxplot de CPU promedio |
| `grafica_barras_comparacion.png` | Barras con error bars + p-valor anotado |
| `grafica_evolucion_replicas.png` | Línea por réplica, ambos grupos |
| `informe_estadistico.txt` | Reporte completo con prueba t e interpretación |

El reporte también se imprime en pantalla al finalizar.

---

## Referencia rápida de todos los scripts

| Script | Qué hace | Argumentos principales |
|---|---|---|
| `recolector.py` | Captura métricas y las guarda en dataset.csv | `--etiqueta ligera\|moderada\|pesada`, `--duracion N`, `--resumen` |
| `carga_cpu.py` | Genera carga CPU-bound | `--intensidad media\|alta`, `--duracion N` |
| `carga_memoria.py` | Genera carga memory-bound | `--intensidad media\|alta`, `--duracion N` |
| `entrenar_modelo.py` | Entrena el árbol de decisión | _(sin argumentos)_ |
| `daemon.py` | Daemon ML de ajuste de prioridades | `--duracion N` _(omitir = corre indefinido)_ |
| `experimento.py` | Ejecuta las réplicas del experimento | `--modo sin_ml\|con_ml`, `--replicas N`, `--resumen` |
| `analisis_resultados.py` | Análisis estadístico y gráficas | _(sin argumentos)_ |

---

## Notas importantes

- El daemon requiere `sudo` para modificar procesos de otros usuarios. Sin sudo solo puede intervenir procesos del propio usuario.
- Los scripts de carga (`carga_cpu.py`, `carga_memoria.py`) se detienen solos al terminar la duración indicada. También se pueden interrumpir con `Ctrl+C`.
- Los logs del daemon se guardan en `logs/daemon.log`.
- `analisis_resultados.py` requiere que existan réplicas de **ambos** grupos (`sin_ml` y `con_ml`) en el CSV antes de correr.
