# Técnicas de Aprendizaje Automático para Sistemas Operativos

**Curso:** Sistemas Operativos y Laboratorio — Universidad de Antioquia  
**Integrantes:** Andrea Correa Arango, Emanuel Vásquez Yepes

## Descripción
Prototipo en espacio de usuario que usa un árbol de decisión para clasificar
el nivel de carga del sistema (ligera, moderada, pesada) y ajustar dinámicamente
la prioridad de procesos mediante renice, sin modificar el kernel.

## Requisitos del sistema
- Ubuntu 22.04 LTS
- Python 3.10+
- Librerías: psutil, scikit-learn, numpy, pandas, matplotlib

## Instalación
```bash
python3 -m venv venv
source venv/bin/activate
pip install psutil scikit-learn numpy pandas matplotlib
```

## Uso
```bash
# Recolectar métricas
python3 recolector.py --etiqueta ligera --duracion 120

# Generar carga
python3 carga_cpu.py --intensidad alta --duracion 60
python3 carga_memoria.py --intensidad alta --duracion 60

# Entrenar el modelo
python3 entrenar_modelo.py
```
