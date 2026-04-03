# Solucionador de EDOs por Transformada de Laplace

Aplicación en Python con interfaz gráfica (Tkinter) que resuelve ecuaciones diferenciales ordinarias de **primer y segundo orden** utilizando la **transformada de Laplace**, mostrando el procedimiento paso a paso y la gráfica de la solución.

Proyecto desarrollado para la materia de Cálculo — **UTEZ, Academia de Ciencias**.

## Requisitos

- Python 3.8 o superior
- sympy
- numpy
- matplotlib
- Pillow (para renderizado de fórmulas LaTeX en la interfaz)
- tkinter (incluido con Python en la mayoría de sistemas)

## Instalación

```bash
pip install sympy numpy matplotlib Pillow
```

En sistemas Linux, si falta tkinter:

```bash
sudo apt-get install python3-tk
```

## Uso

```bash
python3 laplace_solver.py
```

Se abrirá una ventana con:

- **Panel izquierdo:** selector de orden (1° o 2°), campos para ingresar coeficientes (a, b, c), término forzante f(t) y condiciones iniciales.
- **Panel derecho:** procedimiento paso a paso con fórmulas renderizadas en notación matemática LaTeX (fracciones, exponentes, funciones trigonométricas) junto con la gráfica de la solución.
- **Botones de casos de prueba:** cargan automáticamente los 4 casos requeridos.
- **Botón "Guardar gráfica":** exporta la gráfica como PNG, PDF o SVG.

## Casos de prueba incluidos

| # | Ecuación | Condiciones iniciales | Solución |
|---|----------|-----------------------|----------|
| 1 | y' + 2y = 0 | y(0) = 1 | y(t) = e^(-2t) |
| 2 | y'' + 3y' + 2y = 0 | y(0) = 0, y'(0) = 1 | y(t) = e^(-t) - e^(-2t) |
| 3 | y' - y = e^t | y(0) = 1 | y(t) = (t+1)e^t |
| 4 | y'' + y = sin(t) | y(0) = 0, y'(0) = 0 | y(t) = sin(t)/2 - t·cos(t)/2 |

## Estructura del proyecto

```
laplace_solver.py    # Código fuente principal (motor + GUI)
Requerimientos.pdf   # Documento con los requerimientos del proyecto
README.md            # Este archivo
```

## Autor

Estudiante UTEZ — Academia de Ciencias
