"""
Resolución de Ecuaciones Diferenciales Ordinarias
mediante la Transformada de Laplace.

Autor: Estudiante UTEZ - Academia de Ciencias
Librerías: sympy, numpy, matplotlib, tkinter
"""

import sympy as sp
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from sympy import (
    symbols, Function, laplace_transform, inverse_laplace_transform,
    solve, Eq, exp, sin, cos, simplify, apart, Symbol, latex, pretty
)
import threading


# ============================================================
# Variables simbólicas globales
# ============================================================
t, s = symbols('t s', positive=True)
y = Function('y')


# ============================================================
# Motor de cálculo (funciones puras, sin GUI)
# ============================================================

def _extraer_coeficientes_1er(expr):
    """Extrae coeficientes b y c de: b*y'(t) + c*y(t)."""
    dy_sym, y_sym = symbols('dy_sym y_sym')
    expr_sub = expr.subs(y(t).diff(t), dy_sym).subs(y(t), y_sym)
    return expr_sub.coeff(dy_sym), expr_sub.coeff(y_sym)


def _extraer_coeficientes_2do(expr):
    """Extrae coeficientes a, b y c de: a*y''(t) + b*y'(t) + c*y(t)."""
    ddy_sym, dy_sym, y_sym = symbols('ddy_sym dy_sym y_sym')
    expr_sub = (expr
                .subs(y(t).diff(t, 2), ddy_sym)
                .subs(y(t).diff(t), dy_sym)
                .subs(y(t), y_sym))
    return expr_sub.coeff(ddy_sym), expr_sub.coeff(dy_sym), expr_sub.coeff(y_sym)


def aplicar_laplace(ecuacion_lhs, ecuacion_rhs, orden, y0, dy0=0):
    """
    Aplica la transformada de Laplace a una EDO.
    Retorna Y(s) y la ecuación transformada.
    """
    Y = Symbol('Y')

    if orden == 2:
        a, b, c = _extraer_coeficientes_2do(ecuacion_lhs)
        F_s = laplace_transform(ecuacion_rhs, t, s, noconds=True)
        eq_laplace = (a * (s**2 * Y - s * y0 - dy0) +
                      b * (s * Y - y0) + c * Y)
        eq_completa = Eq(eq_laplace, F_s)
    elif orden == 1:
        b, c = _extraer_coeficientes_1er(ecuacion_lhs)
        F_s = laplace_transform(ecuacion_rhs, t, s, noconds=True)
        eq_laplace = b * (s * Y - y0) + c * Y
        eq_completa = Eq(eq_laplace, F_s)
    else:
        raise ValueError("Solo EDOs de orden 1 o 2.")

    Y_s = simplify(solve(eq_completa, Y)[0])
    return Y_s, eq_completa


def obtener_solucion_temporal(Y_s):
    """Aplica la transformada inversa de Laplace para obtener y(t)."""
    return simplify(inverse_laplace_transform(Y_s, s, t))


def resolver_edo(ecuacion_lhs, ecuacion_rhs, y0, dy0=0):
    """
    Resuelve una EDO completa. Retorna diccionario con todos los pasos.
    """
    orden = 2 if ecuacion_lhs.has(y(t).diff(t, 2)) else 1

    # Aplicar Laplace y resolver en el dominio de s
    Y_s, eq_laplace = aplicar_laplace(ecuacion_lhs, ecuacion_rhs, orden, y0, dy0)

    # Obtener solución temporal mediante transformada inversa
    y_t = obtener_solucion_temporal(Y_s)
    Y_parcial = apart(Y_s, s)

    return {
        'orden': orden,
        'ecuacion_lhs': ecuacion_lhs,
        'ecuacion_rhs': ecuacion_rhs,
        'y0': y0,
        'dy0': dy0,
        'eq_laplace': eq_laplace,
        'Y_s': Y_s,
        'Y_parcial': Y_parcial if Y_parcial != Y_s else None,
        'y_t': y_t,
    }


def generar_figura(y_t, titulo="y(t)", t_max=10):
    """Genera una Figure de matplotlib con la gráfica de y(t)."""
    f_num = sp.lambdify(t, y_t, modules=['numpy'])
    t_vals = np.linspace(0, t_max, 500)
    y_vals = f_num(t_vals)

    fig = Figure(figsize=(6, 3.5), dpi=100)
    fig.patch.set_facecolor('#2a2a3d')
    ax = fig.add_subplot(111)

    # Estilo oscuro para la gráfica
    ax.set_facecolor('#1a1a2e')
    ax.plot(t_vals, y_vals, '#2196F3', linewidth=2.2)
    ax.set_xlabel('t', fontsize=11, color='#e2e8f0')
    ax.set_ylabel('y(t)', fontsize=11, color='#e2e8f0')
    ax.set_title(titulo, fontsize=12, fontweight='bold', color='#e2e8f0')
    ax.grid(True, alpha=0.25, color='#4a4a6a')
    ax.axhline(y=0, color='#94a3b8', linewidth=0.4)
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_color('#4a4a6a')
    fig.tight_layout()
    return fig


# ============================================================
# Casos de prueba requeridos
# Los coeficientes (a, b, c, f_str) se definen directamente
# para que _cargar_caso() no dependa de comparar strings.
# ============================================================

CASOS_PRUEBA = [
    {
        'nombre': "y' + 2y = 0",
        'lhs': lambda: y(t).diff(t) + 2 * y(t),
        'rhs': 0, 'y0': 1, 'dy0': 0, 'orden': 1,
        'a': None, 'b': '1', 'c': '2', 'f_str': '0',
        'y0_str': '1', 'dy0_str': '0',
    },
    {
        'nombre': "y'' + 3y' + 2y = 0",
        'lhs': lambda: y(t).diff(t, 2) + 3 * y(t).diff(t) + 2 * y(t),
        'rhs': 0, 'y0': 0, 'dy0': 1, 'orden': 2,
        'a': '1', 'b': '3', 'c': '2', 'f_str': '0',
        'y0_str': '0', 'dy0_str': '1',
    },
    {
        'nombre': "y' - y = e^t",
        'lhs': lambda: y(t).diff(t) - y(t),
        'rhs': exp(t), 'y0': 1, 'dy0': 0, 'orden': 1,
        'a': None, 'b': '1', 'c': '-1', 'f_str': 'exp(t)',
        'y0_str': '1', 'dy0_str': '0',
    },
    {
        'nombre': "y'' + y = sin(t)",
        'lhs': lambda: y(t).diff(t, 2) + y(t),
        'rhs': sin(t), 'y0': 0, 'dy0': 0, 'orden': 2,
        'a': '1', 'b': '0', 'c': '1', 'f_str': 'sin(t)',
        'y0_str': '0', 'dy0_str': '0',
    },
]


# ============================================================
# Interfaz gráfica con Tkinter
# ============================================================

# Paleta de colores
COLOR_BG       = '#1e1e2e'    # Fondo principal (oscuro)
COLOR_PANEL    = '#2a2a3d'    # Paneles
COLOR_ACCENT   = '#7c3aed'    # Morado acento
COLOR_ACCENT2  = '#06b6d4'    # Cyan acento
COLOR_TEXT     = '#e2e8f0'    # Texto claro
COLOR_TEXT_DIM = '#94a3b8'    # Texto secundario
COLOR_INPUT_BG = '#363650'    # Fondo de inputs
COLOR_SUCCESS  = '#22c55e'    # Verde éxito
COLOR_BORDER   = '#4a4a6a'    # Bordes


class AplicacionLaplace(tk.Tk):
    """Ventana principal de la aplicación."""

    def __init__(self):
        super().__init__()
        self.title("Transformada de Laplace - Solucionador de EDOs")
        self.geometry("1100x750")
        self.configure(bg=COLOR_BG)
        self.minsize(900, 650)

        # Referencia a la figura actual (para el botón de guardar)
        self._figura_actual = None

        # Estilos ttk personalizados
        self._configurar_estilos()

        # Layout principal: panel izquierdo (entrada) + panel derecho (resultados)
        self._crear_interfaz()

    def _configurar_estilos(self):
        """Configura los estilos de ttk para el tema oscuro."""
        estilo = ttk.Style(self)
        estilo.theme_use('clam')

        # Frames
        estilo.configure('Panel.TFrame', background=COLOR_PANEL)
        estilo.configure('Main.TFrame', background=COLOR_BG)

        # Labels
        estilo.configure('Titulo.TLabel',
                         background=COLOR_BG, foreground=COLOR_ACCENT2,
                         font=('Segoe UI', 18, 'bold'))
        estilo.configure('Subtitulo.TLabel',
                         background=COLOR_PANEL, foreground=COLOR_TEXT,
                         font=('Segoe UI', 12, 'bold'))
        estilo.configure('Campo.TLabel',
                         background=COLOR_PANEL, foreground=COLOR_TEXT_DIM,
                         font=('Segoe UI', 10))
        estilo.configure('Paso.TLabel',
                         background=COLOR_PANEL, foreground=COLOR_ACCENT2,
                         font=('Segoe UI', 10, 'bold'))

        # Botones
        estilo.configure('Resolver.TButton',
                         font=('Segoe UI', 11, 'bold'),
                         padding=(20, 10))
        estilo.map('Resolver.TButton',
                   background=[('active', '#6d28d9'), ('!active', COLOR_ACCENT)],
                   foreground=[('active', 'white'), ('!active', 'white')])

        estilo.configure('Caso.TButton',
                         font=('Segoe UI', 9),
                         padding=(8, 5))
        estilo.map('Caso.TButton',
                   background=[('active', '#155e75'), ('!active', '#164e63')],
                   foreground=[('active', 'white'), ('!active', COLOR_TEXT)])

        # Radiobuttons
        estilo.configure('Orden.TRadiobutton',
                         background=COLOR_PANEL, foreground=COLOR_TEXT,
                         font=('Segoe UI', 10))

        # Combobox
        estilo.configure('TCombobox', fieldbackground=COLOR_INPUT_BG,
                         background=COLOR_INPUT_BG, foreground=COLOR_TEXT)

    def _crear_interfaz(self):
        """Construye toda la interfaz: título, panel de entrada y panel de resultados."""
        # Título superior
        frame_titulo = ttk.Frame(self, style='Main.TFrame')
        frame_titulo.pack(fill='x', padx=20, pady=(15, 5))

        ttk.Label(frame_titulo,
                  text="Solucionador de EDOs por Transformada de Laplace",
                  style='Titulo.TLabel').pack(side='left')

        ttk.Label(frame_titulo,
                  text="UTEZ - Academia de Ciencias",
                  background=COLOR_BG, foreground=COLOR_TEXT_DIM,
                  font=('Segoe UI', 9)).pack(side='right', pady=(8, 0))

        # Contenedor principal con dos columnas
        contenedor = ttk.Frame(self, style='Main.TFrame')
        contenedor.pack(fill='both', expand=True, padx=20, pady=10)
        contenedor.columnconfigure(0, weight=2, minsize=350)
        contenedor.columnconfigure(1, weight=3, minsize=450)
        contenedor.rowconfigure(0, weight=1)

        self._crear_panel_entrada(contenedor)
        self._crear_panel_resultados(contenedor)

    # ----------------------------------------------------------
    # Panel izquierdo: entrada de datos
    # ----------------------------------------------------------
    def _crear_panel_entrada(self, padre):
        """Panel donde el usuario ingresa la ecuación y sus condiciones iniciales."""
        panel = ttk.Frame(padre, style='Panel.TFrame')
        panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10))

        # --- Sección: selector de orden ---
        ttk.Label(panel, text="Ecuación Diferencial",
                  style='Subtitulo.TLabel').pack(anchor='w', padx=15, pady=(15, 8))

        frame_orden = ttk.Frame(panel, style='Panel.TFrame')
        frame_orden.pack(fill='x', padx=15)
        ttk.Label(frame_orden, text="Orden:", style='Campo.TLabel').pack(side='left')
        self.var_orden = tk.IntVar(value=1)
        ttk.Radiobutton(frame_orden, text="Primer orden", variable=self.var_orden,
                        value=1, style='Orden.TRadiobutton',
                        command=self._actualizar_campos).pack(side='left', padx=(10, 5))
        ttk.Radiobutton(frame_orden, text="Segundo orden", variable=self.var_orden,
                        value=2, style='Orden.TRadiobutton',
                        command=self._actualizar_campos).pack(side='left', padx=5)

        # Label dinámico que muestra el formato de la ecuación
        self.lbl_formato = ttk.Label(panel, text="", style='Campo.TLabel')
        self.lbl_formato.pack(anchor='w', padx=15, pady=(10, 3))

        # --- Sección: campos de coeficientes ---
        frame_coefs = ttk.Frame(panel, style='Panel.TFrame')
        frame_coefs.pack(fill='x', padx=15, pady=5)

        # Coeficiente a (visible solo en 2do orden)
        self.frame_a = ttk.Frame(frame_coefs, style='Panel.TFrame')
        self.frame_a.pack(fill='x', pady=2)
        ttk.Label(self.frame_a, text="a (coef. y''):", style='Campo.TLabel',
                  width=14).pack(side='left')
        self.entry_a = self._crear_entry(self.frame_a)
        self.entry_a.insert(0, "1")

        # Coeficiente b
        frame_b = ttk.Frame(frame_coefs, style='Panel.TFrame')
        frame_b.pack(fill='x', pady=2)
        self.lbl_b = ttk.Label(frame_b, text="b (coef. y'):", style='Campo.TLabel',
                               width=14)
        self.lbl_b.pack(side='left')
        self.entry_b = self._crear_entry(frame_b)
        self.entry_b.insert(0, "1")

        # Coeficiente c
        frame_c = ttk.Frame(frame_coefs, style='Panel.TFrame')
        frame_c.pack(fill='x', pady=2)
        ttk.Label(frame_c, text="c (coef. y):", style='Campo.TLabel',
                  width=14).pack(side='left')
        self.entry_c = self._crear_entry(frame_c)
        self.entry_c.insert(0, "2")

        # Lado derecho f(t)
        frame_f = ttk.Frame(frame_coefs, style='Panel.TFrame')
        frame_f.pack(fill='x', pady=2)
        ttk.Label(frame_f, text="f(t) =", style='Campo.TLabel',
                  width=14).pack(side='left')
        self.entry_f = self._crear_entry(frame_f)
        self.entry_f.insert(0, "0")

        # Texto de ayuda para el campo f(t)
        ttk.Label(panel,
                  text="  Ejemplos: 0, 5, exp(t), sin(t), cos(t), t**2",
                  background=COLOR_PANEL, foreground=COLOR_TEXT_DIM,
                  font=('Segoe UI', 8, 'italic')).pack(anchor='w', padx=15)

        # Separador visual
        sep = ttk.Frame(panel, style='Main.TFrame', height=2)
        sep.pack(fill='x', padx=15, pady=10)

        # --- Sección: condiciones iniciales ---
        ttk.Label(panel, text="Condiciones Iniciales",
                  style='Subtitulo.TLabel').pack(anchor='w', padx=15, pady=(0, 8))

        frame_ci = ttk.Frame(panel, style='Panel.TFrame')
        frame_ci.pack(fill='x', padx=15)

        frame_y0 = ttk.Frame(frame_ci, style='Panel.TFrame')
        frame_y0.pack(fill='x', pady=2)
        ttk.Label(frame_y0, text="y(0) =", style='Campo.TLabel',
                  width=14).pack(side='left')
        self.entry_y0 = self._crear_entry(frame_y0)
        self.entry_y0.insert(0, "1")

        # y'(0) solo visible en 2do orden
        self.frame_dy0 = ttk.Frame(frame_ci, style='Panel.TFrame')
        self.frame_dy0.pack(fill='x', pady=2)
        ttk.Label(self.frame_dy0, text="y'(0) =", style='Campo.TLabel',
                  width=14).pack(side='left')
        self.entry_dy0 = self._crear_entry(self.frame_dy0)
        self.entry_dy0.insert(0, "0")

        # --- Botón resolver ---
        frame_btn = ttk.Frame(panel, style='Panel.TFrame')
        frame_btn.pack(fill='x', padx=15, pady=15)

        self.btn_resolver = tk.Button(
            frame_btn, text="RESOLVER",
            font=('Segoe UI', 12, 'bold'),
            bg=COLOR_ACCENT, fg='white', activebackground='#6d28d9',
            activeforeground='white', relief='flat', cursor='hand2',
            padx=20, pady=8, command=self._resolver)
        self.btn_resolver.pack(fill='x')

        # Separador visual
        sep2 = ttk.Frame(panel, style='Main.TFrame', height=2)
        sep2.pack(fill='x', padx=15, pady=5)

        # --- Sección: botones de casos de prueba ---
        ttk.Label(panel, text="Casos de Prueba",
                  style='Subtitulo.TLabel').pack(anchor='w', padx=15, pady=(5, 8))

        frame_casos = ttk.Frame(panel, style='Panel.TFrame')
        frame_casos.pack(fill='x', padx=15, pady=(0, 15))

        for i, caso in enumerate(CASOS_PRUEBA):
            btn = tk.Button(
                frame_casos,
                text=f"{i+1}. {caso['nombre']}",
                font=('Segoe UI', 9),
                bg='#164e63', fg=COLOR_TEXT,
                activebackground='#155e75', activeforeground='white',
                relief='flat', cursor='hand2', anchor='w',
                padx=10, pady=4,
                command=lambda c=caso: self._cargar_caso(c))
            btn.pack(fill='x', pady=2)

        # Configurar visibilidad inicial de campos según el orden
        self._actualizar_campos()

    def _crear_entry(self, padre):
        """Crea un Entry con estilo oscuro consistente."""
        entry = tk.Entry(padre, bg=COLOR_INPUT_BG, fg=COLOR_TEXT,
                         insertbackground=COLOR_TEXT, relief='flat',
                         font=('Consolas', 11), highlightthickness=1,
                         highlightcolor=COLOR_ACCENT, highlightbackground=COLOR_BORDER)
        entry.pack(side='left', fill='x', expand=True, padx=(5, 0), ipady=3)
        return entry

    def _actualizar_campos(self):
        """Muestra u oculta campos de a y y'(0) según el orden seleccionado."""
        orden = self.var_orden.get()
        if orden == 1:
            self.lbl_formato.config(text="Formato: b*y'(t) + c*y(t) = f(t)")
            self.frame_a.pack_forget()
            self.frame_dy0.pack_forget()
        else:
            self.lbl_formato.config(text="Formato: a*y''(t) + b*y'(t) + c*y(t) = f(t)")
            self.frame_a.pack(fill='x', pady=2, before=self.frame_a.master.winfo_children()[1])
            self.frame_dy0.pack(fill='x', pady=2)

    def _cargar_caso(self, caso):
        """
        Carga un caso de prueba en los campos de entrada.
        Lee los valores directamente del diccionario del caso
        (keys: a, b, c, f_str, y0_str, dy0_str) sin comparar strings.
        """
        # Ajustar selector de orden y visibilidad de campos
        self.var_orden.set(caso['orden'])
        self._actualizar_campos()

        # Limpiar todos los campos de entrada
        for e in [self.entry_a, self.entry_b, self.entry_c, self.entry_f,
                  self.entry_y0, self.entry_dy0]:
            e.delete(0, tk.END)

        # Rellenar coeficientes desde el diccionario
        if caso['a'] is not None:
            self.entry_a.insert(0, caso['a'])
        self.entry_b.insert(0, caso['b'])
        self.entry_c.insert(0, caso['c'])
        self.entry_f.insert(0, caso['f_str'])
        self.entry_y0.insert(0, caso['y0_str'])
        self.entry_dy0.insert(0, caso['dy0_str'])

    # ----------------------------------------------------------
    # Panel derecho: resultados y gráfica
    # ----------------------------------------------------------
    def _crear_panel_resultados(self, padre):
        """Panel donde se muestran los resultados paso a paso y la gráfica."""
        panel = ttk.Frame(padre, style='Panel.TFrame')
        panel.grid(row=0, column=1, sticky='nsew')
        panel.rowconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)

        # --- Área de texto con scroll para los pasos del procedimiento ---
        frame_texto = ttk.Frame(panel, style='Panel.TFrame')
        frame_texto.grid(row=0, column=0, sticky='nsew', padx=10, pady=(10, 5))
        frame_texto.rowconfigure(0, weight=0)
        frame_texto.rowconfigure(1, weight=1)
        frame_texto.columnconfigure(0, weight=1)

        ttk.Label(frame_texto, text="Procedimiento paso a paso",
                  style='Subtitulo.TLabel').grid(row=0, column=0, sticky='w', pady=(0, 5))

        self.texto_resultado = scrolledtext.ScrolledText(
            frame_texto, bg='#1a1a2e', fg=COLOR_TEXT,
            font=('Consolas', 10), relief='flat', wrap='word',
            insertbackground=COLOR_TEXT, highlightthickness=1,
            highlightcolor=COLOR_BORDER, highlightbackground=COLOR_BORDER,
            state='disabled')
        self.texto_resultado.grid(row=1, column=0, sticky='nsew')

        # Tags de colores para diferenciar secciones en el texto
        self.texto_resultado.tag_configure('titulo',
                                           foreground=COLOR_ACCENT2,
                                           font=('Consolas', 11, 'bold'))
        self.texto_resultado.tag_configure('paso',
                                           foreground=COLOR_SUCCESS,
                                           font=('Consolas', 10, 'bold'))
        self.texto_resultado.tag_configure('ecuacion',
                                           foreground='#f8fafc',
                                           font=('Consolas', 11))
        self.texto_resultado.tag_configure('separador',
                                           foreground=COLOR_TEXT_DIM)

        # --- Área de la gráfica con botón para exportar ---
        frame_grafica = ttk.Frame(panel, style='Panel.TFrame')
        frame_grafica.grid(row=1, column=0, sticky='nsew', padx=10, pady=(5, 10))
        frame_grafica.rowconfigure(1, weight=1)
        frame_grafica.columnconfigure(0, weight=1)

        # Encabezado de la gráfica con botón de guardar
        frame_grafica_header = ttk.Frame(frame_grafica, style='Panel.TFrame')
        frame_grafica_header.grid(row=0, column=0, sticky='ew', pady=(0, 5))

        ttk.Label(frame_grafica_header, text="Gráfica de la solución",
                  style='Subtitulo.TLabel').pack(side='left')

        self.btn_guardar = tk.Button(
            frame_grafica_header, text="Guardar gráfica",
            font=('Segoe UI', 9), bg='#164e63', fg=COLOR_TEXT,
            activebackground='#155e75', activeforeground='white',
            relief='flat', cursor='hand2', padx=8, pady=2,
            command=self._guardar_grafica, state='disabled')
        self.btn_guardar.pack(side='right')

        self.frame_canvas = ttk.Frame(frame_grafica, style='Panel.TFrame')
        self.frame_canvas.grid(row=1, column=0, sticky='nsew')
        self.canvas_widget = None

        # Mostrar mensaje de bienvenida al iniciar
        self._mostrar_mensaje_bienvenida()

    def _mostrar_mensaje_bienvenida(self):
        """Muestra instrucciones de uso en el panel de resultados."""
        self.texto_resultado.config(state='normal')
        self.texto_resultado.delete('1.0', tk.END)
        self.texto_resultado.insert(tk.END,
            "  SOLUCIONADOR DE EDOs POR TRANSFORMADA DE LAPLACE\n",
            'titulo')
        self.texto_resultado.insert(tk.END, "=" * 55 + "\n\n", 'separador')
        self.texto_resultado.insert(tk.END,
            "  Ingrese los coeficientes de su ecuación diferencial\n"
            "  en el panel izquierdo y presione RESOLVER.\n\n"
            "  También puede seleccionar uno de los 4 casos de\n"
            "  prueba predefinidos.\n\n", 'ecuacion')
        self.texto_resultado.insert(tk.END,
            "  Ecuaciones soportadas:\n", 'paso')
        self.texto_resultado.insert(tk.END,
            "    - Primer orden:  b*y' + c*y = f(t)\n"
            "    - Segundo orden: a*y'' + b*y' + c*y = f(t)\n\n", 'ecuacion')
        self.texto_resultado.insert(tk.END,
            "  Para f(t) puede usar:\n"
            "    0, 5, exp(t), sin(t), cos(t), t, t**2, etc.\n", 'ecuacion')
        self.texto_resultado.config(state='disabled')

    # ----------------------------------------------------------
    # Lógica de resolución
    # ----------------------------------------------------------
    def _resolver(self):
        """
        Lee los valores de los campos en el hilo principal (Tkinter no es
        thread-safe), luego lanza el cálculo en un hilo secundario.
        """
        # Leer todos los valores de la GUI antes de crear el hilo
        datos = {
            'orden': self.var_orden.get(),
            'b': self.entry_b.get().strip(),
            'c': self.entry_c.get().strip(),
            'f': self.entry_f.get().strip(),
            'y0': self.entry_y0.get().strip(),
            'a': self.entry_a.get().strip() if self.var_orden.get() == 2 else None,
            'dy0': self.entry_dy0.get().strip() if self.var_orden.get() == 2 else '0',
        }

        # Deshabilitar botón mientras calcula
        self.btn_resolver.config(state='disabled', text="Calculando...")
        self.update_idletasks()

        # Ejecutar cálculo en hilo separado para no congelar la GUI
        threading.Thread(target=self._resolver_hilo, args=(datos,), daemon=True).start()

    def _resolver_hilo(self, datos):
        """
        Hilo de cálculo: construye la EDO a partir de los datos leídos,
        resuelve y programa la actualización de la GUI en el hilo principal.
        """
        try:
            orden = datos['orden']

            # Parsear coeficientes con sympify (acepta enteros, decimales, fracciones)
            b_val = sp.sympify(datos['b'])
            c_val = sp.sympify(datos['c'])
            f_t = sp.sympify(datos['f'])
            y0_val = sp.sympify(datos['y0'])

            # Construir el lado izquierdo de la EDO según el orden
            if orden == 2:
                a_val = sp.sympify(datos['a'])
                dy0_val = sp.sympify(datos['dy0'])
                lhs = (a_val * y(t).diff(t, 2) +
                       b_val * y(t).diff(t) +
                       c_val * y(t))
            else:
                a_val = None
                dy0_val = 0
                lhs = b_val * y(t).diff(t) + c_val * y(t)

            # Resolver la EDO
            resultado = resolver_edo(lhs, f_t, y0_val, dy0_val)

            # Generar la figura de matplotlib
            fig = generar_figura(resultado['y_t'],
                                titulo=f"y(t) = {pretty(resultado['y_t'])}")

            # Programar actualización de la GUI desde el hilo principal
            self.after(0, lambda: self._mostrar_resultado(resultado, fig, orden,
                                                          a_val, b_val, c_val,
                                                          f_t, y0_val, dy0_val))

        except Exception as e:
            self.after(0, lambda: self._mostrar_error(str(e)))

    def _mostrar_resultado(self, res, fig, orden, a, b, c, f_t, y0, dy0):
        """Actualiza el panel de texto y la gráfica con los resultados."""
        txt = self.texto_resultado
        txt.config(state='normal')
        txt.delete('1.0', tk.END)

        # Encabezado
        txt.insert(tk.END,
            "  RESOLUCIÓN POR TRANSFORMADA DE LAPLACE\n", 'titulo')
        txt.insert(tk.END, "=" * 55 + "\n\n", 'separador')

        # Paso 1: mostrar la ecuación original con condiciones iniciales
        txt.insert(tk.END, "  PASO 1: Ecuación diferencial\n", 'paso')
        if orden == 1:
            eq_str = f"  {b}*y'(t) + {c}*y(t) = {f_t}"
            ci_str = f"  y(0) = {y0}"
        else:
            eq_str = f"  {a}*y''(t) + {b}*y'(t) + {c}*y(t) = {f_t}"
            ci_str = f"  y(0) = {y0},  y'(0) = {dy0}"
        txt.insert(tk.END, eq_str + "\n", 'ecuacion')
        txt.insert(tk.END, ci_str + "\n\n", 'ecuacion')

        # Paso 2: ecuación transformada al dominio de Laplace
        txt.insert(tk.END, "  PASO 2: Transformada de Laplace\n", 'paso')
        txt.insert(tk.END, f"  {pretty(res['eq_laplace'])}\n\n", 'ecuacion')

        # Paso 3: solución algebraica Y(s)
        txt.insert(tk.END, "  PASO 3: Solución en dominio de Laplace\n", 'paso')
        txt.insert(tk.END, f"  Y(s) = {pretty(res['Y_s'])}\n", 'ecuacion')

        # Mostrar fracciones parciales si difieren de Y(s)
        if res['Y_parcial'] is not None:
            txt.insert(tk.END, "\n  Fracciones parciales:\n", 'paso')
            txt.insert(tk.END, f"  Y(s) = {pretty(res['Y_parcial'])}\n", 'ecuacion')

        txt.insert(tk.END, "\n", '')

        # Paso 4: solución final en el dominio del tiempo
        txt.insert(tk.END, "  PASO 4: Transformada inversa\n", 'paso')
        txt.insert(tk.END, f"  y(t) = {pretty(res['y_t'])}\n\n", 'ecuacion')

        txt.insert(tk.END, "=" * 55 + "\n", 'separador')
        txt.config(state='disabled')

        # --- Embeber la gráfica usando FigureCanvasTkAgg ---
        if self.canvas_widget:
            self.canvas_widget.destroy()
            self.canvas_widget = None

        canvas = FigureCanvasTkAgg(fig, master=self.frame_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas_widget = canvas.get_tk_widget()

        # Guardar referencia a la figura para poder exportarla después
        self._figura_actual = fig

        # Habilitar botón de guardar gráfica
        self.btn_guardar.config(state='normal')

        # Rehabilitar botón resolver
        self.btn_resolver.config(state='normal', text="RESOLVER")

    def _guardar_grafica(self):
        """Abre un diálogo para guardar la gráfica actual como imagen PNG."""
        if self._figura_actual is None:
            return

        ruta = filedialog.asksaveasfilename(
            title="Guardar gráfica",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
            initialfile="solucion_edo.png")

        if ruta:
            self._figura_actual.savefig(ruta, dpi=150, facecolor='#2a2a3d',
                                        bbox_inches='tight')
            messagebox.showinfo("Guardado", f"Gráfica guardada en:\n{ruta}")

    def _mostrar_error(self, mensaje):
        """Muestra un mensaje de error y rehabilita el botón resolver."""
        self.btn_resolver.config(state='normal', text="RESOLVER")
        messagebox.showerror("Error", f"No se pudo resolver la ecuación:\n\n{mensaje}")


# ============================================================
# Punto de entrada
# ============================================================

if __name__ == "__main__":
    app = AplicacionLaplace()
    app.mainloop()
