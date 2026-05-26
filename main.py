"""
=============================================================================
  AGENTE DETECTOR DE ENFERMEDADES EN HOJAS DE PLANTAS - INTERFAZ GRÁFICA
=============================================================================
Autor       : Proyecto Universitario - Visión Computacional
Descripción : Interfaz gráfica con Tkinter para seleccionar imágenes de hojas
              y obtener el diagnóstico del modelo SVM entrenado.
Uso         : python main.py
Requiere    : Haber ejecutado train.py antes para generar el modelo.
=============================================================================
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk   # pip install Pillow
import numpy as np
import joblib

# Importar funciones del módulo de predicción
from predict import (
    cargar_modelo,
    preprocesar_imagen,
    predecir,
    nivel_confianza
)

# ─────────────────────────────────────────────────────────────────────────────
# PALETA DE COLORES Y ESTILOS
# ─────────────────────────────────────────────────────────────────────────────
COLOR_BG         = "#1a1a2e"    # Fondo principal (azul muy oscuro)
COLOR_PANEL      = "#16213e"    # Fondo de paneles
COLOR_ACENTO     = "#0f3460"    # Acento azul
COLOR_VERDE      = "#2ecc71"    # Confianza alta
COLOR_NARANJA    = "#f39c12"    # Confianza media
COLOR_ROJO       = "#e74c3c"    # Confianza baja
COLOR_TEXTO      = "#ecf0f1"    # Texto principal
COLOR_TEXTO_SUB  = "#95a5a6"    # Texto secundario
COLOR_BTN        = "#0f3460"    # Botón normal
COLOR_BTN_HOV    = "#e94560"    # Botón hover
FUENTE_TITULO    = ("Helvetica", 20, "bold")
FUENTE_SUBTIT    = ("Helvetica", 12)
FUENTE_RESULTADO = ("Helvetica", 16, "bold")
FUENTE_NORMAL    = ("Helvetica", 11)
FUENTE_PEQUENA   = ("Helvetica", 9)

PREVIEW_SIZE     = 300          # Tamaño del preview de imagen en px


# ─────────────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL DE LA INTERFAZ
# ─────────────────────────────────────────────────────────────────────────────

class AplicacionPlantDisease(tk.Tk):
    """
    Ventana principal de la aplicación.
    Permite seleccionar una imagen, analizarla con el SVM y ver el resultado.
    """

    def __init__(self):
        super().__init__()

        # Configuración de la ventana
        self.title("Agente Detector de Enfermedades en Plantas 🌿")
        self.geometry("900x700")
        self.resizable(False, False)
        self.configure(bg=COLOR_BG)

        # Variables de estado
        self.ruta_imagen    = None
        self.modelo         = None
        self.scaler         = None
        self.label_encoder  = None
        self.clases         = None
        self.img_tk         = None   # Referencia a imagen (evita garbage collection)

        # Construir la interfaz
        self._construir_ui()

        # Cargar modelo al iniciar (en hilo separado para no bloquear la UI)
        threading.Thread(target=self._cargar_modelo_async, daemon=True).start()

    # ── CONSTRUCCIÓN DE LA INTERFAZ ──────────────────────────────────────────

    def _construir_ui(self):
        """Construye todos los widgets de la interfaz."""

        # ── Encabezado ────────────────────────────────────────────────────────
        frame_header = tk.Frame(self, bg=COLOR_ACENTO, pady=15)
        frame_header.pack(fill="x")

        tk.Label(
            frame_header,
            text="🌿 Detector de Enfermedades en Plantas",
            font=FUENTE_TITULO,
            bg=COLOR_ACENTO,
            fg=COLOR_TEXTO
        ).pack()

        tk.Label(
            frame_header,
            text="Clasificación multiclase usando Support Vector Machine (SVM)",
            font=FUENTE_SUBTIT,
            bg=COLOR_ACENTO,
            fg=COLOR_TEXTO_SUB
        ).pack()

        # ── Cuerpo principal ─────────────────────────────────────────────────
        frame_body = tk.Frame(self, bg=COLOR_BG, padx=20, pady=20)
        frame_body.pack(fill="both", expand=True)

        # Columna izquierda: imagen
        frame_izq = tk.Frame(frame_body, bg=COLOR_PANEL,
                             relief="flat", bd=0)
        frame_izq.pack(side="left", fill="both", expand=True, padx=(0, 10))

        tk.Label(
            frame_izq,
            text="📷  Imagen Analizada",
            font=("Helvetica", 12, "bold"),
            bg=COLOR_PANEL, fg=COLOR_TEXTO
        ).pack(pady=(15, 5))

        # Canvas para mostrar la imagen seleccionada
        self.canvas_img = tk.Canvas(
            frame_izq,
            width=PREVIEW_SIZE, height=PREVIEW_SIZE,
            bg="#0d1117", highlightthickness=2,
            highlightbackground=COLOR_ACENTO
        )
        self.canvas_img.pack(padx=15, pady=10)
        self._mostrar_placeholder()

        # Etiqueta de nombre de archivo
        self.lbl_archivo = tk.Label(
            frame_izq,
            text="Ningún archivo seleccionado",
            font=FUENTE_PEQUENA,
            bg=COLOR_PANEL, fg=COLOR_TEXTO_SUB,
            wraplength=PREVIEW_SIZE
        )
        self.lbl_archivo.pack(pady=(0, 10))

        # Botón seleccionar imagen
        self.btn_seleccionar = self._crear_boton(
            frame_izq,
            texto="📂  Seleccionar Imagen",
            comando=self._seleccionar_imagen
        )
        self.btn_seleccionar.pack(pady=(5, 5))

        # Botón analizar
        self.btn_analizar = self._crear_boton(
            frame_izq,
            texto="🔍  Analizar Imagen",
            comando=self._analizar_imagen,
            color_fondo=COLOR_VERDE,
            estado=tk.DISABLED
        )
        self.btn_analizar.pack(pady=(0, 15))

        # Columna derecha: resultados
        frame_der = tk.Frame(frame_body, bg=COLOR_PANEL, relief="flat", bd=0)
        frame_der.pack(side="right", fill="both", expand=True)

        tk.Label(
            frame_der,
            text="📊  Resultados del Diagnóstico",
            font=("Helvetica", 12, "bold"),
            bg=COLOR_PANEL, fg=COLOR_TEXTO
        ).pack(pady=(15, 10))

        # ── Diagnóstico principal ─────────────────────────────────────────────
        self.frame_diagnostico = tk.Frame(frame_der, bg=COLOR_ACENTO,
                                           padx=10, pady=10)
        self.frame_diagnostico.pack(fill="x", padx=15, pady=(0, 10))

        self.lbl_enfermedad = tk.Label(
            self.frame_diagnostico,
            text="─  Sin análisis  ─",
            font=FUENTE_RESULTADO,
            bg=COLOR_ACENTO, fg=COLOR_TEXTO,
            wraplength=300, justify="center"
        )
        self.lbl_enfermedad.pack()

        # Barra de confianza
        tk.Label(
            frame_der,
            text="Nivel de confianza:",
            font=("Helvetica", 10),
            bg=COLOR_PANEL, fg=COLOR_TEXTO_SUB
        ).pack(anchor="w", padx=15)

        self.frame_barra = tk.Frame(frame_der, bg=COLOR_PANEL)
        self.frame_barra.pack(fill="x", padx=15, pady=(2, 10))

        self.canvas_barra = tk.Canvas(
            self.frame_barra, height=28,
            bg="#0d1117", highlightthickness=0
        )
        self.canvas_barra.pack(fill="x")

        self.lbl_confianza = tk.Label(
            frame_der,
            text="─",
            font=("Helvetica", 14, "bold"),
            bg=COLOR_PANEL, fg=COLOR_TEXTO
        )
        self.lbl_confianza.pack()

        # ── Tabla de probabilidades ────────────────────────────────────────────
        tk.Label(
            frame_der,
            text="Probabilidades por clase:",
            font=("Helvetica", 10, "bold"),
            bg=COLOR_PANEL, fg=COLOR_TEXTO_SUB
        ).pack(anchor="w", padx=15, pady=(10, 2))

        # Frame scrollable para las probabilidades
        frame_scroll = tk.Frame(frame_der, bg=COLOR_PANEL)
        frame_scroll.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        self.canvas_probs = tk.Canvas(
            frame_scroll, bg=COLOR_PANEL,
            highlightthickness=0
        )
        scrollbar = ttk.Scrollbar(frame_scroll, orient="vertical",
                                  command=self.canvas_probs.yview)
        self.frame_probs = tk.Frame(self.canvas_probs, bg=COLOR_PANEL)

        self.frame_probs.bind(
            "<Configure>",
            lambda e: self.canvas_probs.configure(
                scrollregion=self.canvas_probs.bbox("all")
            )
        )

        self.canvas_probs.create_window((0, 0), window=self.frame_probs, anchor="nw")
        self.canvas_probs.configure(yscrollcommand=scrollbar.set)

        self.canvas_probs.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ── Estado del modelo ─────────────────────────────────────────────────
        self.lbl_estado = tk.Label(
            self,
            text="⏳ Cargando modelo...",
            font=FUENTE_PEQUENA,
            bg=COLOR_BG, fg=COLOR_TEXTO_SUB
        )
        self.lbl_estado.pack(pady=5)

    def _crear_boton(self, parent, texto, comando, color_fondo=None,
                     estado=tk.NORMAL):
        """Crea un botón estilizado con efecto hover."""
        c_bg  = color_fondo or COLOR_BTN
        btn   = tk.Button(
            parent,
            text=texto,
            command=comando,
            font=("Helvetica", 11, "bold"),
            bg=c_bg, fg=COLOR_TEXTO,
            relief="flat", bd=0,
            padx=20, pady=10,
            cursor="hand2",
            state=estado
        )

        def on_enter(e):
            btn.config(bg=COLOR_BTN_HOV)

        def on_leave(e):
            btn.config(bg=c_bg)

        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

        return btn

    def _mostrar_placeholder(self):
        """Muestra un placeholder cuando no hay imagen seleccionada."""
        self.canvas_img.delete("all")
        cx, cy = PREVIEW_SIZE // 2, PREVIEW_SIZE // 2
        self.canvas_img.create_text(
            cx, cy,
            text="🌿\n\nSelecciona una imagen\nde hoja de planta",
            fill=COLOR_TEXTO_SUB,
            font=("Helvetica", 13),
            justify="center"
        )

    # ── LÓGICA DE NEGOCIO ─────────────────────────────────────────────────────

    def _cargar_modelo_async(self):
        """Carga el modelo en un hilo secundario para no bloquear la UI."""
        try:
            self.modelo, self.scaler, self.label_encoder, self.clases = cargar_modelo()
            self.after(0, lambda: self.lbl_estado.config(
                text=f"✔ Modelo listo  |  {len(self.clases)} clases cargadas",
                fg=COLOR_VERDE
            ))
        except FileNotFoundError as e:
            self.after(0, lambda: self._mostrar_error_modelo(str(e)))

    def _mostrar_error_modelo(self, mensaje: str):
        """Muestra error si el modelo no está entrenado."""
        self.lbl_estado.config(
            text="✘ Modelo no encontrado – ejecuta train.py primero",
            fg=COLOR_ROJO
        )
        messagebox.showerror(
            "Modelo no encontrado",
            f"No se pudo cargar el modelo SVM.\n\n"
            f"Solución: ejecuta 'python train.py' para entrenar el modelo.\n\n"
            f"Detalle técnico:\n{mensaje}"
        )

    def _seleccionar_imagen(self):
        """Abre el diálogo de selección de archivo."""
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen de hoja de planta",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("Todos los archivos", "*.*")
            ]
        )

        if not ruta:
            return  # El usuario canceló

        self.ruta_imagen = ruta
        self._mostrar_imagen_preview(ruta)
        self.lbl_archivo.config(text=os.path.basename(ruta))

        # Habilitar botón de análisis solo si el modelo está listo
        if self.modelo is not None:
            self.btn_analizar.config(state=tk.NORMAL)

        # Limpiar resultados anteriores
        self._limpiar_resultados()

    def _mostrar_imagen_preview(self, ruta: str):
        """Carga y muestra la imagen en el canvas de preview."""
        try:
            img = Image.open(ruta).convert("RGB")
            img.thumbnail((PREVIEW_SIZE, PREVIEW_SIZE), Image.LANCZOS)

            # Centrar en el canvas
            self.canvas_img.delete("all")
            x = (PREVIEW_SIZE - img.width)  // 2
            y = (PREVIEW_SIZE - img.height) // 2

            self.img_tk = ImageTk.PhotoImage(img)
            self.canvas_img.create_image(x, y, anchor="nw", image=self.img_tk)

        except Exception as e:
            self.canvas_img.delete("all")
            self.canvas_img.create_text(
                PREVIEW_SIZE // 2, PREVIEW_SIZE // 2,
                text=f"Error al cargar imagen:\n{e}",
                fill=COLOR_ROJO, font=FUENTE_PEQUENA, justify="center"
            )

    def _analizar_imagen(self):
        """Ejecuta la predicción en un hilo secundario."""
        if not self.ruta_imagen or self.modelo is None:
            return

        self.btn_analizar.config(state=tk.DISABLED, text="⏳ Analizando...")
        self.lbl_estado.config(text="Analizando imagen...", fg=COLOR_TEXTO_SUB)

        threading.Thread(target=self._ejecutar_prediccion, daemon=True).start()

    def _ejecutar_prediccion(self):
        """Realiza la predicción y actualiza la UI (llamado desde hilo secundario)."""
        vector, _ = preprocesar_imagen(self.ruta_imagen)

        if vector is None:
            self.after(0, lambda: messagebox.showerror(
                "Error",
                "No se pudo procesar la imagen.\n"
                "Verifica que el archivo sea una imagen válida."
            ))
            self.after(0, self._restablecer_boton)
            return

        clase, confianza, probabilidades = predecir(
            vector, self.modelo, self.scaler, self.label_encoder
        )

        # Actualizar UI en el hilo principal
        self.after(0, lambda: self._mostrar_resultado(clase, confianza, probabilidades))

    def _mostrar_resultado(self, clase: str, confianza: float, probabilidades: dict):
        """Actualiza los widgets con los resultados de la predicción."""
        etiqueta_nivel, color = nivel_confianza(confianza)

        # Nombre de la enfermedad
        self.lbl_enfermedad.config(text=clase)
        self.frame_diagnostico.config(bg=color + "44")
        self.lbl_enfermedad.config(bg=color + "44")

        # Confianza numérica
        self.lbl_confianza.config(
            text=f"{confianza*100:.1f}%  –  {etiqueta_nivel}",
            fg=color
        )

        # Barra de confianza animada
        self._animar_barra(confianza, color)

        # Probabilidades por clase
        self._dibujar_probabilidades(probabilidades, clase)

        # Estado
        self.lbl_estado.config(
            text=f"✔ Análisis completado  |  Clase: {clase}",
            fg=COLOR_VERDE
        )
        self._restablecer_boton()

    def _animar_barra(self, confianza: float, color: str):
        """Dibuja la barra de confianza en el canvas."""
        self.canvas_barra.update()
        ancho_total = self.canvas_barra.winfo_width()
        alto        = 28

        self.canvas_barra.delete("all")
        # Fondo
        self.canvas_barra.create_rectangle(
            0, 0, ancho_total, alto, fill="#0d1117", outline=""
        )
        # Barra llena
        ancho_relleno = int(ancho_total * confianza)
        self.canvas_barra.create_rectangle(
            0, 0, ancho_relleno, alto, fill=color, outline=""
        )
        # Texto encima
        self.canvas_barra.create_text(
            ancho_total // 2, alto // 2,
            text=f"{confianza*100:.1f}%",
            fill="white", font=("Helvetica", 10, "bold")
        )

    def _dibujar_probabilidades(self, probabilidades: dict, clase_pred: str):
        """Crea barras de probabilidad para cada clase en el panel lateral."""
        # Limpiar frame anterior
        for widget in self.frame_probs.winfo_children():
            widget.destroy()

        clases_ord = sorted(probabilidades.items(), key=lambda x: x[1], reverse=True)

        for nombre, prob in clases_ord:
            es_predicha = (nombre == clase_pred)
            _, color = nivel_confianza(prob) if es_predicha else (None, "#555")
            color = color if es_predicha else "#4a6fa5"

            # Frame por fila
            fila = tk.Frame(self.frame_probs, bg=COLOR_PANEL)
            fila.pack(fill="x", pady=2)

            # Nombre de clase
            tk.Label(
                fila,
                text=f"{'▶ ' if es_predicha else '   '}{nombre}",
                font=("Helvetica", 8, "bold" if es_predicha else "normal"),
                bg=COLOR_PANEL,
                fg=COLOR_TEXTO if es_predicha else COLOR_TEXTO_SUB,
                width=30, anchor="w"
            ).pack(side="left")

            # Mini barra
            mini_canvas = tk.Canvas(fila, height=14, width=100,
                                    bg="#0d1117", highlightthickness=0)
            mini_canvas.pack(side="left", padx=4)
            mini_canvas.update()
            ancho = int(100 * prob)
            mini_canvas.create_rectangle(0, 0, ancho, 14, fill=color, outline="")

            # Porcentaje
            tk.Label(
                fila,
                text=f"{prob*100:5.1f}%",
                font=("Helvetica", 8),
                bg=COLOR_PANEL,
                fg=COLOR_TEXTO
            ).pack(side="left")

    def _limpiar_resultados(self):
        """Resetea los paneles de resultado."""
        self.lbl_enfermedad.config(text="─  Analizando...  ─")
        self.frame_diagnostico.config(bg=COLOR_ACENTO)
        self.lbl_enfermedad.config(bg=COLOR_ACENTO)
        self.lbl_confianza.config(text="─")
        self.canvas_barra.delete("all")
        for widget in self.frame_probs.winfo_children():
            widget.destroy()

    def _restablecer_boton(self):
        """Reactiva el botón de análisis."""
        self.btn_analizar.config(
            state=tk.NORMAL if self.modelo is not None else tk.DISABLED,
            text="🔍  Analizar Imagen"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Inicia la aplicación de escritorio."""
    try:
        app = AplicacionPlantDisease()
        app.mainloop()
    except ImportError as e:
        if "PIL" in str(e):
            print("\n[ERROR] Falta la librería Pillow para la interfaz gráfica.")
            print("Instálala con:  pip install Pillow\n")
        else:
            raise


if __name__ == "__main__":
    main()
