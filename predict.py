"""
=============================================================================
  AGENTE DETECTOR DE ENFERMEDADES EN HOJAS DE PLANTAS - MÓDULO DE PREDICCIÓN
=============================================================================
Autor       : Proyecto Universitario - Visión Computacional
Descripción : Carga el modelo SVM entrenado y realiza predicciones sobre
              imágenes individuales de hojas de plantas.
Uso         : python predict.py ruta/imagen.jpg
              python predict.py  (selección interactiva por consola)
=============================================================================
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS (deben coincidir con los usados en train.py)
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE   = 128
MODEL_DIR  = "model"

# Mapa de colores por nivel de confianza (para la barra de confianza)
COLORES_CONFIANZA = {
    "alta"  : "#2ecc71",  # Verde  – confianza ≥ 70 %
    "media" : "#f39c12",  # Naranja – confianza 40–70 %
    "baja"  : "#e74c3c",  # Rojo   – confianza < 40 %
}


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 – CARGA DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────

def cargar_modelo():
    """
    Carga desde disco el modelo SVM, el escalador y el codificador de etiquetas.
    Lanza FileNotFoundError si falta algún archivo (el modelo no fue entrenado).
    """
    archivos_requeridos = [
        "svm_model.pkl",
        "scaler.pkl",
        "label_encoder.pkl",
        "classes.pkl"
    ]

    for archivo in archivos_requeridos:
        ruta = os.path.join(MODEL_DIR, archivo)
        if not os.path.exists(ruta):
            raise FileNotFoundError(
                f"\nArchivo no encontrado: '{ruta}'\n"
                "Primero debes entrenar el modelo ejecutando:\n"
                "  python train.py\n"
            )

    modelo        = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
    scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    clases        = joblib.load(os.path.join(MODEL_DIR, "classes.pkl"))

    print(f"\n  ✔ Modelo cargado correctamente")
    print(f"  ✔ Clases reconocidas ({len(clases)}): {', '.join(clases)}\n")

    return modelo, scaler, label_encoder, clases


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2 – PREPROCESAMIENTO DE IMAGEN INDIVIDUAL
# ─────────────────────────────────────────────────────────────────────────────

def preprocesar_imagen(ruta_imagen: str, img_size: int = IMG_SIZE):
    """
    Aplica el mismo pipeline de preprocesamiento que se usó en entrenamiento:
    BGR → RGB → resize → normalize → flatten.
    Retorna (vector_np, imagen_original_rgb) o (None, None) si hay error.
    """
    if not os.path.exists(ruta_imagen):
        print(f"\n  [ERROR] No se encontró el archivo: '{ruta_imagen}'")
        return None, None

    try:
        img_bgr = cv2.imread(ruta_imagen)
        if img_bgr is None:
            raise IOError("OpenCV no pudo leer el archivo. ¿Es una imagen válida?")

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_disp = img_rgb.copy()                              # Copia para mostrar

        img_proc = cv2.resize(img_rgb, (img_size, img_size))   # Redimensionar
        img_proc = img_proc.astype(np.float32) / 255.0         # Normalizar [0,1]
        vector   = img_proc.flatten().reshape(1, -1)           # Vector 2D para predict

        return vector, img_disp

    except Exception as e:
        print(f"\n  [ERROR] Procesando imagen: {e}")
        return None, None


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3 – PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def predecir(vector, modelo, scaler, label_encoder):
    """
    Escala el vector (con el mismo scaler del entrenamiento) y predice la clase.
    Retorna:
        clase_predicha (str)     – nombre de la enfermedad detectada
        confianza      (float)   – probabilidad de la clase predicha [0-1]
        probabilidades (dict)    – probabilidad por cada clase
    """
    vector_escalado = scaler.transform(vector)

    # Predicción de clase
    indice_pred   = modelo.predict(vector_escalado)[0]
    clase_predicha = label_encoder.inverse_transform([indice_pred])[0]

    # Probabilidades por clase (requiere probability=True en SVC)
    probs_array   = modelo.predict_proba(vector_escalado)[0]
    clases_modelo = label_encoder.classes_

    probabilidades = {
        clases_modelo[i]: float(probs_array[i])
        for i in range(len(clases_modelo))
    }

    confianza = probabilidades[clase_predicha]

    return clase_predicha, confianza, probabilidades


def nivel_confianza(confianza: float) -> tuple[str, str]:
    """
    Devuelve (etiqueta_texto, color_hex) según el porcentaje de confianza.
    """
    pct = confianza * 100
    if pct >= 70:
        return "ALTA ✔",   COLORES_CONFIANZA["alta"]
    elif pct >= 40:
        return "MEDIA ⚠",  COLORES_CONFIANZA["media"]
    else:
        return "BAJA ✘",   COLORES_CONFIANZA["baja"]


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4 – VISUALIZACIÓN DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

def mostrar_resultado(imagen_rgb, clase, confianza, probabilidades,
                      ruta_imagen: str = ""):
    """
    Muestra la imagen junto con la predicción y un gráfico de probabilidades.
    Layout: imagen a la izquierda, barras de probabilidad a la derecha.
    """
    etiqueta_nivel, color_nivel = nivel_confianza(confianza)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#f8f9fa")

    # ── Panel izquierdo: imagen analizada ────────────────────────────────────
    ax_img = axes[0]
    ax_img.imshow(imagen_rgb)
    ax_img.axis("off")
    titulo_img = os.path.basename(ruta_imagen) if ruta_imagen else "Imagen analizada"
    ax_img.set_title(titulo_img, fontsize=10, color="#555", pad=8)

    # Recuadro de resultado sobre la imagen
    resultado_texto = (
        f"Diagnóstico: {clase}\n"
        f"Confianza  : {confianza*100:.1f}%  [{etiqueta_nivel}]"
    )
    ax_img.text(
        0.5, -0.05, resultado_texto,
        transform=ax_img.transAxes,
        ha="center", va="top",
        fontsize=11, fontweight="bold",
        color="#2c3e50",
        bbox=dict(boxstyle="round,pad=0.4", fc=color_nivel + "33",
                  ec=color_nivel, lw=1.5)
    )

    # ── Panel derecho: probabilidades por clase ───────────────────────────────
    ax_bar = axes[1]
    clases_ordenadas = sorted(probabilidades.items(), key=lambda x: x[1], reverse=True)
    nombres = [c for c, _ in clases_ordenadas]
    valores  = [v * 100 for _, v in clases_ordenadas]
    colores  = [color_nivel if n == clase else "#adb5bd" for n in nombres]

    barras = ax_bar.barh(nombres, valores, color=colores, edgecolor="white",
                         height=0.6)

    # Etiquetas en cada barra
    for barra, valor in zip(barras, valores):
        ax_bar.text(
            barra.get_width() + 0.5, barra.get_y() + barra.get_height() / 2,
            f"{valor:.1f}%", va="center", fontsize=9, color="#333"
        )

    ax_bar.set_xlim(0, 110)
    ax_bar.set_xlabel("Probabilidad (%)", fontsize=10)
    ax_bar.set_title("Distribución de probabilidades por clase",
                     fontsize=11, fontweight="bold")
    ax_bar.invert_yaxis()
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.set_facecolor("#f8f9fa")

    fig.suptitle(
        "AGENTE DETECTOR DE ENFERMEDADES EN HOJAS DE PLANTAS",
        fontsize=13, fontweight="bold", color="#2c3e50", y=1.01
    )

    plt.tight_layout()
    plt.show()


def imprimir_resultado_consola(clase, confianza, probabilidades):
    """Muestra el resultado en consola de forma clara y formateada."""
    etiqueta_nivel, _ = nivel_confianza(confianza)

    print("\n" + "═"*55)
    print("  RESULTADO DE PREDICCIÓN")
    print("═"*55)
    print(f"  Clase detectada : {clase}")
    print(f"  Confianza       : {confianza*100:.2f}%  [{etiqueta_nivel}]")
    print("─"*55)
    print("  Probabilidades por clase:")

    for nombre, prob in sorted(probabilidades.items(),
                                key=lambda x: x[1], reverse=True):
        barra = "█" * int(prob * 30)
        marcador = " ◄" if nombre == clase else ""
        print(f"    {nombre:<35} {prob*100:5.1f}%  {barra}{marcador}")

    print("═"*55 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def predecir_imagen(ruta_imagen: str):
    """
    Función principal: carga el modelo, procesa la imagen y muestra el resultado.
    Puede ser llamada desde otros módulos (main.py, GUI).
    """
    # Cargar modelo
    modelo, scaler, label_encoder, clases = cargar_modelo()

    # Preprocesar imagen
    vector, imagen_rgb = preprocesar_imagen(ruta_imagen)
    if vector is None:
        return None, None, None

    # Predecir
    clase, confianza, probabilidades = predecir(vector, modelo, scaler, label_encoder)

    # Mostrar en consola
    imprimir_resultado_consola(clase, confianza, probabilidades)

    # Mostrar gráfica
    mostrar_resultado(imagen_rgb, clase, confianza, probabilidades, ruta_imagen)

    return clase, confianza, probabilidades


def main():
    """
    Punto de entrada cuando se ejecuta: python predict.py [ruta_imagen]
    Si no se pasa argumento, pide la ruta por consola.
    """
    print("\n" + "═"*55)
    print("   AGENTE DETECTOR DE ENFERMEDADES – PREDICCIÓN")
    print("═"*55)

    # Obtener ruta de imagen
    if len(sys.argv) > 1:
        ruta = sys.argv[1]
    else:
        ruta = input("\n  Ingresa la ruta de la imagen a analizar:\n  > ").strip()

    if not ruta:
        print("  [ERROR] No se ingresó ninguna ruta.")
        sys.exit(1)

    predecir_imagen(ruta)


if __name__ == "__main__":
    main()
