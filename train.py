"""
=============================================================================
  AGENTE DETECTOR DE ENFERMEDADES EN HOJAS DE PLANTAS - MÓDULO DE ENTRENAMIENTO
=============================================================================
Autor       : Proyecto Universitario - Visión Computacional
Descripción : Entrena un modelo SVM multiclase para clasificar enfermedades
              en hojas de plantas usando imágenes del dataset PlantVillage.
Dependencias: opencv-python, numpy, scikit-learn, matplotlib, joblib
=============================================================================
"""

import os
import time
import warnings
import numpy as np
import cv2
import matplotlib.pyplot as plt
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS GLOBALES
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE      = 128          # Tamaño al que se redimensionan todas las imágenes
TEST_SIZE     = 0.20         # 20 % para prueba, 80 % para entrenamiento
RANDOM_STATE  = 42           # Semilla para reproducibilidad
DATASET_PATH  = "dataset"    # Ruta a la carpeta con las imágenes
MODEL_DIR     = "model"      # Carpeta donde se guarda el modelo entrenado
EXTENSIONS    = (".jpg", ".jpeg", ".png", ".bmp")  # Formatos aceptados


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 – CARGA DEL DATASET
# ─────────────────────────────────────────────────────────────────────────────

def cargar_dataset(ruta_dataset: str, img_size: int = IMG_SIZE):
    """
    Recorre las subcarpetas de `ruta_dataset`. Cada subcarpeta es una clase.
    Retorna:
        imagenes  (np.ndarray) – arreglo de vectores de píxeles normalizados
        etiquetas (list[str])  – nombre de clase para cada imagen
    """
    imagenes  = []
    etiquetas = []
    clases_encontradas = []

    if not os.path.exists(ruta_dataset):
        raise FileNotFoundError(
            f"No se encontró la carpeta del dataset: '{ruta_dataset}'\n"
            "Crea la carpeta 'dataset/' y coloca subcarpetas con imágenes."
        )

    # Obtener subcarpetas ordenadas (cada una = una clase)
    carpetas = sorted([
        d for d in os.listdir(ruta_dataset)
        if os.path.isdir(os.path.join(ruta_dataset, d))
    ])

    if not carpetas:
        raise ValueError(
            "La carpeta 'dataset/' está vacía. "
            "Agrega subcarpetas con imágenes por clase."
        )

    print(f"\n{'═'*60}")
    print(f"  CARGANDO DATASET DESDE: {os.path.abspath(ruta_dataset)}")
    print(f"{'═'*60}")

    for nombre_clase in carpetas:
        ruta_clase = os.path.join(ruta_dataset, nombre_clase)
        archivos = [
            f for f in os.listdir(ruta_clase)
            if f.lower().endswith(EXTENSIONS)
        ]

        if not archivos:
            print(f"  [ADVERTENCIA] '{nombre_clase}' no contiene imágenes válidas.")
            continue

        clases_encontradas.append(nombre_clase)
        cargadas = 0

        for archivo in archivos:
            ruta_img = os.path.join(ruta_clase, archivo)
            img = leer_y_preprocesar(ruta_img, img_size)
            if img is not None:
                imagenes.append(img)
                etiquetas.append(nombre_clase)
                cargadas += 1

        print(f"  ✔ {nombre_clase:<35} {cargadas:>4} imágenes")

    print(f"{'─'*60}")
    print(f"  Total clases : {len(clases_encontradas)}")
    print(f"  Total imágenes : {len(imagenes)}")
    print(f"{'═'*60}\n")

    return np.array(imagenes), etiquetas, clases_encontradas


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2 – PREPROCESAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def leer_y_preprocesar(ruta_imagen: str, img_size: int = IMG_SIZE):
    """
    Lee una imagen, la convierte a RGB, la redimensiona y la normaliza.
    Retorna un vector 1-D (flatten) o None si hay error de lectura.

    Por qué normalizamos:
        Los píxeles van de 0-255. Al dividir entre 255.0 los llevamos a [0,1].
        Esto evita que los valores grandes dominen el margen del SVM y
        acelera la convergencia del optimizador.
    """
    try:
        img = cv2.imread(ruta_imagen)
        if img is None:
            raise IOError(f"No se pudo leer: {ruta_imagen}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # BGR → RGB
        img = cv2.resize(img, (img_size, img_size))          # Redimensionar
        img = img.astype(np.float32) / 255.0                 # Normalizar [0,1]
        return img.flatten()                                  # Aplanar a vector

    except Exception as e:
        print(f"  [ERROR] {ruta_imagen}: {e}")
        return None


def codificar_etiquetas(etiquetas: list):
    """
    Convierte nombres de clases (str) a enteros usando LabelEncoder.
    Retorna el encoder (para decodificar predicciones más tarde) y el array.
    """
    le = LabelEncoder()
    y  = le.fit_transform(etiquetas)
    return le, y


def escalar_caracteristicas(X_train, X_test):
    """
    Aplica StandardScaler (media=0, desv=1) solo sobre el conjunto de
    entrenamiento y transforma el de prueba con los mismos parámetros.
    Esto evita 'data leakage' (fuga de información entre conjuntos).
    """
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # Ajusta Y transforma entrenamiento
    X_test  = scaler.transform(X_test)        # Solo transforma prueba
    return scaler, X_train, X_test


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3 – ENTRENAMIENTO DEL MODELO SVM
# ─────────────────────────────────────────────────────────────────────────────

def entrenar_svm(X_train, y_train, kernel: str = "rbf"):
    """
    Entrena un SVM multiclase con estrategia One-vs-One (OvO).
    Parámetros:
        kernel : 'rbf' (Radial Basis Function) o 'linear'
        C      : Parámetro de regularización (penalización de errores)
        gamma  : Solo para RBF – controla la influencia de cada muestra
        probability=True permite obtener puntuaciones de confianza

    Por qué RBF:
        El kernel RBF transforma el espacio de características a una dimensión
        mayor donde las clases son linealmente separables, sin necesidad de
        calcular explícitamente esa transformación (truco del kernel).
    """
    print(f"\n{'═'*60}")
    print(f"  ENTRENANDO MODELO SVM  |  kernel={kernel.upper()}")
    print(f"{'═'*60}")
    print(f"  Muestras de entrenamiento : {X_train.shape[0]}")
    print(f"  Dimensión del vector      : {X_train.shape[1]}")

    modelo = SVC(
        kernel      = kernel,
        C           = 10,
        gamma       = "scale",   # gamma = 1 / (n_features * X.var())
        decision_function_shape = "ovr",
        probability = True,      # Habilita predict_proba()
        random_state = RANDOM_STATE
    )

    t0 = time.time()
    modelo.fit(X_train, y_train)
    t1 = time.time()

    print(f"  ✔ Entrenamiento completado en {t1 - t0:.2f} segundos")
    print(f"{'═'*60}\n")

    return modelo


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4 – EVALUACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def evaluar_modelo(modelo, X_test, y_test, label_encoder, clases):
    """
    Muestra métricas completas: accuracy, reporte por clase y matriz de confusión.
    """
    y_pred    = modelo.predict(X_test)
    acc       = accuracy_score(y_test, y_pred)
    reporte   = classification_report(y_test, y_pred, target_names=clases)
    cm        = confusion_matrix(y_test, y_pred)

    print(f"\n{'═'*60}")
    print(f"  RESULTADOS DE EVALUACIÓN")
    print(f"{'═'*60}")
    print(f"  Accuracy global : {acc * 100:.2f}%")
    print(f"\n{reporte}")

    return acc, cm, reporte


def graficar_matriz_confusion(cm, clases: list, guardar: bool = True):
    """Genera y guarda la gráfica de la matriz de confusión."""
    fig, ax = plt.subplots(figsize=(max(8, len(clases) * 1.5),
                                    max(6, len(clases) * 1.2)))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)

    ax.set_title("Matriz de Confusión – SVM Detector de Enfermedades",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    if guardar:
        os.makedirs(MODEL_DIR, exist_ok=True)
        ruta = os.path.join(MODEL_DIR, "confusion_matrix.png")
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"  ✔ Matriz guardada en: {ruta}")

    plt.show()


def graficar_distribucion_clases(etiquetas: list, guardar: bool = True):
    """Gráfico de barras con la cantidad de imágenes por clase."""
    clases_unicas, conteos = np.unique(etiquetas, return_counts=True)

    fig, ax = plt.subplots(figsize=(max(8, len(clases_unicas) * 1.5), 5))
    bars = ax.bar(clases_unicas, conteos, color="steelblue", edgecolor="navy")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_xlabel("Clase", fontsize=11)
    ax.set_ylabel("Número de imágenes", fontsize=11)
    ax.set_title("Distribución de imágenes por clase", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if guardar:
        os.makedirs(MODEL_DIR, exist_ok=True)
        ruta = os.path.join(MODEL_DIR, "class_distribution.png")
        plt.savefig(ruta, dpi=150, bbox_inches="tight")
        print(f"  ✔ Distribución guardada en: {ruta}")

    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 5 – GUARDADO DEL MODELO
# ─────────────────────────────────────────────────────────────────────────────

def guardar_modelo(modelo, scaler, label_encoder, clases: list):
    """
    Guarda el modelo SVM, el escalador y el codificador de etiquetas
    en archivos .pkl usando joblib (más eficiente que pickle para arrays).
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(modelo,         os.path.join(MODEL_DIR, "svm_model.pkl"))
    joblib.dump(scaler,         os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(label_encoder,  os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(clases,         os.path.join(MODEL_DIR, "classes.pkl"))

    print(f"\n{'═'*60}")
    print(f"  MODELO GUARDADO EN CARPETA: '{MODEL_DIR}/'")
    print(f"{'═'*60}")
    print(f"  • svm_model.pkl      – Modelo SVM entrenado")
    print(f"  • scaler.pkl         – Escalador StandardScaler")
    print(f"  • label_encoder.pkl  – Codificador de etiquetas")
    print(f"  • classes.pkl        – Lista de clases")
    print(f"{'═'*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*60)
    print("   AGENTE DETECTOR DE ENFERMEDADES EN HOJAS DE PLANTAS")
    print("   Modelo: Support Vector Machine (SVM)")
    print("═"*60)

    # 1. Carga del dataset
    X_raw, etiquetas, clases = cargar_dataset(DATASET_PATH)

    # 2. Visualizar distribución
    graficar_distribucion_clases(etiquetas)

    # 3. Codificación de etiquetas
    label_encoder, y = codificar_etiquetas(etiquetas)

    # 4. División train/test con estratificación
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y,
        test_size    = TEST_SIZE,
        random_state = RANDOM_STATE,
        stratify     = y     # Mantiene proporción de clases en ambos conjuntos
    )
    print(f"  División del dataset:")
    print(f"    Entrenamiento : {X_train.shape[0]} muestras")
    print(f"    Prueba        : {X_test.shape[0]} muestras\n")

    # 5. Escalado de características
    scaler, X_train, X_test = escalar_caracteristicas(X_train, X_test)

    # 6. Entrenamiento del SVM
    modelo = entrenar_svm(X_train, y_train, kernel="rbf")

    # 7. Evaluación
    acc, cm, _ = evaluar_modelo(modelo, X_test, y_test, label_encoder, clases)

    # 8. Gráficas
    graficar_matriz_confusion(cm, clases)

    # 9. Guardar modelo
    guardar_modelo(modelo, scaler, label_encoder, clases)

    print(f"  ✔ Proceso completado. Accuracy: {acc * 100:.2f}%")
    print("  Ahora puedes ejecutar: python predict.py  o  python main.py\n")


if __name__ == "__main__":
    main()
