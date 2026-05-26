"""
=============================================================================
  AGENTE DETECTOR DE ENFERMEDADES – DESCARGA Y CONFIGURACIÓN DEL DATASET
=============================================================================
Descripción : Script auxiliar para descargar el dataset PlantVillage y
              preparar la estructura de carpetas requerida por el proyecto.

OPCIONES:
  1. Descarga automática via Kaggle API (requiere cuenta Kaggle)
  2. Generación de dataset DEMO con imágenes sintéticas (para pruebas rápidas)

Uso: python setup_dataset.py
=============================================================================
"""

import os
import sys
import random
import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR = "dataset"

# Clases del dataset PlantVillage (subconjunto de tomate)
CLASES = [
    "Tomato_Healthy",
    "Tomato_Early_Blight",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_Leaf_Spot",
]

# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR DE DATASET DEMO (imágenes sintéticas con texturas realistas)
# ─────────────────────────────────────────────────────────────────────────────

# Características visuales aproximadas por enfermedad
PERFILES_VISUALES = {
    "Tomato_Healthy": {
        "color_base" : (40, 150, 40),    # Verde intenso
        "manchas"    : False,
        "descripcion": "Hoja verde uniforme, sin lesiones"
    },
    "Tomato_Early_Blight": {
        "color_base" : (50, 120, 40),    # Verde medio
        "manchas"    : True,
        "color_mancha": (80, 50, 20),    # Café oscuro
        "n_manchas"  : (3, 8),
        "descripcion": "Manchas concéntricas café oscuro (diana)"
    },
    "Tomato_Late_Blight": {
        "color_base" : (45, 100, 35),    # Verde apagado
        "manchas"    : True,
        "color_mancha": (20, 30, 60),    # Azul-negro acuoso
        "n_manchas"  : (2, 5),
        "descripcion": "Lesiones acuosas grandes, borde irregular"
    },
    "Tomato_Leaf_Mold": {
        "color_base" : (70, 130, 30),    # Verde-amarillo
        "manchas"    : True,
        "color_mancha": (120, 100, 20),  # Amarillo-café
        "n_manchas"  : (5, 15),
        "descripcion": "Manchas amarillas en haz, moho en envés"
    },
    "Tomato_Septoria_Leaf_Spot": {
        "color_base" : (55, 125, 45),    # Verde normal
        "manchas"    : True,
        "color_mancha": (200, 180, 150), # Crema/gris claro
        "n_manchas"  : (8, 20),
        "descripcion": "Manchas pequeñas circulares con centro claro"
    },
}


def generar_imagen_sintetica(perfil: dict, img_size: int = 128,
                              seed: int = None) -> np.ndarray:
    """
    Genera una imagen sintética de hoja con las características del perfil.
    Esto es SOLO para pruebas rápidas – usa imágenes reales para el proyecto.
    """
    rng = np.random.default_rng(seed)

    # Fondo verde base con textura de ruido
    base_r, base_g, base_b = perfil["color_base"]
    imagen = np.zeros((img_size, img_size, 3), dtype=np.uint8)

    # Textura base con ruido Perlin aproximado
    for c, val_base in enumerate([base_r, base_g, base_b]):
        ruido = rng.integers(-20, 20, (img_size, img_size))
        canal = np.clip(val_base + ruido, 0, 255).astype(np.uint8)
        imagen[:, :, c] = canal

    # Máscara de hoja ovalada
    mask = np.zeros((img_size, img_size), dtype=np.uint8)
    cv2.ellipse(mask,
                center=(img_size // 2, img_size // 2),
                axes=(img_size // 2 - 5, img_size // 2 - 10),
                angle=rng.integers(-20, 20),
                startAngle=0, endAngle=360,
                color=255, thickness=-1)

    # Aplicar máscara (fondo oscuro)
    fondo = np.full_like(imagen, 20)
    imagen = np.where(mask[:, :, np.newaxis] > 0, imagen, fondo)

    # Agregar manchas si la enfermedad las tiene
    if perfil.get("manchas"):
        n_manchas = rng.integers(*perfil["n_manchas"])
        mr, mg, mb = perfil["color_mancha"]

        for _ in range(n_manchas):
            cx = rng.integers(20, img_size - 20)
            cy = rng.integers(20, img_size - 20)
            rx = rng.integers(5, 18)
            ry = rng.integers(5, 15)

            # Solo dentro de la hoja
            if mask[cy, cx] > 0:
                cv2.ellipse(imagen,
                            center=(cx, cy),
                            axes=(rx, ry),
                            angle=rng.integers(0, 180),
                            startAngle=0, endAngle=360,
                            color=(mb, mg, mr),   # BGR
                            thickness=-1)

                # Borde oscuro alrededor de la mancha
                cv2.ellipse(imagen,
                            center=(cx, cy),
                            axes=(rx + 2, ry + 2),
                            angle=0, startAngle=0, endAngle=360,
                            color=(20, 20, 20),
                            thickness=1)

    # Suavizar ligeramente
    imagen = cv2.GaussianBlur(imagen, (3, 3), 0)
    return imagen


def crear_dataset_demo(n_por_clase: int = 80):
    """
    Crea un dataset demo con imágenes sintéticas.
    Útil para probar el pipeline sin descargar el dataset real.
    """
    print("\n" + "═" * 60)
    print("  CREANDO DATASET DEMO (imágenes sintéticas)")
    print("  ⚠ Para el proyecto final usa imágenes reales de PlantVillage")
    print("═" * 60)

    total = 0
    for nombre_clase in CLASES:
        ruta_clase = os.path.join(DATASET_DIR, nombre_clase)
        os.makedirs(ruta_clase, exist_ok=True)

        perfil = PERFILES_VISUALES[nombre_clase]

        for i in range(n_por_clase):
            img = generar_imagen_sintetica(perfil, seed=i + hash(nombre_clase) % 1000)
            ruta_img = os.path.join(ruta_clase, f"img_{i:04d}.jpg")
            cv2.imwrite(ruta_img, img)
            total += 1

        print(f"  ✔ {nombre_clase:<35} {n_por_clase} imágenes  │  {perfil['descripcion']}")

    print("─" * 60)
    print(f"  Total generado : {total} imágenes en {len(CLASES)} clases")
    print(f"  Ruta           : {os.path.abspath(DATASET_DIR)}")
    print("═" * 60)
    print("\n  Ahora ejecuta:  python train.py\n")


# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA VIA KAGGLE API
# ─────────────────────────────────────────────────────────────────────────────

def descargar_plantvillage_kaggle():
    """
    Descarga el dataset PlantVillage desde Kaggle.
    Requiere:
        1. Cuenta en kaggle.com
        2. API token en ~/.kaggle/kaggle.json
        3. pip install kaggle
    """
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("\n  [ERROR] Librería 'kaggle' no instalada.")
        print("  Instala con:  pip install kaggle")
        return False

    print("\n  Descargando PlantVillage desde Kaggle...")
    os.system(
        "kaggle datasets download -d emmarex/plantdisease "
        f"--unzip -p {DATASET_DIR}"
    )
    print("  ✔ Descarga completada.")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# MENÚ INTERACTIVO
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 60)
    print("  CONFIGURACIÓN DEL DATASET – Plant Disease SVM")
    print("═" * 60)
    print("  Selecciona una opción:\n")
    print("  [1] Crear dataset DEMO con imágenes sintéticas (rápido)")
    print("  [2] Descargar PlantVillage via Kaggle API (requiere cuenta)")
    print("  [3] Salir (ya tengo mi dataset en la carpeta 'dataset/')")
    print()

    opcion = input("  Opción [1/2/3]: ").strip()

    if opcion == "1":
        n = input("\n  ¿Cuántas imágenes por clase? (recomendado: 80-200): ").strip()
        n = int(n) if n.isdigit() else 80
        crear_dataset_demo(n_por_clase=n)

    elif opcion == "2":
        descargar_plantvillage_kaggle()

    elif opcion == "3":
        print("\n  Asegúrate de que la estructura sea:")
        print("  dataset/")
        print("  ├── Tomato_Healthy/       (imágenes .jpg/.png)")
        print("  ├── Tomato_Early_Blight/")
        print("  └── ...más clases...")
        print("\n  Luego ejecuta:  python train.py\n")

    else:
        print("  Opción no válida.")


if __name__ == "__main__":
    main()
