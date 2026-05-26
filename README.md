# 🌿 Agente Detector de Enfermedades en Hojas de Plantas
## Clasificación multiclase con Support Vector Machine (SVM)

> **Proyecto universitario** | Visión Computacional & Machine Learning  
> Dataset: PlantVillage | Lenguaje: Python | Modelo: SVM (scikit-learn)

---

## 📁 Estructura del Proyecto

```
plant_disease_svm/
│
├── dataset/                     ← Imágenes organizadas por clase
│   ├── Tomato_Healthy/
│   ├── Tomato_Early_Blight/
│   ├── Tomato_Late_Blight/
│   ├── Tomato_Leaf_Mold/
│   └── Tomato_Septoria_Leaf_Spot/
│
├── model/                       ← Archivos generados tras entrenar
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   ├── classes.pkl
│   ├── confusion_matrix.png
│   └── class_distribution.png
│
├── test_images/                 ← Imágenes para probar el sistema
│
├── train.py                     ← Pipeline de entrenamiento
├── predict.py                   ← Módulo de predicción individual
├── main.py                      ← Interfaz gráfica (Tkinter)
├── setup_dataset.py             ← Descarga/preparación del dataset
├── requirements.txt             ← Dependencias
└── README.md                    ← Este archivo
```

---

## 🧠 Explicación Teórica

### ¿Qué es un SVM (Support Vector Machine)?

Un SVM (Máquina de Vectores de Soporte) es un algoritmo de aprendizaje supervisado que busca encontrar el **hiperplano óptimo** que separa las clases de datos con el **máximo margen posible**.

```
Clase A (●)   Clase B (▲)
  ●               
    ●         ▲
      ●   |  ▲
       ●  | ▲
          |▲
         margen máximo
```

#### Conceptos clave:

| Término | Definición |
|---------|-----------|
| **Hiperplano** | Frontera de decisión que separa las clases |
| **Vectores de soporte** | Puntos más cercanos al hiperplano (los más difíciles de clasificar) |
| **Margen** | Distancia entre el hiperplano y los vectores de soporte |
| **Kernel** | Función que transforma el espacio de características |
| **C (regularización)** | Controla el balance entre margen amplio y errores |

### ¿Cómo funciona el kernel RBF?

El kernel RBF (Radial Basis Function) mapea los datos a un espacio de mayor dimensión donde son linealmente separables, **sin calcular explícitamente esa transformación** (truco del kernel):

```
K(x, z) = exp(-γ · ||x - z||²)
```

Esto permite que el SVM separe clases que **no son linealmente separables** en el espacio original de píxeles.

### ¿Por qué se usa preprocesamiento?

| Paso | Razón |
|------|-------|
| **Redimensionar (128×128)** | Todos los vectores deben tener la misma dimensión |
| **Convertir a RGB** | OpenCV lee en BGR; unificamos el espacio de color |
| **Normalizar [0,1]** | Evita que valores grandes (0-255) dominen el margen del SVM |
| **Aplanar (flatten)** | El SVM trabaja con vectores 1D, no matrices 2D |
| **StandardScaler** | Media=0, desv=1 por característica – mejora convergencia |

### Clasificación multiclase (OvO)

Para **K clases**, scikit-learn entrena **K(K-1)/2** clasificadores binarios (One-vs-One) y elige la clase con más "votos". Para 5 clases = 10 clasificadores internos.

### Ventajas y Limitaciones del SVM en este contexto

| ✅ Ventajas | ⚠ Limitaciones |
|------------|----------------|
| Buen rendimiento con pocos datos | Lento en datasets muy grandes (>50k muestras) |
| No requiere arquitectura compleja | No captura jerarquías espaciales como CNN |
| Evita sobreajuste con margen óptimo | Sensible a escala (por eso usamos scaler) |
| Resultados interpretables | RAM alta en inferencia con kernel RBF |
| Robusto con clases bien separadas | Rendimiento inferior a Deep Learning con muchos datos |

---

## 🔧 Instalación

### 1. Clonar / descargar el proyecto

```bash
# Crear carpeta y entrar
mkdir plant_disease_svm && cd plant_disease_svm
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 📥 Preparación del Dataset

### Opción A – Imágenes sintéticas (prueba rápida)

```bash
python setup_dataset.py
# Seleccionar opción [1]
# Recomendar: 80–200 imágenes por clase
```

### Opción B – Dataset PlantVillage real (recomendado)

1. Crear cuenta en [kaggle.com](https://www.kaggle.com)
2. Descargar API token en: Perfil → Settings → API → Create New Token
3. Colocar `kaggle.json` en `~/.kaggle/`

```bash
pip install kaggle
python setup_dataset.py
# Seleccionar opción [2]
```

### Opción C – Descarga manual

1. Ir a: https://www.kaggle.com/datasets/emmarex/plantdisease
2. Descargar y extraer
3. Copiar carpetas deseadas en `dataset/`

**Estructura requerida:**
```
dataset/
├── Tomato_Healthy/
│   ├── imagen001.jpg
│   └── ...
├── Tomato_Early_Blight/
│   └── ...
└── ...
```

---

## 🚀 Ejecución

### Paso 1: Entrenar el modelo

```bash
python train.py
```

**Salida esperada:**
```
════════════════════════════════════════════════════════════
   AGENTE DETECTOR DE ENFERMEDADES EN HOJAS DE PLANTAS
   Modelo: Support Vector Machine (SVM)
════════════════════════════════════════════════════════════

  CARGANDO DATASET DESDE: /ruta/al/proyecto/dataset
  ✔ Tomato_Early_Blight               1000 imágenes
  ✔ Tomato_Healthy                    1591 imágenes
  ✔ Tomato_Late_Blight                 973 imágenes
  ✔ Tomato_Leaf_Mold                   952 imágenes
  ✔ Tomato_Septoria_Leaf_Spot         1771 imágenes
  Total clases   : 5
  Total imágenes : 6287

  División del dataset:
    Entrenamiento : 5029 muestras
    Prueba        :  258 muestras

  ENTRENANDO MODELO SVM  |  kernel=RBF
  ✔ Entrenamiento completado en 47.23 segundos
```

### Paso 2: Lanzar interfaz gráfica

```bash
python main.py
```

### Paso 3: Predicción por consola

```bash
python predict.py test_images/hoja.jpg
```

---

## 📊 Resultados Esperados

### Accuracy (con imágenes reales PlantVillage)

| Configuración | Accuracy esperado |
|--------------|-------------------|
| Imágenes sintéticas (demo) | 60–75% |
| PlantVillage + RBF (5 clases) | 78–88% |
| PlantVillage + Linear (5 clases) | 72–82% |
| PlantVillage + RBF (todas las clases) | 70–85% |

### Ejemplo de classification report

```
                           precision  recall  f1-score  support
     Tomato_Early_Blight       0.85    0.82      0.83      200
         Tomato_Healthy        0.92    0.95      0.93      318
      Tomato_Late_Blight        0.80    0.78      0.79      195
        Tomato_Leaf_Mold        0.84    0.81      0.82      190
Tomato_Septoria_Leaf_Spot       0.88    0.90      0.89      354

                accuracy                           0.87     1257
               macro avg       0.86    0.85      0.85     1257
            weighted avg       0.87    0.87      0.87     1257
```

### Ejemplo de predicción individual

```
═══════════════════════════════════════════════════════
  RESULTADO DE PREDICCIÓN
═══════════════════════════════════════════════════════
  Clase detectada : Tomato_Early_Blight
  Confianza       : 82.40%  [ALTA ✔]
───────────────────────────────────────────────────────
  Probabilidades por clase:
    Tomato_Early_Blight          82.4%  ████████████████████████  ◄
    Tomato_Late_Blight           10.2%  ███
    Tomato_Healthy                4.1%  █
    Tomato_Leaf_Mold              2.3%
    Tomato_Septoria_Leaf_Spot     1.0%
═══════════════════════════════════════════════════════
```

---

## 🎯 Recomendaciones para Mejorar la Precisión

### 1. Características adicionales (Feature Engineering)

En lugar de usar solo píxeles crudos, extrae descriptores más informativos:

```python
# HOG (Histogram of Oriented Gradients) – captura formas y texturas
from skimage.feature import hog
features = hog(imagen, orientations=8, pixels_per_cell=(16,16))

# Histograma de color en HSV – diferencia mejor los colores de manchas
img_hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
hist     = cv2.calcHist([img_hsv], [0,1,2], None, [8,8,8], [0,256]*3)
hist     = hist.flatten() / hist.sum()  # Normalizar

# LBP (Local Binary Patterns) – textura de la hoja
from skimage.feature import local_binary_pattern
lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
```

### 2. Aumentación de datos (Data Augmentation)

```python
# Voltear, rotar y ajustar brillo para multiplicar muestras
flipped = cv2.flip(img, 1)
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
bright  = cv2.convertScaleAbs(img, alpha=1.2, beta=20)
```

### 3. Búsqueda de hiperparámetros óptimos

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C'    : [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01],
    'kernel': ['rbf', 'linear']
}
grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Mejores parámetros:", grid_search.best_params_)
```

### 4. PCA para reducción dimensional

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Conservar 95% de varianza
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)
# Puede reducir tiempo de entrenamiento significativamente
```

### 5. Pasar a Deep Learning (siguiente nivel)

Para proyectos de mayor escala, considera:
- **Transfer Learning** con ResNet50 o EfficientNet (PyTorch / TensorFlow)
- Los embeddings de la CNN como features para el SVM

---

## 📚 Referencias

- Mohanty, S.P. et al. (2016). *Using Deep Learning for Image-Based Plant Disease Detection*. Front. Plant Sci. 7:1419
- Cortes, C. & Vapnik, V. (1995). *Support-vector networks*. Machine Learning, 20(3), 273-297.
- Dataset PlantVillage: https://www.kaggle.com/datasets/emmarex/plantdisease
- Documentación scikit-learn SVM: https://scikit-learn.org/stable/modules/svm.html

---

## 👨‍💻 Notas para el Estudiante

> **¿Por qué SVM y no una red neuronal?**  
> El SVM es ideal para proyectos universitarios porque:
> 1. No requiere GPU ni hardware especializado
> 2. Entrena en minutos (vs. horas para CNNs)
> 3. Es matemáticamente interpretable
> 4. Produce resultados competitivos con datasets medianos
> 
> Para tu proyecto de tesis o industria, considera migrar a Transfer Learning con CNNs preentrenadas.
