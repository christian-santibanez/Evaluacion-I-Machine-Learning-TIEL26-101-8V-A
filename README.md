# 👕 Fashion-MNIST Neural Network Classifier

## Evaluación I Machine Learning TIEL26-101-8V-A

Una implementación completa de red neuronal para clasificación de imágenes Fashion-MNIST, desarrollada como parte de la primera evaluación del curso Machine Learning.

**✅ PROYECTO COMPLETADO Y EJECUTADO EXITOSAMENTE**

---

## 📖 Descripción del Proyecto

Este proyecto implementa una red neuronal completamente conectada para clasificar imágenes del dataset **Fashion-MNIST** en 10 categorías de prendas de vestir:

* T-shirt/top
* Trouser
* Pullover
* Dress
* Coat
* Sandal
* Shirt
* Sneaker
* Bag
* Ankle boot

---

## 🎯 Objetivos de Aprendizaje

* Identificar tipos de aprendizaje automático
* Reconocer pasos del entrenamiento supervisado
* Explicar el proceso de entrenamiento de redes neuronales
* Asociar técnicas de ML con problemas específicos

---

## 🏗️ Arquitectura del Modelo

```
Entrada (784) → Dense(128, ReLU) → Dropout(0.2) 
              → Dense(64, ReLU)  → Dropout(0.2) 
              → Dense(10, Softmax)
```

**Características técnicas:**

* Framework: TensorFlow/Keras
* Optimizador: Adam
* Función de pérdida: Categorical Crossentropy
* Regularización: Dropout (0.2)
* Activación: ReLU (capas ocultas), Softmax (salida)

---

## 📊 Dataset

**Fashion-MNIST**:

* 60,000 imágenes de entrenamiento
* 10,000 imágenes de prueba
* Resolución: 28x28 píxeles (escala de grises)
* 10 clases balanceadas

---

## ⚙️ Cómo Ejecutar el Proyecto

1. **Instalar dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar el clasificador:**

   ```bash
   python fashion_mnist_classifier.py
   ```

3. **Resultados generados automáticamente:**

   * Modelo entrenado (`fashion_mnist_model.h5`)
   * Visualizaciones (`.png`)
   * Métricas en consola

---

## 📈 Resultados del Proyecto

* **Accuracy entrenamiento:** 91.08%
* **Accuracy validación/prueba:** 89.15%
* **Loss de prueba:** 0.3138
* **Modelo:** Red neuronal completamente conectada
* **Dataset:** Fashion-MNIST (70,000 imágenes)

---

## 📂 Estructura del Proyecto

```
Evaluación I Machine Learning TIEL26-101-8V-A/
├── fashion_mnist_classifier.py   # Implementación principal
├── requirements.txt              # Dependencias
├── informe_tecnico.md            # Reporte técnico detallado
├── README.md                     # Documentación del proyecto
├── .gitignore                    # Configuración de Git
│
├── training_history.png          # Curvas de entrenamiento [GENERADO]
├── confusion_matrix.png          # Matriz de confusión [GENERADO]
├── sample_predictions.png        # Predicciones de ejemplo [GENERADO]
│
├── Figure_1.png                  # Figura adicional 1 [GENERADO]
├── Figure_2.png                  # Figura adicional 2 [GENERADO]
├── Figure_3.png                  # Figura adicional 3 [GENERADO]
│
└── fashion_mnist_model.h5        # Modelo entrenado [GENERADO] (1.3MB)
```

---

## 🧩 Reporte de Clasificación Completo

| Clase       | Precisión | Recall | F1-Score | Soporte |
| ----------- | --------- | ------ | -------- | ------- |
| T-shirt/top | 0.86      | 0.84   | 0.85     | 1000    |
| Trouser     | 0.99      | 0.97   | 0.98     | 1000    |
| Pullover    | 0.80      | 0.81   | 0.80     | 1000    |
| Dress       | 0.89      | 0.90   | 0.90     | 1000    |
| Coat        | 0.80      | 0.81   | 0.81     | 1000    |
| Sandal      | 0.97      | 0.97   | 0.97     | 1000    |
| Shirt       | 0.72      | 0.72   | 0.72     | 1000    |
| Sneaker     | 0.95      | 0.96   | 0.96     | 1000    |
| Bag         | 0.97      | 0.98   | 0.97     | 1000    |
| Ankle boot  | 0.96      | 0.96   | 0.96     | 1000    |

**Resultados globales:**

* Accuracy: **0.89 (10,000 muestras)**
* Macro promedio: Precisión 0.89 | Recall 0.89 | F1-score 0.89
* Ponderado promedio: Precisión 0.89 | Recall 0.89 | F1-score 0.89

---

## 🔎 Análisis de Errores

* **Trouser:** Mejor clase identificada (98% F1-score).
* **Shirt:** Mayor dificultad (72% F1-score).
* **Accessories (Bag, Sandal, Ankle boot):** Excelente rendimiento (>95%).
* **Upper garments:** Variabilidad por similitudes visuales.

---

## 📊 Visualizaciones Incluidas

1. **Training History** → `training_history.png`
2. **Confusion Matrix** → `confusion_matrix.png`
3. **Sample Predictions** → `sample_predictions.png`

Puedes verlas directamente en el repositorio.

---

## 📘 Información Académica

* Curso: Machine Learning (TIEL26)
* Sección: TIEL26-101-8V-A
* Docente: Felipe Oyarzún
* Institución: INACAP
* Evaluación: 15% nota final
* Fecha de entrega: 04 de Octubre, 2025

---

## 👤 Información del Estudiante

**Estudiante:** Christian Santibáñez
**Institución:** INACAP

---

✍️ *Este proyecto demuestra la aplicación práctica de Machine Learning en un problema real de clasificación de imágenes.*

