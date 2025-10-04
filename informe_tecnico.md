# Informe Técnico: Red Neuronal para Clasificación de Fashion-MNIST

**Evaluación I Machine Learning TIEL26-101-8V-A**  
**Estudiante:** Christian Santibáñez Martínez 
**Fecha:** 04 de Octubre, 2025  
**Profesor:** Felipe Oyarzún  

---

## 1. Introducción

Este informe presenta la implementación y análisis de una red neuronal para la clasificación de imágenes del dataset Fashion-MNIST. El objetivo principal es demostrar la comprensión de los conceptos fundamentales del aprendizaje automático supervisado y el entrenamiento de redes neuronales mediante una aplicación práctica.

Fashion-MNIST fue seleccionado como dataset por su relevancia práctica en el reconocimiento de prendas de vestir, su estructura similar a MNIST (facilitando la comprensión) pero con mayor complejidad visual, lo que presenta un desafío más realista para la clasificación automática.

---

## 2. Tipos de Aprendizaje Automático

### 2.1 Clasificación General

El Machine Learning se divide en tres categorías principales:

**Aprendizaje Supervisado:**
- Utiliza datos etiquetados (entrada-salida conocida)
- Objetivo: aprender una función que mapee entradas a salidas
- Ejemplos: clasificación, regresión
- Nuestro proyecto utiliza este tipo

**Aprendizaje No Supervisado:**
- Trabaja con datos sin etiquetas
- Objetivo: encontrar patrones ocultos en los datos
- Ejemplos: clustering, reducción de dimensionalidad
- Aplicación: segmentación de clientes, recomendaciones

**Aprendizaje por Refuerzo:**
- Aprende a través de recompensas y castigos
- Interactúa con un entorno para maximizar recompensas
- Ejemplos: juegos, robótica, sistemas de control

### 2.2 Justificación de Elección

Para el problema de clasificación de imágenes de Fashion-MNIST, el **aprendizaje supervisado** es la elección correcta porque:
- Disponemos de etiquetas conocidas para cada imagen
- El objetivo es predecir la categoría de nuevas imágenes
- Podemos medir el rendimiento comparando predicciones con etiquetas reales

---

## 3. Pasos del Entrenamiento Supervisado

### 3.1 Proceso General

1. **Recolección de Datos:** Obtener dataset con ejemplos etiquetados
2. **Preprocesamiento:** Limpiar y preparar datos para el modelo
3. **División de Datos:** Separar en conjuntos de entrenamiento y prueba
4. **Selección de Modelo:** Elegir algoritmo apropiado
5. **Entrenamiento:** Ajustar parámetros del modelo con datos de entrenamiento
6. **Evaluación:** Medir rendimiento con datos de prueba
7. **Optimización:** Ajustar hiperparámetros para mejorar rendimiento

### 3.2 Aplicación en Nuestro Proyecto

**Dataset:** Fashion-MNIST (60,000 imágenes de entrenamiento, 10,000 de prueba)
**Preprocesamiento:** Normalización de píxeles [0,1], reshape a vectores 784D
**Modelo:** Red neuronal completamente conectada
**Entrenamiento:** Backpropagation con optimizador Adam
**Evaluación:** Accuracy, matriz de confusión, reporte de clasificación

---

## 4. Entrenamiento de Redes Neuronales

### 4.1 Pasos Específicos para Redes Neuronales

**Forward Propagation:**
- Los datos fluyen desde la entrada hacia la salida
- Cada neurona aplica función de activación a la suma ponderada
- Se genera una predicción para cada muestra

**Cálculo de Pérdida:**
- Se compara la predicción con la etiqueta real
- Se utiliza función de pérdida (categorical crossentropy)
- Se mide qué tan "equivocado" está el modelo

**Backpropagation:**
- Se calculan gradientes de la pérdida respecto a cada peso
- Los errores se propagan hacia atrás en la red
- Se identifican qué pesos necesitan ajuste

**Actualización de Pesos:**
- Se utilizan gradientes para actualizar parámetros
- Optimizador (Adam) determina magnitud del cambio
- El modelo aprende iterativamente

### 4.2 Arquitectura Implementada

```
Entrada (784 neuronas) → Dense(128, ReLU) → Dropout(0.2) 
→ Dense(64, ReLU) → Dropout(0.2) → Dense(10, Softmax)
```

**Justificación de Diseño:**
- **784 entradas:** Una por cada píxel de imagen 28x28
- **Capas ocultas (128, 64):** Suficientes para capturar patrones complejos
- **ReLU:** Función de activación que evita desvanecimiento de gradiente
- **Dropout:** Regularización para prevenir sobreajuste
- **Softmax:** Salida probabilística para clasificación multiclase

---

## 5. Implementación y Resultados

### 5.1 Pasos de Entrenamiento

**Dataset:** Fashion-MNIST cargado desde Keras
**Preprocesamiento:**
- Normalización: píxeles escalados de [0,255] a [0,1]
- Reshape: imágenes 28x28 convertidas a vectores 784D
- One-hot encoding: etiquetas convertidas a formato categórico

**Modelo:**
- Arquitectura: 3 capas densas con dropout
- Optimizador: Adam (learning rate adaptativo)
- Función de pérdida: Categorical crossentropy
- Métrica: Accuracy

**Entrenamiento:**
- Épocas: 20 (con early stopping)
- Batch size: 128
- Callbacks: Early stopping, reducción de learning rate

### 5.2 Resultados Reales

Tras ejecutar el entrenamiento, el modelo logró los siguientes resultados:

**Métricas de Entrenamiento:**
- **Accuracy de entrenamiento final:** 91.08%
- **Loss de entrenamiento final:** 0.2407
- **Accuracy de validación final:** 89.15%
- **Loss de validación final:** 0.3094
- **Learning rate final:** 1.0000e-04

**Métricas de Evaluación (Test Set):**
- **Accuracy de prueba:** 89.15%
- **Loss de prueba:** 0.3138

**Duración del Entrenamiento:**
- **Épocas completadas:** 20/20
- **Tiempo por época:** ~7-14 segundos
- **Convergencia:** Progresiva y estable

### 5.3 Análisis Detallado de Resultados

**Reporte de Clasificación Completo:**
```
              precision    recall  f1-score   support
 T-shirt/top       0.86      0.84      0.85      1000
     Trouser       0.99      0.97      0.98      1000
    Pullover       0.80      0.81      0.80      1000
       Dress       0.89      0.90      0.90      1000
        Coat       0.80      0.81      0.81      1000
      Sandal       0.97      0.97      0.97      1000
       Shirt       0.72      0.72      0.72      1000
     Sneaker       0.95      0.96      0.96      1000
         Bag       0.97      0.98      0.97      1000
  Ankle boot       0.96      0.96      0.96      1000

    accuracy                           0.89     10000
   macro avg       0.89      0.89      0.89     10000
weighted avg       0.89      0.89      0.89     10000
```

**Visualizaciones Generadas:**

![Historial de Entrenamiento](training_history.png)
*Figura 1: Curvas de aprendizaje mostrando la evolución de accuracy y loss*

![Matriz de Confusión](confusion_matrix.png)
*Figura 2: Matriz de confusión detallada por clase*

![Predicciones de Muestra](sample_predictions.png)
*Figura 3: Ejemplos de predicciones del modelo con etiquetas reales y predichas*

**Análisis por Categoría:**
- **Mejor rendimiento (>95% F1-score):** Trouser (0.98), Sneaker (0.96), Bag (0.97), Ankle boot (0.96), Sandal (0.97)
- **Rendimiento moderado (80-90% F1-score):** T-shirt/top (0.85), Dress (0.90)
- **Mayor dificultad (70-80% F1-score):** Pullover (0.80), Coat (0.81), Shirt (0.72)

**Patrón de Errores Identificado:**
Las prendas superiores (T-shirt/top, Pullover, Coat, Shirt) presentan mayor confusión entre sí debido a similitudes visuales, especialmente cuando las diferencias son sutiles en imágenes de 28x28 píxeles.### 5.3 Métricas de Evaluación

**Accuracy:** Porcentaje de predicciones correctas
**Matriz de Confusión:** Visualiza errores por clase
**Reporte de Clasificación:** Precision, recall, F1-score por clase
**Curvas de Aprendizaje:** Evolución de accuracy y loss durante entrenamiento

---

## 6. Análisis y Conclusiones

### 6.1 Evaluación del Rendimiento Alcanzado

**Resultados Exitosos:**
- **Accuracy superior al objetivo:** 89.15% supera expectativas iniciales
- **Convergencia estable:** Sin signos de overfitting severo
- **Métricas equilibradas:** Precision y recall balanceados en la mayoría de clases
- **Entrenamiento eficiente:** Convergencia en 20 épocas con callbacks apropiados

**Clases con Mejor Rendimiento:**
- Trouser, Sneaker, Bag, Ankle boot, Sandal (>95% F1-score)
- Estas clases tienen características visuales más distintivas

**Áreas de Mejora Identificadas:**
- Shirt presenta el mayor desafío (72% F1-score)
- Confusión entre prendas superiores (Pullover, Coat, T-shirt/top)
- Margen de mejora en diferenciación de texturas y cortes### 6.2 Limitaciones Identificadas

- **Pérdida de información espacial:** Al aplanar imágenes se pierde estructura 2D
- **Capacidad limitada:** Para patrones visuales complejos, CNN sería superior
- **Sensibilidad a ruido:** Sin convoluciones, más susceptible a variaciones

### 6.3 Posibles Mejoras

- **Arquitectura:** Implementar CNN para mejor procesamiento de imágenes
- **Regularización:** Experimentar con diferentes técnicas (batch norm, weight decay)
- **Data augmentation:** Incrementar diversidad del dataset
- **Ensemble methods:** Combinar múltiples modelos

### 6.4 Aplicaciones Prácticas

Este tipo de modelo tiene aplicaciones en:
- **E-commerce:** Clasificación automática de productos
- **Inventario:** Organización automática de almacenes
- **Recomendaciones:** Sistemas basados en tipo de prenda
- **Control de calidad:** Detección de defectos en manufactura textil

---

## 7. Conclusiones Finales

La implementación exitosa de esta red neuronal para Fashion-MNIST demuestra la comprensión de:

1. **Tipos de aprendizaje automático** y selección apropiada para el problema
2. **Pasos del aprendizaje supervisado** desde datos hasta evaluación
3. **Proceso de entrenamiento de redes neuronales** incluyendo forward/backward propagation
4. **Asociación de técnicas ML** con problemas específicos del mundo real

El proyecto ilustra cómo el aprendizaje supervisado puede resolver problemas prácticos de clasificación, estableciendo una base sólida para futuras implementaciones más complejas en deep learning.

La metodología seguida es escalable y aplicable a diversos dominios, demostrando el valor del machine learning en automatización de tareas de reconocimiento visual.

---

**Repositorio GitHub:** [https://github.com/christian-santibanez/Evaluacion-I-Machine-Learning-TIEL26-101-8V-A]  
**Archivos del proyecto:**
- `fashion_mnist_classifier.py` - Implementación principal
- `requirements.txt` - Dependencias
- `README.md` - Documentación adicional

