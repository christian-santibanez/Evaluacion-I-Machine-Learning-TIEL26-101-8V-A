# Fashion-MNIST Neural Network Classifier

## Evaluación I Machine Learning TIEL26-101-8V-A
Una implementación completa de red neuronal para clasificación de imágenes Fashion-MNIST, desarrollada como parte de la primera evaluación del curso Machine Learning.

**PROYECTO COMPLETADO Y EJECUTADO EXITOSAMENTE**

##  Descripción del Proyecto

Este proyecto implementa una red neuronal completamente conectada para clasificar imágenes del dataset Fashion-MNIST en 10 categorías de prendas de vestir:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

##  Objetivos de Aprendizaje

- Identificar tipos de aprendizaje automático
- Reconocer pasos del entrenamiento supervisado
- Explicar el proceso de entrenamiento de redes neuronales
- Asociar técnicas de ML con problemas específicos

##  Arquitectura del Modelo

`
Entrada (784)  Dense(128, ReLU)  Dropout(0.2)  Dense(64, ReLU)  Dropout(0.2)  Dense(10, Softmax)
`

**Características técnicas:**
- Framework: TensorFlow/Keras
- Optimizador: Adam
- Función de pérdida: Categorical Crossentropy
- Regularización: Dropout (0.2)
- Activación: ReLU (capas ocultas), Softmax (salida)

##  Dataset

**Fashion-MNIST:**
- 60,000 imágenes de entrenamiento
- 10,000 imágenes de prueba
- Resolución: 28x28 píxeles (escala de grises)
- 10 clases balanceadas

##  Cómo Ejecutar el Proyecto

1. **Instalar dependencias:**
   `ash
   pip install -r requirements.txt
   `

2. **Ejecutar el clasificador:**
   `ash
   python fashion_mnist_classifier.py
   `

3. **Resultados:** El script generará automáticamente:
   - Modelo entrenado (ashion_mnist_model.h5)
   - Visualizaciones (archivos PNG)
   - Métricas de evaluación en consola

##  Resultados del Proyecto

- **Accuracy final**: 89.15%
- **Modelo**: Red neuronal completamente conectada
- **Dataset**: Fashion-MNIST (70,000 imágenes)
- **Framework**: TensorFlow/Keras

**Estado del Proyecto:**  **COMPLETADO Y EJECUTADO**

##  Estructura del Proyecto

`
Evaluación I Machine Learning TIEL26-101-8V-A/
 fashion_mnist_classifier.py    # Implementación principal
 requirements.txt               # Dependencias
 informe_tecnico.md            # Reporte técnico detallado
 README.md                     # Documentación del proyecto
 .gitignore                    # Configuración de Git
 training_history.png          # Curvas de entrenamiento [GENERADO]
 confusion_matrix.png          # Matriz de confusión [GENERADO]
 sample_predictions.png        # Predicciones de ejemplo [GENERADO]
 Figure_1.png                  # Figura adicional 1 [GENERADO]
 Figure_2.png                  # Figura adicional 2 [GENERADO]
 Figure_3.png                  # Figura adicional 3 [GENERADO]
 fashion_mnist_model.h5        # Modelo entrenado [GENERADO] (1.3MB)
`

##  Funcionalidades Implementadas

### Clase FashionMNISTClassifier

- load_and_preprocess_data(): Carga y normaliza el dataset
- create_neural_network(): Construye la arquitectura del modelo
- 	rain_model(): Entrena el modelo con callbacks optimizados
- evaluate_model(): Calcula métricas de rendimiento
- plot_training_history(): Visualiza curvas de aprendizaje
- plot_confusion_matrix(): Genera matriz de confusión
- show_sample_predictions(): Muestra predicciones de ejemplo

### Características Avanzadas

- **Early Stopping**: Previene sobreajuste
- **Learning Rate Scheduling**: Optimiza convergencia
- **Reproducibilidad**: Seeds fijos para resultados consistentes
- **Visualizaciones**: Gráficos informativos del rendimiento
- **Logging completo**: Seguimiento detallado del proceso

##  Métricas de Evaluación Detalladas

### Reporte de Clasificación Completo
`
              precision    recall  f1-score   support
T-shirt/top       0.86      0.84      0.85      1000
Trouser           0.99      0.97      0.98      1000
Pullover          0.80      0.81      0.80      1000
Dress             0.89      0.90      0.90      1000
Coat              0.80      0.81      0.81      1000
Sandal            0.97      0.97      0.97      1000
Shirt             0.72      0.72      0.72      1000
Sneaker           0.95      0.96      0.96      1000
Bag               0.97      0.98      0.97      1000
Ankle boot        0.96      0.96      0.96      1000

accuracy                           0.89     10000
macro avg         0.89      0.89      0.89     10000
weighted avg      0.89      0.89      0.89     10000
`

### Análisis de Errores por Clase
- **Trouser**: Mejor clase identificada (98% F1-score) - forma distintiva
- **Shirt**: Mayor dificultad (72% F1-score) - confusión con T-shirt/top y Pullover
- **Accessories** (Bag, Sandal, Ankle boot): Excelente rendimiento (>95% F1-score)
- **Upper garments**: Mayor variabilidad en rendimiento debido a similitudes visuales

### Visualizaciones Incluidas
1. **Training History**: Curvas de accuracy y loss durante 20 épocas
2. **Confusion Matrix**: Matriz 10x10 mostrando patrones de error
3. **Sample Predictions**: Ejemplos de clasificaciones correctas e incorrectas

##  Análisis Técnico

### Preprocesamiento
- Normalización de píxeles: [0,255]  [0,1]
- Reshape: 28x28  784 (vector plano)
- One-hot encoding para etiquetas

### Regularización
- Dropout (20%) para prevenir overfitting
- Early stopping basado en validation accuracy
- Learning rate decay adaptativo

### Optimización
- Adam optimizer con parámetros por defecto
- Batch size: 128 (balance eficiencia/memoria)
- Callbacks inteligentes para mejor convergencia

##  Conceptos de Machine Learning Demostrados

### Tipos de Aprendizaje
- **Supervisado**: Utilizado en este proyecto (datos etiquetados)
- **No Supervisado**: Clustering, dimensionality reduction
- **Por Refuerzo**: Aprendizaje basado en recompensas

### Pasos del Aprendizaje Supervisado
1. Recolección y preparación de datos
2. División train/test
3. Selección y configuración del modelo
4. Entrenamiento con optimización
5. Evaluación y validación
6. Refinamiento e iteración

### Entrenamiento de Redes Neuronales
1. **Forward Propagation**: Cálculo de predicciones
2. **Loss Calculation**: Medición del error
3. **Backpropagation**: Cálculo de gradientes
4. **Weight Update**: Optimización de parámetros

##  Documentación

- **Informe Técnico**: informe_tecnico.md - Análisis detallado del proyecto
- **Informe Breve**: informe_breve_final_word.md - Informe para entregar en Word
- **README.md**: Documentación completa del proyecto
- **Resultados**: Todos los archivos de imagen y modelo generados

**Todos los documentos incluyen los resultados reales del entrenamiento ejecutado.**

##  Extensiones Futuras

1. **Convolutional Neural Networks**: Para mejor procesamiento de imágenes
2. **Data Augmentation**: Incrementar diversidad del dataset
3. **Transfer Learning**: Usar modelos preentrenados
4. **Hyperparameter Tuning**: Optimización automática de parámetros
5. **Model Ensemble**: Combinación de múltiples modelos

##  Información Académica

- **Curso**: Machine Learning (TIEL26)
- **Sección**: TIEL26-101-8V-A
- **Instructor**: Felipe Oyarzún
- **Institución**: INACAP
- **Evaluación**: 15% de la nota final
- **Fecha de entrega**: 04 de Octubre, 2025

##  Información del Estudiante

**Estudiante**: Christian Santibáñez
**Institución**: INACAP

---

*Este proyecto demuestra la aplicación práctica de conceptos fundamentales de Machine Learning en un problema real de clasificación de imágenes.*

** PROYECTO COMPLETADO EXITOSAMENTE**
