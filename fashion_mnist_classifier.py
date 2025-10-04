"""
Evaluación I Machine Learning TIEL26-101-8V-A
Red Neuronal para Clasificación de Fashion-MNIST

Este script implementa una red neuronal para clasificar imágenes de Fashion-MNIST
utilizando TensorFlow/Keras como framework de deep learning.

Autor: Chrstian Santibáñez Martínez
Fecha: 04/10/2025
Curso: TIEL26 - Machine Learning
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any, Union, List

keras = tf.keras
layers = tf.keras.layers

import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configuración para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

class FashionMNISTClassifier:
    """
    Clase principal para el clasificador de Fashion-MNIST
    Implementa una red neuronal completamente conectada
    """
    
    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.history: Optional[Any] = None
        self.class_names: List[str] = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Carga y preprocesa el dataset Fashion-MNIST
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test) datos preprocesados
        """
        print("Cargando dataset Fashion-MNIST...")
        
        # Cargar datos desde Keras datasets
        (X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
        
        print(f"Shape datos de entrenamiento: {X_train.shape}")
        print(f"Shape datos de prueba: {X_test.shape}")
        print(f"Número de clases: {len(self.class_names)}")
        
        # Normalización: escalar píxeles a rango [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Reshape: aplanar imágenes de 28x28 a vector de 784 elementos
        X_train = X_train.reshape(X_train.shape[0], 784)
        X_test = X_test.reshape(X_test.shape[0], 784)
        
        # Convertir etiquetas a categorical (one-hot encoding)
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        
        print("Preprocesamiento completado.")
        print(f"Shape final X_train: {X_train.shape}")
        print(f"Shape final y_train: {y_train.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def create_neural_network(self, input_dim: int = 784, hidden_units: Optional[List[int]] = None, num_classes: int = 10) -> Any:
        """
        Crea la arquitectura de la red neuronal
        
        Args:
            input_dim (int): Dimensión de entrada (784 para Fashion-MNIST)
            hidden_units (List[int], optional): Lista con número de neuronas en capas ocultas
            num_classes (int): Número de clases de salida
            
        Returns:
            tensorflow.keras.Model: Modelo de red neuronal
        """
        if hidden_units is None:
            hidden_units = [128, 64]
        
        print("Creando arquitectura de red neuronal...")
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(hidden_units[0], activation='relu', name='hidden_1'),
            layers.Dropout(0.2),
            layers.Dense(hidden_units[1], activation='relu', name='hidden_2'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax', name='output')
        ])
        
        # Compilar modelo
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Resumen del modelo:")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int = 20, batch_size: int = 128) -> Optional[Any]:
        """
        Entrena el modelo de red neuronal
        
        Args:
            X_train, y_train: Datos de entrenamiento
            X_test, y_test: Datos de validación
            epochs (int): Número de épocas de entrenamiento
            batch_size (int): Tamaño de lote para entrenamiento
            
        Returns:
            tensorflow.keras.callbacks.History: Historia del entrenamiento
        """
        print(f"Iniciando entrenamiento por {epochs} épocas...")
        
        # Callbacks para mejorar entrenamiento
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001
            )
        ]
        
        # Entrenar modelo
        if self.model is not None:
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
        else:
            raise ValueError("El modelo no ha sido creado. Llama a create_neural_network() primero.")
        
        print("Entrenamiento completado.")
        return self.history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evalúa el rendimiento del modelo entrenado
        
        Args:
            X_test, y_test: Datos de prueba
            
        Returns:
            dict: Métricas de evaluación
        """
        print("Evaluando modelo...")
        
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a train_model() primero.")
        
        # Predicciones
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Métricas
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Reporte de clasificación
        classification_rep = classification_report(
            y_true, y_pred_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Matriz de confusión
        conf_matrix = confusion_matrix(y_true, y_pred_classes)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix,
            'y_true': y_true,
            'y_pred': y_pred_classes
        }
        
        print(f"Precisión en datos de prueba: {test_accuracy:.4f}")
        print(f"Pérdida en datos de prueba: {test_loss:.4f}")
        
        return results
    
    def plot_training_history(self) -> None:
        """
        Visualiza la historia del entrenamiento
        """
        if self.history is None:
            print("No hay historia de entrenamiento disponible.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Precisión
        ax1.plot(self.history.history['accuracy'], label='Entrenamiento')
        ax1.plot(self.history.history['val_accuracy'], label='Validación')
        ax1.set_title('Precisión del Modelo')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Precisión')
        ax1.legend()
        ax1.grid(True)
        
        # Pérdida
        ax2.plot(self.history.history['loss'], label='Entrenamiento')
        ax2.plot(self.history.history['val_loss'], label='Validación')
        ax2.set_title('Pérdida del Modelo')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Pérdida')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, conf_matrix: np.ndarray) -> None:
        """
        Visualiza la matriz de confusión
        
        Args:
            conf_matrix: Matriz de confusión
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Matriz de Confusión - Fashion MNIST')
        plt.xlabel('Predicción')
        plt.ylabel('Verdadero')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def show_sample_predictions(self, X_test: np.ndarray, y_test: np.ndarray, results: Dict[str, Any], num_samples: int = 8) -> None:
        """
        Muestra predicciones de ejemplo
        
        Args:
            X_test: Datos de prueba
            y_test: Etiquetas verdaderas
            results: Resultados de evaluación
            num_samples: Número de muestras a mostrar
        """
        # Seleccionar muestras aleatorias
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Reconstruir imagen original
            image = X_test[idx].reshape(28, 28)
            
            # Etiquetas
            true_label = results['y_true'][idx]
            pred_label = results['y_pred'][idx]
            
            # Mostrar imagen
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'Real: {self.class_names[true_label]}\n'
                            f'Pred: {self.class_names[pred_label]}',
                            fontsize=10)
            axes[i].axis('off')
            
            # Colorear borde según si la predicción es correcta
            if true_label == pred_label:
                axes[i].add_patch(patches.Rectangle((0, 0), 27, 27, fill=False, 
                                                  edgecolor='green', lw=2))
            else:
                axes[i].add_patch(patches.Rectangle((0, 0), 27, 27, fill=False, 
                                                  edgecolor='red', lw=2))
        
        plt.tight_layout()
        plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Función principal que ejecuta todo el pipeline de entrenamiento
    """
    print("=" * 60)
    print("EVALUACIÓN I MACHINE LEARNING - TIEL26-101-8V-A")
    print("Red Neuronal para Clasificación de Fashion-MNIST")
    print("=" * 60)
    
    # Crear instancia del clasificador
    classifier = FashionMNISTClassifier()
    
    # Paso 1: Cargar y preprocesar datos
    X_train, y_train, X_test, y_test = classifier.load_and_preprocess_data()
    
    # Paso 2: Crear arquitectura de red neuronal
    model = classifier.create_neural_network()
    
    # Paso 3: Entrenar modelo
    history = classifier.train_model(X_train, y_train, X_test, y_test, epochs=20)
    
    # Paso 4: Evaluar modelo
    results = classifier.evaluate_model(X_test, y_test)
    
    # Paso 5: Visualizar resultados
    print("\n" + "=" * 40)
    print("RESULTADOS FINALES")
    print("=" * 40)
    
    # Mostrar reporte de clasificación
    print("\nReporte de Clasificación:")
    print(classification_report(results['y_true'], results['y_pred'], 
                              target_names=classifier.class_names))
    
    # Generar visualizaciones
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(results['confusion_matrix'])
    classifier.show_sample_predictions(X_test, y_test, results)
    
    # Guardar modelo
    if classifier.model is not None:
        classifier.model.save('fashion_mnist_model.h5')
        print("\nModelo guardado como 'fashion_mnist_model.h5'")
    else:
        print("\nNo se pudo guardar el modelo: modelo no entrenado.")
    
    print("\n¡Evaluación completada exitosamente!")
    return classifier, results


if __name__ == "__main__":
    # Ejecutar pipeline completo
    classifier, results = main()