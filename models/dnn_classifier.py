"""
Deep Neural Network Classifier for multi-class intrusion detection.
Integrates raw features with anomaly scores from LSTM Autoencoder and Isolation Forest.
"""
import os
import logging
import numpy as np
from typing import Optional, Tuple, Dict, List

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import (
        Dense, Dropout, BatchNormalization, Input
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. DNN Classifier will not function.")

logger = logging.getLogger(__name__)


class DNNClassifier:
    """
    Deep Neural Network Classifier for intrusion detection.
    
    Architecture (optimized for IDS):
    - Input: n_features + 2 (original features + lstm_score + if_score)
    - Hidden Layer 1: 128 neurons + BatchNorm + Dropout(0.3)
    - Hidden Layer 2: 64 neurons + BatchNorm + Dropout(0.3)
    - Hidden Layer 3: 32 neurons + BatchNorm + Dropout(0.2)
    - Output: n_classes with Softmax
    
    This classifier integrates signals from both LSTM Autoencoder
    and Isolation Forest for final decision making.
    """
    
    def __init__(self, n_features: int, n_classes: int = 2,
                 hidden_layers: List[int] = None,
                 dropout_rates: List[float] = None,
                 learning_rate: float = 0.001,
                 activation: str = 'relu',
                 model_name: str = "dnn_classifier"):
        """
        Initialize DNN Classifier.
        
        Args:
            n_features: Number of input features (including anomaly scores)
            n_classes: Number of output classes (2 for binary, 10 for UNSW-NB15)
            hidden_layers: List of hidden layer units [default: [128, 64, 32]]
            dropout_rates: Dropout rate per layer [default: [0.3, 0.3, 0.2]]
            learning_rate: Adam optimizer learning rate
            activation: Activation function for hidden layers
            model_name: Name for saved model files
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for DNN Classifier")
        
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_layers = hidden_layers or [128, 64, 32]
        self.dropout_rates = dropout_rates or [0.3, 0.3, 0.2]
        self.learning_rate = learning_rate
        self.activation = activation
        self.model_name = model_name
        
        self.model: Optional[Model] = None
        self.history = None
        
    def build_model(self) -> Model:
        """
        Build DNN Classifier architecture.
        
        Returns:
            Compiled Keras Model
        """
        logger.info(f"Building DNN Classifier...")
        logger.info(f"  Input features: {self.n_features}")
        logger.info(f"  Output classes: {self.n_classes}")
        logger.info(f"  Hidden layers: {self.hidden_layers}")
        
        # Input layer
        inputs = Input(shape=(self.n_features,), name='input')
        
        # Hidden layers
        x = inputs
        for i, (units, dropout) in enumerate(zip(self.hidden_layers, self.dropout_rates)):
            x = Dense(units, activation=self.activation, name=f'hidden_{i+1}')(x)
            x = BatchNormalization(name=f'batchnorm_{i+1}')(x)
            x = Dropout(dropout, name=f'dropout_{i+1}')(x)
        
        # Output layer
        if self.n_classes == 2:
            outputs = Dense(1, activation='sigmoid', name='output')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            outputs = Dense(self.n_classes, activation='softmax', name='output')(x)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        # Create model
        self.model = Model(inputs, outputs, name=self.model_name)
        
        # Compile
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        self.model.summary(print_fn=logger.info)
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 50, batch_size: int = 64,
              patience: int = 10,
              checkpoint_dir: str = "outputs/models/checkpoints") -> Dict:
        """
        Train DNN classifier.
        
        Args:
            X_train: Training features (flat, with anomaly scores appended)
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Maximum training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            checkpoint_dir: Directory for model checkpoints
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        logger.info(f"Training DNN Classifier...")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Epochs: {epochs}, Batch size: {batch_size}")
        
        # Callbacks
        callbacks = []
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
        
        # Model checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'{self.model_name}_best.keras')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Learning rate reduction (decay rate 0.9 every 10 epochs)
        lr_reduce = ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.9, patience=patience // 2, min_lr=1e-6, verbose=1
        )
        callbacks.append(lr_reduce)
        
        # Validation data
        val_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        validation_split = 0.2 if val_data is None else 0.0
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Training complete. Final accuracy: {self.history.history['accuracy'][-1]:.4f}")
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        probabilities = self.model.predict(X, verbose=0)
        
        if self.n_classes == 2:
            predictions = (probabilities > 0.5).astype(int).flatten()
            probs = probabilities.flatten()
        else:
            predictions = np.argmax(probabilities, axis=1)
            probs = np.max(probabilities, axis=1)
        
        return predictions, probs
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: True labels
            
        Returns:
            Dictionary of metrics
        """
        predictions, probabilities = self.predict(X_test)
        
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                     f1_score, confusion_matrix, roc_auc_score)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_test, predictions)
        
        logger.info(f"\nDNN Classifier Evaluation:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"DNN Classifier saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = load_model(path)
        logger.info(f"DNN Classifier loaded from {path}")
