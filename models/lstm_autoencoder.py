"""
LSTM Autoencoder for temporal anomaly detection in network traffic.
Learns compressed representations of normal traffic patterns;
anomalies are detected via reconstruction error.
"""
import os
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Input, BatchNormalization
    )
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM Autoencoder will not function.")

logger = logging.getLogger(__name__)


class LSTMAutoencoder:
    """
    LSTM Autoencoder for network traffic anomaly detection.
    
    Architecture:
    - Encoder: 2 LSTM layers (64 -> 32 units) with dropout
    - Bottleneck: Dense latent representation (16 units)
    - Decoder: 2 LSTM layers (32 -> 64 units) with dropout
    - Output: TimeDistributed Dense (reconstruction)
    
    Detection: Reconstruction error > threshold = anomaly
    """
    
    def __init__(self, sequence_length: int = 100, n_features: int = 47,
                 latent_dim: int = 16, lstm_units: list = None,
                 dropout_rate: float = 0.2, learning_rate: float = 0.001,
                 model_name: str = "lstm_autoencoder"):
        """
        Initialize LSTM Autoencoder.
        
        Args:
            sequence_length: Number of time steps
            n_features: Number of features per time step
            latent_dim: Dimension of latent/bottleneck representation
            lstm_units: List of encoder LSTM units [default: [64, 32]]
            dropout_rate: Dropout regularization rate
            learning_rate: Adam optimizer learning rate
            model_name: Name for saved model files
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM Autoencoder")
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units or [64, 32]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_name = model_name
        
        self.model: Optional[Model] = None
        self.threshold: Optional[float] = None
        self.history = None
        
    def build_model(self) -> Model:
        """
        Build LSTM Autoencoder architecture.
        
        Returns:
            Compiled Keras Model
        """
        logger.info(f"Building LSTM Autoencoder...")
        logger.info(f"  Input: ({self.sequence_length}, {self.n_features})")
        logger.info(f"  Encoder LSTM units: {self.lstm_units}")
        logger.info(f"  Latent dimension: {self.latent_dim}")
        
        # Encoder
        inputs = Input(shape=(self.sequence_length, self.n_features), name='encoder_input')
        
        # LSTM Encoder layers
        x = inputs
        for i, units in enumerate(self.lstm_units):
            return_seq = (i < len(self.lstm_units) - 1)  # Only last layer returns last output
            x = LSTM(units, activation='tanh', return_sequences=return_seq,
                    dropout=self.dropout_rate, name=f'encoder_lstm_{i+1}')(x)
        
        # Latent/bottleneck representation
        latent = Dense(self.latent_dim, activation='relu', name='bottleneck')(x)
        
        # Decoder
        x = RepeatVector(self.sequence_length, name='repeat_vector')(latent)
        
        # Reverse the LSTM units for decoder
        decoder_units = list(reversed(self.lstm_units))
        for i, units in enumerate(decoder_units):
            return_seq = True
            x = LSTM(units, activation='tanh', return_sequences=return_seq,
                    dropout=self.dropout_rate, name=f'decoder_lstm_{i+1}')(x)
        
        # Output layer - reconstruct all features at each timestep
        outputs = TimeDistributed(
            Dense(self.n_features, activation='sigmoid'), name='decoder_output'
        )(x)
        
        # Create model
        self.model = Model(inputs, outputs, name=self.model_name)
        
        # Compile
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Print summary
        self.model.summary(print_fn=logger.info)
        
        return self.model
    
    def train(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None,
              epochs: int = 50, batch_size: int = 64,
              patience: int = 10, checkpoint_dir: str = "outputs/models/checkpoints") -> Dict:
        """
        Train autoencoder on normal traffic only.
        
        The autoencoder learns to reconstruct normal traffic patterns.
        During inference, high reconstruction error indicates anomalous traffic.
        
        Args:
            X_train: Training sequences (normal traffic only)
            X_val: Validation sequences (optional)
            epochs: Maximum training epochs
            batch_size: Training batch size
            patience: Early stopping patience
            checkpoint_dir: Directory to save model checkpoints
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        logger.info(f"Training LSTM Autoencoder...")
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
        checkpoint_path = os.path.join(checkpoint_dir, f'{self.model_name}_best.h5')
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Learning rate reduction
        lr_reduce = ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5, patience=patience // 2, min_lr=1e-6, verbose=1
        )
        callbacks.append(lr_reduce)
        
        # Train (autoencoder: input == output)
        val_data = (X_val, X_val) if X_val is not None else None
        
        self.history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Training complete. Final loss: {self.history.history['loss'][-1]:.6f}")
        
        return self.history.history
    
    def compute_threshold(self, X_normal: np.ndarray, method: str = 'percentile',
                         percentile: float = 95, k_sigma: float = 3.0) -> float:
        """
        Compute anomaly detection threshold from normal data.
        
        Args:
            X_normal: Normal traffic sequences
            method: 'percentile', 'statistical', or 'max'
            percentile: Percentile for percentile method
            k_sigma: Multiplier for statistical method (mean + k*sigma)
            
        Returns:
            Computed threshold value
        """
        logger.info(f"Computing anomaly threshold (method={method})...")
        
        # Get reconstructions
        X_pred = self.model.predict(X_normal, verbose=0)
        
        # Compute per-sample MSE
        mse = np.mean(np.power(X_normal - X_pred, 2), axis=(1, 2))
        
        if method == 'percentile':
            self.threshold = float(np.percentile(mse, percentile))
            logger.info(f"  Percentile method ({percentile}%): {self.threshold:.6f}")
            
        elif method == 'statistical':
            mean = np.mean(mse)
            std = np.std(mse)
            self.threshold = float(mean + k_sigma * std)
            logger.info(f"  Statistical method (mean + {k_sigma}*std): {self.threshold:.6f}")
            logger.info(f"  Mean={mean:.6f}, Std={std:.6f}")
            
        elif method == 'max':
            self.threshold = float(np.max(mse))
            logger.info(f"  Max method: {self.threshold:.6f}")
        
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        return self.threshold
    
    def predict_anomaly(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies based on reconstruction error.
        
        Args:
            X: Input sequences to evaluate
            
        Returns:
            (predictions, reconstruction_errors)
            predictions: 1 = anomaly, 0 = normal
            reconstruction_errors: MSE for each sample
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if self.threshold is None:
            raise ValueError("Threshold not set. Call compute_threshold() first.")
        
        # Get reconstructions
        X_pred = self.model.predict(X, verbose=0)
        
        # Compute reconstruction error
        mse = np.mean(np.power(X - X_pred, 2), axis=(1, 2))
        
        # Predict anomaly if error exceeds threshold
        predictions = (mse > self.threshold).astype(int)
        
        return predictions, mse
    
    def get_latent_representations(self, X: np.ndarray) -> np.ndarray:
        """
        Extract latent representations (bottleneck layer output).
        
        Args:
            X: Input sequences
            
        Returns:
            Latent feature vectors
        """
        if self.model is None:
            raise ValueError("Model not built")
        
        # Create encoder model (up to bottleneck)
        encoder = Model(inputs=self.model.input, 
                       outputs=self.model.get_layer('bottleneck').output)
        
        return encoder.predict(X, verbose=0)
    
    def save(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
        # Save threshold separately
        threshold_path = path.replace('.h5', '_threshold.txt').replace('.keras', '_threshold.txt')
        with open(threshold_path, 'w') as f:
            f.write(str(self.threshold))
        
        logger.info(f"Model saved to {path}")
        logger.info(f"Threshold ({self.threshold:.6f}) saved to {threshold_path}")
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = load_model(path)
        
        # Load threshold
        threshold_path = path.replace('.h5', '_threshold.txt').replace('.keras', '_threshold.txt')
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                self.threshold = float(f.read().strip())
            logger.info(f"Threshold loaded: {self.threshold:.6f}")
        
        logger.info(f"Model loaded from {path}")
    
    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Get reconstruction errors without thresholding."""
        if self.model is None:
            raise ValueError("Model not trained")
        
        X_pred = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - X_pred, 2), axis=(1, 2))
        return mse
