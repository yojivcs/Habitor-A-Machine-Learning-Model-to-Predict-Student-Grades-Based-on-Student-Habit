import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

class BiLSTMModel:
    def __init__(self, input_shape):
        """
        Initialize the Bidirectional LSTM model
        
        Args:
            input_shape (tuple): Shape of input data (features)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
    
    def build_model(self, lstm_units=64, dropout_rate=0.2):
        """
        Build a Bidirectional LSTM model for regression
        
        Args:
            lstm_units (int): Number of LSTM units
            dropout_rate (float): Dropout rate for regularization
        """
        # Reshape input for LSTM [samples, time steps, features]
        reshaped_input = (1, self.input_shape)
        
        model = Sequential()
        
        # Bidirectional LSTM layer (as mentioned in your notes)
        model.add(Bidirectional(
            LSTM(lstm_units, return_sequences=True),
            input_shape=reshaped_input
        ))
        model.add(Dropout(dropout_rate))
        
        # Second Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(lstm_units // 2)))
        model.add(Dropout(dropout_rate))
        
        # Dense output layer for regression
        model.add(Dense(1))
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )
        
        self.model = model
        print(model.summary())
        return model
    
    def reshape_data_for_lstm(self, X):
        """
        Reshape the input data for LSTM (samples, timesteps, features)
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Reshaped input for LSTM
        """
        # For this regression task with no temporal component, we'll use timestep=1
        return X.reshape(X.shape[0], 1, X.shape[1])
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """
        Train the Bidirectional LSTM model
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training target
            X_val (numpy.ndarray): Validation features
            y_val (numpy.ndarray): Validation target
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            History object with training metrics
        """
        if self.model is None:
            self.build_model()
        
        # Reshape data for LSTM
        X_train_reshaped = self.reshape_data_for_lstm(X_train)
        
        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_reshaped = self.reshape_data_for_lstm(X_val)
            validation_data = (X_val_reshaped, y_val)
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train_reshaped, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test target
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Reshape test data
        X_test_reshaped = self.reshape_data_for_lstm(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_reshaped)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Reshape input data
        X_reshaped = self.reshape_data_for_lstm(X)
        
        # Make predictions
        return self.model.predict(X_reshaped)
    
    def plot_training_history(self):
        """Plot the training history"""
        if self.history is None:
            raise ValueError("Model not trained. Call train() first.")
        
        plt.figure(figsize=(12, 4))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'])
        if 'val_loss' in self.history.history:
            plt.plot(self.history.history['val_loss'])
            plt.legend(['Train', 'Validation'])
        else:
            plt.legend(['Train'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        
        print("Training history plot saved to disk")
    
    def save_model(self, filepath):
        """Save the model to disk"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model 