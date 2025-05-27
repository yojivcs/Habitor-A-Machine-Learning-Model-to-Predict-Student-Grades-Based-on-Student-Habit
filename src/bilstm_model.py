import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import pickle

class BiLSTMModel:
    def __init__(self, input_shape):
        """
        Initialize the Bidirectional LSTM model
        
        Args:
            input_shape (int): Number of features in the input data
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.reshape_required = True  # LSTM requires 3D input
        
        # Create output directory if it doesn't exist
        os.makedirs('output/bilstm', exist_ok=True)
    
    def build_model(self, lstm_units=64, dropout_rate=0.2, learning_rate=0.001):
        """
        Build a Bidirectional LSTM model for regression
        
        Args:
            lstm_units (int): Number of LSTM units
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for the optimizer
            
        Returns:
            tensorflow.keras.Model: Built model
        """
        print("Building BI-LSTM model...")
        
        # Create sequential model
        model = Sequential(name="BiLSTM_Model")
        
        # Add input reshape layer for LSTM (samples, timesteps, features)
        model.add(Reshape((1, self.input_shape), input_shape=(self.input_shape,)))
        
        # First Bidirectional LSTM layer with return sequences
        model.add(Bidirectional(
            LSTM(lstm_units, return_sequences=True),
            name="Bidirectional_LSTM_1"
        ))
        model.add(Dropout(dropout_rate))
        
        # Second Bidirectional LSTM layer
        model.add(Bidirectional(
            LSTM(lstm_units // 2),
            name="Bidirectional_LSTM_2"
        ))
        model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='linear', name="Output"))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        # Print model summary
        model.summary()
        
        # Save model architecture visualization
        try:
            tf.keras.utils.plot_model(
                model,
                to_file='output/bilstm/model_architecture.png',
                show_shapes=True,
                show_layer_names=True
            )
            print("Saved model architecture to output/bilstm/model_architecture.png")
        except ImportError as e:
            print(f"Warning: Could not save model architecture visualization. {e}")
            print("To enable model visualization, install pydot and graphviz: pip install pydot graphviz")
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32):
        """
        Train the BI-LSTM model
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training target
            X_val (numpy.ndarray): Validation features
            y_val (numpy.ndarray): Validation target
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        print("Training BI-LSTM model...")
        
        if self.model is None:
            self.build_model()
        
        # Prepare callbacks
        callbacks = []
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Model checkpoint to save best model
        checkpoint = ModelCheckpoint(
            'output/bilstm/best_model.h5',
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        
        # Plot training history
        self.plot_training_history()
        
        # Save trained model
        self.model.save('output/bilstm/final_model.h5')
        print("Saved trained model to output/bilstm/final_model.h5")
        
        # Save training history
        with open('output/bilstm/training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)
        print("Saved training history to output/bilstm/training_history.pkl")
        
        return history
    
    def plot_training_history(self):
        """
        Plot the training history
        """
        if self.history is None:
            print("Model not trained yet. Call train() first.")
            return
        
        # Create figure
        plt.figure(figsize=(12, 5))
        
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
        plt.ylabel('Loss (MSE)')
        plt.grid(True)
        
        # Plot training & validation MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'])
        if 'val_mae' in self.history.history:
            plt.plot(self.history.history['val_mae'])
            plt.legend(['Train', 'Validation'])
        else:
            plt.legend(['Train'])
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('output/bilstm/training_history.png')
        plt.close()
        
        print("Saved training history plot to output/bilstm/training_history.png")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test target
            
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating BI-LSTM model...")
        
        if self.model is None:
            print("Model not trained yet. Call train() first.")
            return None
        
        # Evaluate the model
        loss, mae, mse = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate RMSE
        rmse = np.sqrt(mse)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Create evaluation metrics
        metrics = {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
        
        # Print metrics
        print("\nBI-LSTM Model Evaluation Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        
        # Visualize predictions
        self.visualize_predictions(y_test, y_pred)
        
        # Save metrics to file
        with open('output/bilstm/evaluation_metrics.txt', 'w') as f:
            f.write("=== BI-LSTM Model Evaluation Metrics ===\n\n")
            f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
            f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        
        print("Saved evaluation metrics to output/bilstm/evaluation_metrics.txt")
        
        return metrics
    
    def visualize_predictions(self, y_true, y_pred):
        """
        Visualize the predictions vs actual values
        
        Args:
            y_true (numpy.ndarray): True target values
            y_pred (numpy.ndarray): Predicted target values
        """
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('BI-LSTM: Actual vs Predicted CGPA')
        plt.xlabel('Actual CGPA')
        plt.ylabel('Predicted CGPA')
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/bilstm/predictions.png')
        plt.close()
        
        # Create error distribution plot
        errors = y_pred.flatten() - y_true
        
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20, alpha=0.7)
        plt.title('BI-LSTM: Prediction Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/bilstm/error_distribution.png')
        plt.close()
        
        print("Saved prediction visualizations to output/bilstm/")
    
    def predict(self, X):
        """
        Make predictions with the trained model
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Predicted values
        """
        if self.model is None:
            print("Model not trained yet. Call train() first.")
            return None
        
        # Make predictions
        predictions = self.model.predict(X)
        
        return predictions.flatten()


# For testing the module independently
if __name__ == "__main__":
    from data_collection import DataCollector
    from preprocessing import DataPreprocessor
    from count_vectorization import FeatureVectorizer
    import numpy as np
    
    # Load data
    collector = DataCollector("Reports and Dataset/student_habits_dataset.xlsx")
    data = collector.load_data()
    
    # Preprocess data
    preprocessor = DataPreprocessor(data)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data()
    
    # Vectorize features
    vectorizer = FeatureVectorizer(X_train, X_test)
    X_train_vectorized, X_test_vectorized = vectorizer.vectorize_and_combine()
    
    # Convert to dense arrays for LSTM
    X_train_dense = X_train_vectorized.toarray()
    X_test_dense = X_test_vectorized.toarray()
    
    # Build and train BI-LSTM model
    bilstm = BiLSTMModel(X_train_dense.shape[1])
    bilstm.build_model()
    bilstm.train(X_train_dense, y_train, X_test_dense, y_test, epochs=50)
    
    # Evaluate model
    metrics = bilstm.evaluate(X_test_dense, y_test)
    
    print("\nBI-LSTM model training and evaluation completed successfully!") 