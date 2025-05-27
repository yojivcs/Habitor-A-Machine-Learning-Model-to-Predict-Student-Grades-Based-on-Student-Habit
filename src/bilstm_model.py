import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import pandas as pd

class BiLSTMModel:
    def __init__(self, input_shape, is_classifier=False, num_classes=None):
        """
        Initialize the Bidirectional LSTM model
        
        Args:
            input_shape (int): Number of features in the input data
            is_classifier (bool): Whether the model is a classifier
            num_classes (int): Number of classes for classification (required if is_classifier is True)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.reshape_required = True  # LSTM requires 3D input
        self.is_classifier = is_classifier
        self.num_classes = num_classes
        
        # Create output directory if it doesn't exist
        os.makedirs('output/bilstm', exist_ok=True)
    
    def build_model(self, lstm_units=64, dropout_rate=0.2, learning_rate=0.001):
        """
        Build a Bidirectional LSTM model for regression or classification
        
        Args:
            lstm_units (int): Number of LSTM units
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for the optimizer
            
        Returns:
            tensorflow.keras.Model: Built model
        """
        print("Building BI-LSTM model...")
        print(f"Model type: {'Classification' if self.is_classifier else 'Regression'}")
        
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
        
        # Output layer - different for classification vs regression
        if self.is_classifier:
            if self.num_classes is None:
                raise ValueError("num_classes must be specified for classification")
                
            if self.num_classes == 2:
                # Binary classification
                model.add(Dense(1, activation='sigmoid', name="Output"))
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                # Multi-class classification
                model.add(Dense(self.num_classes, activation='softmax', name="Output"))
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            # Regression
            model.add(Dense(1, activation='linear', name="Output"))
            loss = 'mean_squared_error'
            metrics = ['mae', 'mse']
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
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
        
        if self.is_classifier:
            # Plot training & validation accuracy
            plt.subplot(1, 2, 1)
            plt.plot(self.history.history['accuracy'])
            if 'val_accuracy' in self.history.history:
                plt.plot(self.history.history['val_accuracy'])
                plt.legend(['Train', 'Validation'])
            else:
                plt.legend(['Train'])
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
            
            # Plot training & validation loss
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['loss'])
            if 'val_loss' in self.history.history:
                plt.plot(self.history.history['val_loss'])
                plt.legend(['Train', 'Validation'])
            else:
                plt.legend(['Train'])
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
        else:
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
        
        # For classification model
        if self.is_classifier:
            # Get raw predictions
            y_pred_raw = self.model.predict(X_test)
            
            # Convert to class predictions
            if self.num_classes == 2:
                # Binary classification
                y_pred_classes = (y_pred_raw > 0.5).astype(int).flatten()
            else:
                # Multi-class classification
                y_pred_classes = np.argmax(y_pred_raw, axis=1)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred_classes)
            
            # Generate classification report
            report = classification_report(y_test, y_pred_classes, output_dict=True)
            
            # Generate confusion matrix
            cm = confusion_matrix(y_test, y_pred_classes)
            
            # Visualize confusion matrix
            self.visualize_confusion_matrix(cm)
            
            # Create metrics dictionary
            metrics = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm
            }
            
            # Print metrics
            print("\nBI-LSTM Classification Model Evaluation Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred_classes))
            
            # Save metrics to file
            with open('output/bilstm/classification_metrics.txt', 'w') as f:
                f.write("=== BI-LSTM Classification Model Evaluation Metrics ===\n\n")
                f.write(f"Accuracy: {accuracy:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(classification_report(y_test, y_pred_classes))
            
            print("Saved classification metrics to output/bilstm/classification_metrics.txt")
            
            return metrics
        else:
            # For regression model
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
    
    def visualize_confusion_matrix(self, cm, class_names=None):
        """
        Visualize confusion matrix with a more stylish design
        
        Args:
            cm (numpy.ndarray): Confusion matrix
            class_names (list): List of class names
        """
        # If class names not provided, use numeric indices
        if class_names is None:
            if hasattr(self, 'grade_encoder') and self.grade_encoder is not None:
                class_names = self.grade_encoder.classes_
            else:
                class_names = [str(i) for i in range(cm.shape[0])]
        
        # Check if binary classification (2x2 matrix)
        if cm.shape[0] == 2:
            # For binary classification, use the stylish TP, FP, TN, FN format
            plt.figure(figsize=(10, 8))
            
            # Extract values
            tn, fp = cm[0, 0], cm[0, 1]
            fn, tp = cm[1, 0], cm[1, 1]
            
            # Create a 2x2 grid for the confusion matrix
            ax = plt.subplot(111)
            
            # Define colors
            tp_color = '#1e7b5e'  # Dark green
            tn_color = '#1e7b5e'  # Dark green
            fp_color = '#e15a4c'  # Red
            fn_color = '#e15a4c'  # Red
            
            # Remove axes
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            # Create the 2x2 grid with custom patches
            # True Positive (top-left)
            tp_rect = plt.Rectangle((0, 1), 1, 1, fill=True, color=tp_color, alpha=0.8)
            ax.add_patch(tp_rect)
            ax.text(0.5, 1.5, 'TRUE POSITIVE', ha='center', va='center', color='white', fontsize=18, fontweight='bold')
            ax.text(0.5, 1.25, str(tp), ha='center', va='center', color='white', fontsize=24, fontweight='bold')
            
            # False Negative (top-right)
            fn_rect = plt.Rectangle((1, 1), 1, 1, fill=True, color=fn_color, alpha=0.8)
            ax.add_patch(fn_rect)
            ax.text(1.5, 1.5, 'FALSE NEGATIVE', ha='center', va='center', color='white', fontsize=18, fontweight='bold')
            ax.text(1.5, 1.25, str(fn), ha='center', va='center', color='white', fontsize=24, fontweight='bold')
            
            # False Positive (bottom-left)
            fp_rect = plt.Rectangle((0, 0), 1, 1, fill=True, color=fp_color, alpha=0.8)
            ax.add_patch(fp_rect)
            ax.text(0.5, 0.5, 'FALSE POSITIVE', ha='center', va='center', color='white', fontsize=18, fontweight='bold')
            ax.text(0.5, 0.25, str(fp), ha='center', va='center', color='white', fontsize=24, fontweight='bold')
            
            # True Negative (bottom-right)
            tn_rect = plt.Rectangle((1, 0), 1, 1, fill=True, color=tn_color, alpha=0.8)
            ax.add_patch(tn_rect)
            ax.text(1.5, 0.5, 'TRUE NEGATIVE', ha='center', va='center', color='white', fontsize=18, fontweight='bold')
            ax.text(1.5, 0.25, str(tn), ha='center', va='center', color='white', fontsize=24, fontweight='bold')
            
            # Add labels
            plt.text(1.0, 2.1, 'PREDICTED', ha='center', va='center', fontsize=20, fontweight='bold')
            plt.text(-0.3, 1.0, 'ACTUAL', ha='center', va='center', rotation=90, fontsize=20, fontweight='bold')
            
            plt.text(0.5, 2.0, 'Positive', ha='center', va='center', fontsize=16)
            plt.text(1.5, 2.0, 'Negative', ha='center', va='center', fontsize=16)
            plt.text(-0.2, 1.5, 'Positive', ha='center', va='center', rotation=90, fontsize=16)
            plt.text(-0.2, 0.5, 'Negative', ha='center', va='center', rotation=90, fontsize=16)
            
            # Set limits and remove ticks
            ax.set_xlim(-0.5, 2.5)
            ax.set_ylim(-0.5, 2.5)
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.title('Confusion Matrix', fontsize=22, fontweight='bold', pad=20)
            
        else:
            # For multi-class, use a heatmap but with improved styling
            plt.figure(figsize=(12, 10))
            
            # Create a DataFrame for better labeling
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            
            # Normalize the confusion matrix
            cm_norm = cm_df.astype('float') / cm_df.sum(axis=1).values.reshape(-1, 1)
            
            # Create a mask for the diagonal (correctly classified)
            mask = np.zeros_like(cm, dtype=bool)
            np.fill_diagonal(mask, True)
            
            # Create a custom colormap: green for diagonal, red for off-diagonal
            cmap = plt.cm.Reds
            
            # Plot the heatmap with improved styling
            ax = plt.subplot(111)
            
            # Plot the correctly classified instances (diagonal) in green
            sns.heatmap(cm_df, annot=True, fmt="d", cmap='Greens', mask=~mask,
                       linewidths=1, linecolor='white', cbar=False,
                       annot_kws={"size": 14, "weight": "bold"}, ax=ax)
            
            # Plot the misclassified instances (off-diagonal) in red
            sns.heatmap(cm_df, annot=True, fmt="d", cmap='Reds', mask=mask,
                       linewidths=1, linecolor='white', cbar=False,
                       annot_kws={"size": 14, "weight": "bold"}, ax=ax)
            
            # Add labels
            plt.title('Confusion Matrix', fontsize=22, fontweight='bold', pad=20)
            plt.ylabel('True Label', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')
            
            # Rotate the tick labels and set alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=12)
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=12)
            
            # Add text with accuracy information
            accuracy = np.trace(cm) / np.sum(cm)
            plt.figtext(0.5, 0.01, f'Accuracy: {accuracy:.2%}', ha='center', fontsize=14, 
                       bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/bilstm/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved confusion matrix visualization to output/bilstm/confusion_matrix.png")
    
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
        
        # For classification, convert to class predictions if needed
        if self.is_classifier and self.num_classes > 2:
            predictions = np.argmax(predictions, axis=1)
        elif self.is_classifier:
            predictions = (predictions > 0.5).astype(int).flatten()
            
        return predictions


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