import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

class FuturePredictor:
    def __init__(self, bilstm_model=None, gradient_model=None, is_classifier=False):
        """
        Initialize the FuturePredictor with trained models
        
        Args:
            bilstm_model: Trained BI-LSTM model (optional)
            gradient_model: Trained Gradient Boosting model (optional)
            is_classifier (bool): Whether the models are classifiers
        """
        self.bilstm_model = bilstm_model
        self.gradient_model = gradient_model
        self.ensemble_weights = None  # For weighted ensemble predictions
        self.prediction_results = {}
        self.is_classifier = is_classifier
        self.grade_encoder = None
        
        # Create output directory if it doesn't exist
        os.makedirs('output/future_prediction', exist_ok=True)
        
        print(f"Future Predictor initialized in {'classification' if is_classifier else 'regression'} mode")
    
    def load_models(self, bilstm_path=None, gradient_path=None, grade_encoder_path=None):
        """
        Load trained models from files
        
        Args:
            bilstm_path (str): Path to saved BI-LSTM model
            gradient_path (str): Path to saved Gradient Boosting model
            grade_encoder_path (str): Path to grade encoder for classification (optional)
        """
        print("Loading trained models...")
        
        # Load BI-LSTM model if path provided
        if bilstm_path:
            try:
                self.bilstm_model = tf.keras.models.load_model(bilstm_path)
                print(f"Loaded BI-LSTM model from {bilstm_path}")
            except Exception as e:
                print(f"Error loading BI-LSTM model: {e}")
        
        # Load Gradient Boosting model if path provided
        if gradient_path:
            try:
                with open(gradient_path, 'rb') as f:
                    self.gradient_model = pickle.load(f)
                print(f"Loaded Gradient Boosting model from {gradient_path}")
            except Exception as e:
                print(f"Error loading Gradient Boosting model: {e}")
        
        # Load grade encoder if path provided and in classification mode
        if self.is_classifier and grade_encoder_path:
            try:
                with open(grade_encoder_path, 'rb') as f:
                    self.grade_encoder = pickle.load(f)
                print(f"Loaded grade encoder from {grade_encoder_path}")
            except Exception as e:
                print(f"Error loading grade encoder: {e}")
    
    def predict_with_bilstm(self, X):
        """
        Make predictions using the BI-LSTM model
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Predicted values
        """
        if self.bilstm_model is None:
            print("BI-LSTM model not loaded. Call load_models() first.")
            return None
        
        # Make predictions
        predictions = self.bilstm_model.predict(X)
        
        # For classification, convert to class predictions if needed
        if self.is_classifier:
            # Check if output shape is multi-class
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Multi-class classification
                predictions = np.argmax(predictions, axis=1)
            else:
                # Binary classification
                predictions = (predictions > 0.5).astype(int).flatten()
        else:
            # Flatten regression predictions
            predictions = predictions.flatten()
        
        return predictions
    
    def predict_with_gradient(self, X):
        """
        Make predictions using the Gradient Boosting model
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            numpy.ndarray: Predicted values
        """
        if self.gradient_model is None:
            print("Gradient Boosting model not loaded. Call load_models() first.")
            return None
        
        # Make predictions
        predictions = self.gradient_model.predict(X)
        
        return predictions
    
    def predict_ensemble(self, X, weights=None):
        """
        Make predictions using an ensemble of models
        
        Args:
            X (numpy.ndarray): Input features
            weights (list): Weights for each model [bilstm_weight, gradient_weight]
            
        Returns:
            numpy.ndarray: Ensemble predicted values
        """
        if self.bilstm_model is None or self.gradient_model is None:
            print("Both models must be loaded for ensemble prediction. Call load_models() first.")
            return None
        
        # Use default weights if not provided
        if weights is None:
            weights = [0.5, 0.5]  # Equal weights by default
        
        # Store weights for future reference
        self.ensemble_weights = weights
        
        # Get predictions from each model
        bilstm_preds = self.predict_with_bilstm(X)
        gradient_preds = self.predict_with_gradient(X)
        
        if self.is_classifier:
            # For classification, use majority vote or weighted probability
            # Here we'll implement a simple majority vote
            ensemble_preds = np.zeros_like(bilstm_preds)
            
            # If weights[0] > weights[1], prefer bilstm prediction, else prefer gradient prediction
            for i in range(len(ensemble_preds)):
                if weights[0] > weights[1]:
                    ensemble_preds[i] = bilstm_preds[i]
                else:
                    ensemble_preds[i] = gradient_preds[i]
        else:
            # For regression, calculate weighted average
            ensemble_preds = (weights[0] * bilstm_preds) + (weights[1] * gradient_preds)
        
        return ensemble_preds
    
    def predict_all(self, X, y_true=None):
        """
        Make predictions using all available models and evaluate if true values are provided
        
        Args:
            X (numpy.ndarray): Input features
            y_true (numpy.ndarray): True target values (optional)
            
        Returns:
            dict: Dictionary of prediction results
        """
        print(f"Making predictions with all available models ({('classification' if self.is_classifier else 'regression')})...")
        
        # Initialize results dictionary
        results = {}
        
        # Predict with BI-LSTM if available
        if self.bilstm_model is not None:
            bilstm_preds = self.predict_with_bilstm(X)
            results['bilstm'] = bilstm_preds
            print("Made predictions with BI-LSTM model")
        
        # Predict with Gradient Boosting if available
        if self.gradient_model is not None:
            gradient_preds = self.predict_with_gradient(X)
            results['gradient'] = gradient_preds
            print("Made predictions with Gradient Boosting model")
        
        # Predict with ensemble if both models are available
        if self.bilstm_model is not None and self.gradient_model is not None:
            ensemble_preds = self.predict_ensemble(X)
            results['ensemble'] = ensemble_preds
            print("Made predictions with ensemble model")
        
        # Evaluate predictions if true values are provided
        if y_true is not None:
            evaluation = self.evaluate_predictions(results, y_true)
            results['evaluation'] = evaluation
        
        # Store results
        self.prediction_results = results
        
        # Visualize predictions
        if y_true is not None:
            self.visualize_predictions(results, y_true)
        
        return results
    
    def evaluate_predictions(self, predictions, y_true):
        """
        Evaluate the predictions against true values
        
        Args:
            predictions (dict): Dictionary of predictions from different models
            y_true (numpy.ndarray): True target values
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        print("Evaluating predictions...")
        
        # Initialize evaluation dictionary
        evaluation = {}
        
        # Evaluate each model's predictions
        for model_name, y_pred in predictions.items():
            if model_name == 'evaluation':
                continue
            
            if self.is_classifier:
                # Calculate classification metrics
                accuracy = accuracy_score(y_true, y_pred)
                report = classification_report(y_true, y_pred, output_dict=True)
                cm = confusion_matrix(y_true, y_pred)
                
                # Save confusion matrix visualization
                self.visualize_confusion_matrix(cm, model_name)
                
                # Store metrics
                evaluation[model_name] = {
                    'accuracy': accuracy,
                    'classification_report': report,
                    'confusion_matrix': cm
                }
                
                # Print metrics
                print(f"\nEvaluation Metrics for {model_name.upper()} model:")
                print(f"Accuracy: {accuracy:.4f}")
                print("\nClassification Report:")
                print(classification_report(y_true, y_pred))
            else:
                # Calculate regression metrics
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                # Store metrics
                evaluation[model_name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                # Print metrics
                print(f"\nEvaluation Metrics for {model_name.upper()} model:")
                print(f"Mean Squared Error (MSE): {mse:.4f}")
                print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
                print(f"Mean Absolute Error (MAE): {mae:.4f}")
                print(f"R-squared (R2): {r2:.4f}")
        
        # Save evaluation metrics
        with open('output/future_prediction/evaluation_metrics.txt', 'w') as f:
            f.write(f"=== Prediction Evaluation Metrics ({('CLASSIFICATION' if self.is_classifier else 'REGRESSION')}) ===\n\n")
            
            for model_name, metrics in evaluation.items():
                f.write(f"{model_name.upper()} Model:\n")
                
                if self.is_classifier:
                    f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")
                    f.write("Classification Report Summary:\n")
                    
                    # Extract overall metrics from report
                    if 'weighted avg' in metrics['classification_report']:
                        weighted = metrics['classification_report']['weighted avg']
                        f.write(f"Precision (weighted): {weighted['precision']:.4f}\n")
                        f.write(f"Recall (weighted): {weighted['recall']:.4f}\n")
                        f.write(f"F1-score (weighted): {weighted['f1-score']:.4f}\n\n")
                else:
                    f.write(f"Mean Squared Error (MSE): {metrics['mse']:.4f}\n")
                    f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n")
                    f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}\n")
                    f.write(f"R-squared (R2): {metrics['r2']:.4f}\n\n")
        
        print("Saved evaluation metrics to output/future_prediction/evaluation_metrics.txt")
        
        return evaluation
    
    def visualize_confusion_matrix(self, cm, model_name, class_names=None):
        """
        Visualize confusion matrix with a more stylish design
        
        Args:
            cm (numpy.ndarray): Confusion matrix
            model_name (str): Name of the model
            class_names (list): List of class names
        """
        # If class names not provided, use numeric indices or grade encoder if available
        if class_names is None:
            if self.grade_encoder is not None:
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
            
            plt.title(f'{model_name.upper()} Model Confusion Matrix', fontsize=22, fontweight='bold', pad=20)
            
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
            plt.title(f'{model_name.upper()} Model Confusion Matrix', fontsize=22, fontweight='bold', pad=20)
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
        plt.savefig(f'output/future_prediction/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrix visualization to output/future_prediction/confusion_matrix_{model_name}.png")
    
    def visualize_predictions(self, predictions, y_true):
        """
        Visualize predictions from different models
        
        Args:
            predictions (dict): Dictionary of predictions from different models
            y_true (numpy.ndarray): True target values
        """
        print("Visualizing predictions...")
        
        if self.is_classifier:
            # For classification, create confusion matrices
            for model_name, y_pred in predictions.items():
                if model_name == 'evaluation':
                    continue
                
                # Create confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                
                # Visualize confusion matrix
                self.visualize_confusion_matrix(cm, model_name)
            
            # Create bar chart comparing model accuracies
            if 'evaluation' in predictions:
                model_names = []
                accuracies = []
                
                for model_name, metrics in predictions['evaluation'].items():
                    model_names.append(model_name.upper())
                    accuracies.append(metrics['accuracy'])
                
                plt.figure(figsize=(10, 6))
                plt.bar(model_names, accuracies)
                plt.title('Model Accuracy Comparison')
                plt.xlabel('Model')
                plt.ylabel('Accuracy')
                plt.ylim(0, 1)
                plt.grid(True, axis='y')
                plt.tight_layout()
                
                plt.savefig('output/future_prediction/accuracy_comparison.png')
                plt.close()
                
                print("Saved accuracy comparison to output/future_prediction/accuracy_comparison.png")
        else:
            # Create scatter plot comparing all models
            plt.figure(figsize=(12, 8))
            
            # Plot true vs predicted for each model
            markers = ['o', 's', 'd']  # Different markers for different models
            colors = ['blue', 'green', 'red']  # Different colors for different models
            
            i = 0
            for model_name, y_pred in predictions.items():
                if model_name == 'evaluation':
                    continue
                    
                plt.scatter(y_true, y_pred, alpha=0.5, marker=markers[i % len(markers)],
                           color=colors[i % len(colors)], label=f'{model_name.upper()}')
                i += 1
            
            # Add perfect prediction line
            min_val = min(np.min(y_true), min([np.min(p) for p in predictions.values() if isinstance(p, np.ndarray)]))
            max_val = max(np.max(y_true), max([np.max(p) for p in predictions.values() if isinstance(p, np.ndarray)]))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
            
            plt.title('Model Predictions Comparison')
            plt.xlabel('Actual CGPA')
            plt.ylabel('Predicted CGPA')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Save figure
            plt.savefig('output/future_prediction/predictions_comparison.png')
            plt.close()
            
            # Create error distribution comparison
            plt.figure(figsize=(12, 8))
            
            for model_name, y_pred in predictions.items():
                if model_name == 'evaluation':
                    continue
                    
                errors = y_pred - y_true
                sns.kdeplot(errors, label=f'{model_name.upper()}')
            
            plt.title('Prediction Error Distributions')
            plt.xlabel('Prediction Error')
            plt.ylabel('Density')
            plt.axvline(x=0, color='k', linestyle='--')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Save figure
            plt.savefig('output/future_prediction/error_distributions.png')
            plt.close()
        
        print("Saved prediction visualizations to output/future_prediction/")
    
    def generate_prediction_report(self, feature_names=None):
        """
        Generate a comprehensive prediction report
        
        Args:
            feature_names (list): Names of the features (optional)
        """
        if not self.prediction_results or 'evaluation' not in self.prediction_results:
            print("No prediction results available. Call predict_all() first.")
            return
        
        print("Generating prediction report...")
        
        # Create report file
        with open('output/future_prediction/prediction_report.txt', 'w') as f:
            f.write("======================================\n")
            f.write(f"FUTURE PREDICTION REPORT ({('CLASSIFICATION' if self.is_classifier else 'REGRESSION')})\n")
            f.write("======================================\n\n")
            
            # Write summary of models used
            f.write("Models Used:\n")
            if self.bilstm_model is not None:
                f.write("- BI-LSTM (Bidirectional Long Short-Term Memory)\n")
            if self.gradient_model is not None:
                f.write("- BI-GRD (Gradient Boosting)\n")
            if self.bilstm_model is not None and self.gradient_model is not None:
                f.write("- Ensemble (Combined predictions)\n")
                if self.ensemble_weights:
                    f.write(f"  - Ensemble weights: BI-LSTM={self.ensemble_weights[0]}, BI-GRD={self.ensemble_weights[1]}\n")
            f.write("\n")
            
            # Write evaluation metrics
            f.write("Prediction Performance:\n")
            for model_name, metrics in self.prediction_results['evaluation'].items():
                f.write(f"\n{model_name.upper()} Model:\n")
                
                if self.is_classifier:
                    f.write(f"- Accuracy: {metrics['accuracy']:.4f}\n")
                    
                    # Extract overall metrics from report
                    if 'classification_report' in metrics and 'weighted avg' in metrics['classification_report']:
                        weighted = metrics['classification_report']['weighted avg']
                        f.write(f"- Precision (weighted): {weighted['precision']:.4f}\n")
                        f.write(f"- Recall (weighted): {weighted['recall']:.4f}\n")
                        f.write(f"- F1-score (weighted): {weighted['f1-score']:.4f}\n")
                else:
                    f.write(f"- Mean Squared Error (MSE): {metrics['mse']:.4f}\n")
                    f.write(f"- Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n")
                    f.write(f"- Mean Absolute Error (MAE): {metrics['mae']:.4f}\n")
                    f.write(f"- R-squared (R2): {metrics['r2']:.4f}\n")
            
            # Write conclusion
            f.write("\n======================================\n")
            f.write("CONCLUSION\n")
            f.write("======================================\n\n")
            
            # Find best model based on accuracy or RMSE
            if self.is_classifier:
                best_model = max(self.prediction_results['evaluation'].items(), 
                                key=lambda x: x[1]['accuracy'])[0]
                
                f.write(f"The best performing model is: {best_model.upper()}\n")
                f.write(f"- Accuracy: {self.prediction_results['evaluation'][best_model]['accuracy']:.4f}\n\n")
                
                f.write("Visualizations generated:\n")
                for model_name in self.prediction_results:
                    if model_name != 'evaluation':
                        f.write(f"- confusion_matrix_{model_name}.png: Confusion matrix for {model_name} model\n")
                f.write("- accuracy_comparison.png: Bar chart comparing model accuracies\n\n")
            else:
                best_model = min(self.prediction_results['evaluation'].items(), 
                                key=lambda x: x[1]['rmse'])[0]
                
                f.write(f"The best performing model is: {best_model.upper()}\n")
                f.write(f"- RMSE: {self.prediction_results['evaluation'][best_model]['rmse']:.4f}\n")
                f.write(f"- R2 Score: {self.prediction_results['evaluation'][best_model]['r2']:.4f}\n\n")
                
                f.write("Visualizations generated:\n")
                f.write("- predictions_comparison.png: Scatter plot comparing all model predictions\n")
                f.write("- error_distributions.png: Distribution of prediction errors for each model\n\n")
            
            f.write("This report was generated automatically by the Future Prediction module.\n")
        
        print("Saved prediction report to output/future_prediction/prediction_report.txt")


# For testing the module independently
if __name__ == "__main__":
    from data_collection import DataCollector
    from preprocessing import DataPreprocessor
    from count_vectorization import FeatureVectorizer
    from bilstm_model import BiLSTMModel
    import numpy as np
    
    try:
        # Load data
        collector = DataCollector("Reports and Dataset/student_habits_dataset.xlsx")
        data = collector.load_data()
        
        # Preprocess data
        preprocessor = DataPreprocessor(data)
        X_train, X_test, y_train, y_test = preprocessor.preprocess_data()
        
        # Vectorize features
        vectorizer = FeatureVectorizer(X_train, X_test)
        X_train_vectorized, X_test_vectorized = vectorizer.vectorize_and_combine()
        
        # Convert to dense arrays
        X_train_dense = X_train_vectorized.toarray()
        X_test_dense = X_test_vectorized.toarray()
        
        # Load models (assuming they've been trained and saved)
        predictor = FuturePredictor()
        
        # Try to load existing models
        try:
            predictor.load_models(
                bilstm_path='output/bilstm/final_model.h5',
                gradient_path='output/gradient_boosting/model.pkl'
            )
        except:
            print("Could not load saved models. Training a new BI-LSTM model...")
            
            # Train BI-LSTM model if not loaded
            if predictor.bilstm_model is None:
                bilstm = BiLSTMModel(X_train_dense.shape[1])
                bilstm.build_model()
                bilstm.train(X_train_dense, y_train, X_test_dense, y_test, epochs=10)
                predictor.bilstm_model = bilstm.model
        
        # Make predictions
        predictions = predictor.predict_all(X_test_dense, y_test)
        
        # Generate report
        feature_names = list(X_train.columns)
        predictor.generate_prediction_report(feature_names)
        
        print("\nFuture prediction completed successfully!")
        
    except Exception as e:
        print(f"Error during future prediction: {e}") 