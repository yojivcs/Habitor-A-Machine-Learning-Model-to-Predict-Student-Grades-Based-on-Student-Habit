import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, accuracy_score
)
import os
import pickle
from datetime import datetime

class ModelEvaluator:
    def __init__(self):
        """
        Initialize the ModelEvaluator
        """
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.best_model = None
        self.confusion_matrices = {}
        
        # Create output directory if it doesn't exist
        os.makedirs('output/evaluation', exist_ok=True)
    
    def add_model(self, name, model, is_classifier=False):
        """
        Add a model to the evaluator
        
        Args:
            name (str): Name of the model
            model: Trained model object
            is_classifier (bool): Whether the model is a classifier
        """
        self.models[name] = {
            'model': model,
            'is_classifier': is_classifier
        }
        print(f"Added {name} to models for evaluation")
    
    def predict_all(self, X):
        """
        Make predictions with all added models
        
        Args:
            X (numpy.ndarray): Input features
            
        Returns:
            dict: Dictionary of predictions from all models
        """
        print("Making predictions with all models...")
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            print(f"Making predictions with {name}...")
            
            # Check if model is TensorFlow/Keras model
            if hasattr(model, 'predict'):
                predictions = model.predict(X)
                if len(predictions.shape) > 1 and predictions.shape[1] == 1:
                    predictions = predictions.flatten()
            else:
                predictions = model.predict(X)
            
            self.predictions[name] = predictions
        
        return self.predictions
    
    def evaluate_all(self, X_test, y_test):
        """
        Evaluate all models against test data
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test target
            
        Returns:
            dict: Dictionary of evaluation metrics for all models
        """
        print("Evaluating all models...")
        
        # Make predictions if not already made
        if not self.predictions:
            self.predict_all(X_test)
        
        # Initialize metrics dictionary
        self.metrics = {}
        
        # Evaluate each model
        for name, predictions in self.predictions.items():
            model_info = self.models[name]
            is_classifier = model_info['is_classifier']
            
            print(f"Evaluating {name}...")
            
            if is_classifier:
                # For classifiers, calculate classification metrics
                self.metrics[name] = self.evaluate_classifier(y_test, predictions, name)
            else:
                # For regressors, calculate regression metrics
                self.metrics[name] = self.evaluate_regressor(y_test, predictions)
        
        # Find best model
        self.determine_best_model()
        
        # Generate comparative visualizations
        self.visualize_model_comparison(y_test)
        
        # Generate evaluation report
        self.generate_evaluation_report()
        
        return self.metrics
    
    def evaluate_regressor(self, y_true, y_pred):
        """
        Evaluate a regression model
        
        Args:
            y_true (numpy.ndarray): True target values
            y_pred (numpy.ndarray): Predicted target values
            
        Returns:
            dict: Evaluation metrics
        """
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Create metrics dictionary
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # Print metrics
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R2): {r2:.4f}")
        
        return metrics
    
    def evaluate_classifier(self, y_true, y_pred, model_name):
        """
        Evaluate a classification model
        
        Args:
            y_true (numpy.ndarray): True target values
            y_pred (numpy.ndarray): Predicted target values
            model_name (str): Name of the model
            
        Returns:
            dict: Evaluation metrics
        """
        # If predictions are probabilities, convert to class labels
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred_classes = np.argmax(y_pred, axis=1)
        else:
            # For binary classification or already predicted classes
            y_pred_classes = np.round(y_pred).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_classes)
        
        # Generate classification report
        report = classification_report(y_true, y_pred_classes, output_dict=True)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        self.confusion_matrices[model_name] = cm
        
        # Visualize confusion matrix
        self.visualize_confusion_matrix(cm, model_name)
        
        # Create metrics dictionary
        metrics = {
            'accuracy': accuracy,
            'classification_report': report
        }
        
        # Print metrics
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_true, y_pred_classes))
        
        return metrics
    
    def visualize_confusion_matrix(self, cm, model_name):
        """
        Visualize confusion matrix
        
        Args:
            cm (numpy.ndarray): Confusion matrix
            model_name (str): Name of the model
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f'output/evaluation/confusion_matrix_{model_name}.png')
        plt.close()
        
        print(f"Saved confusion matrix visualization to output/evaluation/confusion_matrix_{model_name}.png")
    
    def determine_best_model(self):
        """
        Determine the best model based on metrics
        
        Returns:
            str: Name of the best model
        """
        if not self.metrics:
            print("No models evaluated yet. Call evaluate_all() first.")
            return None
        
        # For regression models, use RMSE as the metric (lower is better)
        regression_models = {name: metrics for name, metrics in self.metrics.items() 
                           if 'rmse' in metrics}
        
        # For classification models, use accuracy as the metric (higher is better)
        classification_models = {name: metrics for name, metrics in self.metrics.items() 
                               if 'accuracy' in metrics}
        
        # Determine best model
        if regression_models:
            # Find model with lowest RMSE
            best_regression_model = min(regression_models.items(), key=lambda x: x[1]['rmse'])
            best_regression_name = best_regression_model[0]
            best_regression_rmse = best_regression_model[1]['rmse']
            print(f"Best regression model: {best_regression_name} (RMSE: {best_regression_rmse:.4f})")
            
            self.best_model = best_regression_name
        
        if classification_models:
            # Find model with highest accuracy
            best_classification_model = max(classification_models.items(), key=lambda x: x[1]['accuracy'])
            best_classification_name = best_classification_model[0]
            best_classification_accuracy = best_classification_model[1]['accuracy']
            print(f"Best classification model: {best_classification_name} (Accuracy: {best_classification_accuracy:.4f})")
            
            if not self.best_model:
                self.best_model = best_classification_name
        
        return self.best_model
    
    def visualize_model_comparison(self, y_true):
        """
        Visualize model comparison
        
        Args:
            y_true (numpy.ndarray): True target values
        """
        if not self.predictions:
            print("No predictions available. Call predict_all() first.")
            return
        
        # Check if models are regressors or classifiers
        regression_models = {name: pred for name, pred in self.predictions.items() 
                           if not self.models[name]['is_classifier']}
        
        # For regression models, create scatter plots
        if regression_models:
            self.visualize_regression_comparison(y_true, regression_models)
        
        # For classification models, compare accuracy, precision, recall, etc.
        classification_models = {name: self.metrics[name] for name in self.predictions 
                               if self.models[name]['is_classifier'] and name in self.metrics}
        
        if classification_models:
            self.visualize_classification_comparison(classification_models)
    
    def visualize_regression_comparison(self, y_true, predictions):
        """
        Visualize regression model comparison
        
        Args:
            y_true (numpy.ndarray): True target values
            predictions (dict): Dictionary of predictions from regression models
        """
        # 1. Scatter plot comparison
        plt.figure(figsize=(12, 8))
        
        markers = ['o', 's', 'd', '^', 'v']
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, (name, y_pred) in enumerate(predictions.items()):
            plt.scatter(y_true, y_pred, alpha=0.5, marker=markers[i % len(markers)],
                      color=colors[i % len(colors)], label=name)
        
        # Add perfect prediction line
        min_val = min(np.min(y_true), min([np.min(p) for p in predictions.values()]))
        max_val = max(np.max(y_true), max([np.max(p) for p in predictions.values()]))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
        
        plt.title('Model Predictions Comparison')
        plt.xlabel('Actual CGPA')
        plt.ylabel('Predicted CGPA')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/evaluation/regression_comparison_scatter.png')
        plt.close()
        
        # 2. Error distribution comparison
        plt.figure(figsize=(12, 8))
        
        for name, y_pred in predictions.items():
            errors = y_pred - y_true
            sns.kdeplot(errors, label=name)
        
        plt.title('Prediction Error Distributions')
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.axvline(x=0, color='k', linestyle='--')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/evaluation/regression_comparison_error_dist.png')
        plt.close()
        
        # 3. Metrics comparison
        regression_metrics = {name: self.metrics[name] for name in predictions if name in self.metrics}
        
        if regression_metrics:
            # Extract metrics
            names = list(regression_metrics.keys())
            rmse_values = [metrics['rmse'] for metrics in regression_metrics.values()]
            mae_values = [metrics['mae'] for metrics in regression_metrics.values()]
            r2_values = [metrics['r2'] for metrics in regression_metrics.values()]
            
            # Create figure with 3 subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # RMSE (lower is better)
            axes[0].bar(names, rmse_values)
            axes[0].set_title('RMSE Comparison (Lower is Better)')
            axes[0].set_ylabel('RMSE')
            axes[0].grid(axis='y')
            
            # MAE (lower is better)
            axes[1].bar(names, mae_values)
            axes[1].set_title('MAE Comparison (Lower is Better)')
            axes[1].set_ylabel('MAE')
            axes[1].grid(axis='y')
            
            # R² (higher is better)
            axes[2].bar(names, r2_values)
            axes[2].set_title('R² Comparison (Higher is Better)')
            axes[2].set_ylabel('R²')
            axes[2].grid(axis='y')
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig('output/evaluation/regression_metrics_comparison.png')
            plt.close()
        
        print("Saved regression model comparison visualizations to output/evaluation/")
    
    def visualize_classification_comparison(self, metrics):
        """
        Visualize classification model comparison
        
        Args:
            metrics (dict): Dictionary of metrics from classification models
        """
        # Extract model names
        names = list(metrics.keys())
        
        # 1. Accuracy comparison
        accuracy_values = [m['accuracy'] for m in metrics.values()]
        
        plt.figure(figsize=(10, 6))
        plt.bar(names, accuracy_values)
        plt.title('Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(axis='y')
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/evaluation/classification_accuracy_comparison.png')
        plt.close()
        
        # 2. Precision, Recall, F1-score comparison (for each class if multiclass)
        # First, determine if it's binary or multiclass classification
        first_report = list(metrics.values())[0]['classification_report']
        classes = [c for c in first_report.keys() if c not in ['accuracy', 'macro avg', 'weighted avg']]
        
        for metric_name in ['precision', 'recall', 'f1-score']:
            plt.figure(figsize=(12, 6))
            
            for i, class_label in enumerate(classes):
                class_values = [m['classification_report'][class_label][metric_name] for m in metrics.values()]
                
                x = np.arange(len(names))
                width = 0.8 / len(classes)
                offset = width * i - (width * (len(classes) - 1)) / 2
                
                plt.bar(x + offset, class_values, width=width, label=f'Class {class_label}')
            
            plt.title(f'{metric_name.capitalize()} Comparison')
            plt.ylabel(metric_name.capitalize())
            plt.xlabel('Model')
            plt.xticks(range(len(names)), names)
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(axis='y')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f'output/evaluation/classification_{metric_name}_comparison.png')
            plt.close()
        
        print("Saved classification model comparison visualizations to output/evaluation/")
    
    def generate_evaluation_report(self):
        """
        Generate a comprehensive evaluation report
        """
        if not self.metrics:
            print("No models evaluated yet. Call evaluate_all() first.")
            return
        
        print("Generating evaluation report...")
        
        # Create report file
        with open('output/evaluation/model_evaluation_report.txt', 'w') as f:
            f.write("======================================\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("======================================\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary of models evaluated
            f.write("Models Evaluated:\n")
            for name, model_info in self.models.items():
                model_type = "Classification" if model_info['is_classifier'] else "Regression"
                f.write(f"- {name} ({model_type} Model)\n")
            f.write("\n")
            
            # Write evaluation metrics for each model
            f.write("Evaluation Metrics:\n\n")
            
            # Group by model type
            regression_models = {name: metrics for name, metrics in self.metrics.items() 
                               if 'rmse' in metrics}
            
            classification_models = {name: metrics for name, metrics in self.metrics.items() 
                                   if 'accuracy' in metrics}
            
            # Write regression metrics
            if regression_models:
                f.write("Regression Models:\n")
                f.write("-----------------\n\n")
                
                for name, metrics in regression_models.items():
                    f.write(f"{name}:\n")
                    f.write(f"- Mean Squared Error (MSE): {metrics['mse']:.4f}\n")
                    f.write(f"- Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n")
                    f.write(f"- Mean Absolute Error (MAE): {metrics['mae']:.4f}\n")
                    f.write(f"- R-squared (R2): {metrics['r2']:.4f}\n\n")
            
            # Write classification metrics
            if classification_models:
                f.write("Classification Models:\n")
                f.write("---------------------\n\n")
                
                for name, metrics in classification_models.items():
                    f.write(f"{name}:\n")
                    f.write(f"- Accuracy: {metrics['accuracy']:.4f}\n")
                    
                    # Add confusion matrix info
                    if name in self.confusion_matrices:
                        f.write(f"- Confusion Matrix: See confusion_matrix_{name}.png\n")
                    
                    # Add classification report
                    f.write("- Classification Report:\n")
                    report = metrics['classification_report']
                    
                    for class_label, class_metrics in report.items():
                        if isinstance(class_metrics, dict):
                            f.write(f"  Class {class_label}:\n")
                            f.write(f"    - Precision: {class_metrics['precision']:.4f}\n")
                            f.write(f"    - Recall: {class_metrics['recall']:.4f}\n")
                            f.write(f"    - F1-score: {class_metrics['f1-score']:.4f}\n")
                            f.write(f"    - Support: {class_metrics['support']}\n")
                    f.write("\n")
            
            # Write conclusion
            f.write("======================================\n")
            f.write("CONCLUSION\n")
            f.write("======================================\n\n")
            
            if self.best_model:
                f.write(f"The best performing model is: {self.best_model}\n\n")
                
                if self.best_model in regression_models:
                    best_metrics = regression_models[self.best_model]
                    f.write(f"Performance:\n")
                    f.write(f"- RMSE: {best_metrics['rmse']:.4f}\n")
                    f.write(f"- R2 Score: {best_metrics['r2']:.4f}\n\n")
                else:
                    best_metrics = classification_models[self.best_model]
                    f.write(f"Performance:\n")
                    f.write(f"- Accuracy: {best_metrics['accuracy']:.4f}\n\n")
            
            f.write("Visualizations generated:\n")
            
            if regression_models:
                f.write("- regression_comparison_scatter.png: Scatter plot comparing all model predictions\n")
                f.write("- regression_comparison_error_dist.png: Distribution of prediction errors\n")
                f.write("- regression_metrics_comparison.png: Comparison of regression metrics\n")
            
            if classification_models:
                f.write("- classification_accuracy_comparison.png: Comparison of model accuracies\n")
                f.write("- classification_precision_comparison.png: Comparison of precision by class\n")
                f.write("- classification_recall_comparison.png: Comparison of recall by class\n")
                f.write("- classification_f1-score_comparison.png: Comparison of F1-scores by class\n")
                
                for name in classification_models:
                    f.write(f"- confusion_matrix_{name}.png: Confusion matrix for {name}\n")
            
            f.write("\nThis report was generated automatically by the Model Evaluation module.\n")
        
        print("Saved evaluation report to output/evaluation/model_evaluation_report.txt")


# For testing the module independently
if __name__ == "__main__":
    from data_collection import DataCollector
    from preprocessing import DataPreprocessor
    from count_vectorization import FeatureVectorizer
    from bilstm_model import BiLSTMModel
    from gradient_boosting_model import GradientBoostingModel
    
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
        
        # Convert to dense arrays if needed
        if hasattr(X_train_vectorized, 'toarray'):
            X_train_dense = X_train_vectorized.toarray()
            X_test_dense = X_test_vectorized.toarray()
        else:
            X_train_dense = X_train_vectorized
            X_test_dense = X_test_vectorized
        
        # Create and train models
        
        # 1. Gradient Boosting model
        gb_model = GradientBoostingModel()
        gb_model.build_model()
        gb_model.train(X_train_dense, y_train)
        
        # 2. LSTM model (if TensorFlow is available)
        try:
            import tensorflow as tf
            bilstm = BiLSTMModel(X_train_dense.shape[1])
            bilstm.build_model()
            bilstm.train(X_train_dense, y_train, X_test_dense, y_test, epochs=10)
            have_lstm = True
        except:
            have_lstm = False
        
        # Create model evaluator
        evaluator = ModelEvaluator()
        
        # Add models to evaluator
        evaluator.add_model('Gradient_Boosting', gb_model.model)
        if have_lstm:
            evaluator.add_model('BI_LSTM', bilstm.model)
        
        # Evaluate models
        metrics = evaluator.evaluate_all(X_test_dense, y_test)
        
        print("\nModel evaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during model evaluation: {e}") 