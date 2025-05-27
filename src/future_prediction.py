import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

class FuturePredictor:
    def __init__(self, bilstm_model=None, gradient_model=None):
        """
        Initialize the FuturePredictor with trained models
        
        Args:
            bilstm_model: Trained BI-LSTM model (optional)
            gradient_model: Trained Gradient Boosting model (optional)
        """
        self.bilstm_model = bilstm_model
        self.gradient_model = gradient_model
        self.ensemble_weights = None  # For weighted ensemble predictions
        self.prediction_results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs('output/future_prediction', exist_ok=True)
    
    def load_models(self, bilstm_path=None, gradient_path=None):
        """
        Load trained models from files
        
        Args:
            bilstm_path (str): Path to saved BI-LSTM model
            gradient_path (str): Path to saved Gradient Boosting model
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
        
        return predictions.flatten()
    
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
        
        # Calculate weighted ensemble
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
        print("Making predictions with all available models...")
        
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
                
            # Calculate metrics
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
            f.write("=== Prediction Evaluation Metrics ===\n\n")
            for model_name, metrics in evaluation.items():
                f.write(f"{model_name.upper()} Model:\n")
                f.write(f"Mean Squared Error (MSE): {metrics['mse']:.4f}\n")
                f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n")
                f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.4f}\n")
                f.write(f"R-squared (R2): {metrics['r2']:.4f}\n\n")
        
        print("Saved evaluation metrics to output/future_prediction/evaluation_metrics.txt")
        
        return evaluation
    
    def visualize_predictions(self, predictions, y_true):
        """
        Visualize predictions from different models
        
        Args:
            predictions (dict): Dictionary of predictions from different models
            y_true (numpy.ndarray): True target values
        """
        print("Visualizing predictions...")
        
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
            f.write("FUTURE PREDICTION REPORT\n")
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
                f.write(f"- Mean Squared Error (MSE): {metrics['mse']:.4f}\n")
                f.write(f"- Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n")
                f.write(f"- Mean Absolute Error (MAE): {metrics['mae']:.4f}\n")
                f.write(f"- R-squared (R2): {metrics['r2']:.4f}\n")
            
            # Write conclusion
            f.write("\n======================================\n")
            f.write("CONCLUSION\n")
            f.write("======================================\n\n")
            
            # Find best model based on RMSE
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