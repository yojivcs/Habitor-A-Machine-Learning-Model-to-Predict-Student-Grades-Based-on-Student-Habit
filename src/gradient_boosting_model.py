import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, learning_curve
import os
import pickle

class GradientBoostingModel:
    def __init__(self):
        """
        Initialize the Gradient Boosting model
        """
        self.model = None
        self.best_params = None
        self.feature_importances = None
        
        # Create output directory if it doesn't exist
        os.makedirs('output/gradient_boosting', exist_ok=True)
    
    def build_model(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        """
        Build a Gradient Boosting model with specified parameters
        
        Args:
            n_estimators (int): Number of boosting stages
            learning_rate (float): Learning rate
            max_depth (int): Maximum depth of individual regression estimators
            random_state (int): Random seed for reproducibility
            
        Returns:
            GradientBoostingRegressor: Built model
        """
        print("Building Gradient Boosting model...")
        
        # Create model
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
        
        # Save parameters
        self.best_params = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'random_state': random_state
        }
        
        # Save model configuration
        with open('output/gradient_boosting/model_config.txt', 'w') as f:
            f.write("=== Gradient Boosting Model Configuration ===\n\n")
            for param, value in self.best_params.items():
                f.write(f"{param}: {value}\n")
        
        print("Saved model configuration to output/gradient_boosting/model_config.txt")
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train, cv=5):
        """
        Perform hyperparameter tuning using GridSearchCV
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training target
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Best parameters
        """
        print("Performing hyperparameter tuning for Gradient Boosting model...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
        
        # Create base model
        model = GradientBoostingRegressor(random_state=42)
        
        # Create grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best parameters
        self.best_params = grid_search.best_params_
        
        # Create model with best parameters
        self.model = GradientBoostingRegressor(
            n_estimators=self.best_params['n_estimators'],
            learning_rate=self.best_params['learning_rate'],
            max_depth=self.best_params['max_depth'],
            subsample=self.best_params['subsample'],
            random_state=42
        )
        
        # Save best parameters
        with open('output/gradient_boosting/best_params.txt', 'w') as f:
            f.write("=== Gradient Boosting Best Parameters ===\n\n")
            for param, value in self.best_params.items():
                f.write(f"{param}: {value}\n")
        
        print("Saved best parameters to output/gradient_boosting/best_params.txt")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def train(self, X_train, y_train):
        """
        Train the Gradient Boosting model
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training target
            
        Returns:
            GradientBoostingRegressor: Trained model
        """
        print("Training Gradient Boosting model...")
        
        if self.model is None:
            self.build_model()
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Get feature importances
        self.feature_importances = self.model.feature_importances_
        
        # Visualize feature importances
        self.visualize_feature_importances(X_train)
        
        # Visualize learning curve
        self.plot_learning_curve(X_train, y_train)
        
        # Save trained model
        with open('output/gradient_boosting/model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        print("Saved trained model to output/gradient_boosting/model.pkl")
        
        return self.model
    
    def visualize_feature_importances(self, X_train, top_n=20):
        """
        Visualize feature importances
        
        Args:
            X_train (numpy.ndarray): Training features
            top_n (int): Number of top features to display
        """
        if self.feature_importances is None or not hasattr(X_train, 'columns'):
            return
            
        # Create DataFrame of feature importances
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
        else:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importances))]
            
        importances_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances
        })
        
        # Sort by importance
        importances_df = importances_df.sort_values('importance', ascending=False)
        
        # Take top N features
        top_features = importances_df.head(min(top_n, len(importances_df)))
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {len(top_features)} Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/gradient_boosting/feature_importances.png')
        plt.close()
        
        # Save feature importances to file
        importances_df.to_csv('output/gradient_boosting/feature_importances.csv', index=False)
        
        print("Saved feature importances to output/gradient_boosting/")
    
    def plot_learning_curve(self, X_train, y_train, cv=5):
        """
        Plot learning curve to evaluate model performance with varying training sizes
        
        Args:
            X_train (numpy.ndarray): Training features
            y_train (numpy.ndarray): Training target
            cv (int): Number of cross-validation folds
        """
        print("Generating learning curve...")
        
        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv, scoring='neg_mean_squared_error'
        )
        
        # Calculate mean and standard deviation
        train_mean = -np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = -np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.grid()
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
        plt.title('Learning Curve')
        plt.xlabel('Training Size')
        plt.ylabel('Mean Squared Error')
        plt.legend(loc="best")
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/gradient_boosting/learning_curve.png')
        plt.close()
        
        print("Saved learning curve to output/gradient_boosting/learning_curve.png")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test (numpy.ndarray): Test features
            y_test (numpy.ndarray): Test target
            
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating Gradient Boosting model...")
        
        if self.model is None:
            print("Model not trained yet. Call train() first.")
            return None
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Create metrics dictionary
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # Print metrics
        print("\nGradient Boosting Model Evaluation Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R2): {r2:.4f}")
        
        # Visualize predictions
        self.visualize_predictions(y_test, y_pred)
        
        # Save metrics to file
        with open('output/gradient_boosting/evaluation_metrics.txt', 'w') as f:
            f.write("=== Gradient Boosting Model Evaluation Metrics ===\n\n")
            f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
            f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
            f.write(f"R-squared (R2): {r2:.4f}\n")
        
        print("Saved evaluation metrics to output/gradient_boosting/evaluation_metrics.txt")
        
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
        
        plt.title('Gradient Boosting: Actual vs Predicted CGPA')
        plt.xlabel('Actual CGPA')
        plt.ylabel('Predicted CGPA')
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/gradient_boosting/predictions.png')
        plt.close()
        
        # Create error distribution plot
        errors = y_pred - y_true
        
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=20, alpha=0.7)
        plt.title('Gradient Boosting: Prediction Error Distribution')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/gradient_boosting/error_distribution.png')
        plt.close()
        
        # Create residual plot
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, errors, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Gradient Boosting: Residual Plot')
        plt.xlabel('Predicted CGPA')
        plt.ylabel('Residual')
        plt.grid(True)
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/gradient_boosting/residual_plot.png')
        plt.close()
        
        print("Saved prediction visualizations to output/gradient_boosting/")
    
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
        
        return predictions


# For testing the module independently
if __name__ == "__main__":
    from data_collection import DataCollector
    from preprocessing import DataPreprocessor
    from count_vectorization import FeatureVectorizer
    
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
    
    # Build and train Gradient Boosting model
    gb_model = GradientBoostingModel()
    gb_model.build_model()
    gb_model.train(X_train_dense, y_train)
    
    # Evaluate model
    metrics = gb_model.evaluate(X_test_dense, y_test)
    
    print("\nGradient Boosting model training and evaluation completed successfully!") 