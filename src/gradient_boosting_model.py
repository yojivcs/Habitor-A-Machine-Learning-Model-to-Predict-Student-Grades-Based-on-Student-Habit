import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, learning_curve
import os
import pickle

class GradientBoostingModel:
    def __init__(self, is_classifier=False):
        """
        Initialize the Gradient Boosting model
        
        Args:
            is_classifier (bool): Whether to use a classifier instead of regressor
        """
        self.model = None
        self.best_params = None
        self.feature_importances = None
        self.is_classifier = is_classifier
        
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
            GradientBoostingRegressor or GradientBoostingClassifier: Built model
        """
        print("Building Gradient Boosting model...")
        print(f"Model type: {'Classification' if self.is_classifier else 'Regression'}")
        
        # Create model
        if self.is_classifier:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state
            )
        else:
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
            f.write(f"=== Gradient Boosting {'Classification' if self.is_classifier else 'Regression'} Model Configuration ===\n\n")
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
        if self.is_classifier:
            model = GradientBoostingClassifier(random_state=42)
            scoring = 'accuracy'
        else:
            model = GradientBoostingRegressor(random_state=42)
            scoring = 'neg_mean_squared_error'
        
        # Create grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            verbose=1,
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Get best parameters
        self.best_params = grid_search.best_params_
        
        # Create model with best parameters
        if self.is_classifier:
            self.model = GradientBoostingClassifier(
                n_estimators=self.best_params['n_estimators'],
                learning_rate=self.best_params['learning_rate'],
                max_depth=self.best_params['max_depth'],
                subsample=self.best_params['subsample'],
                random_state=42
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=self.best_params['n_estimators'],
                learning_rate=self.best_params['learning_rate'],
                max_depth=self.best_params['max_depth'],
                subsample=self.best_params['subsample'],
                random_state=42
            )
        
        # Save best parameters
        with open('output/gradient_boosting/best_params.txt', 'w') as f:
            f.write(f"=== Gradient Boosting {'Classification' if self.is_classifier else 'Regression'} Best Parameters ===\n\n")
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
            GradientBoostingRegressor or GradientBoostingClassifier: Trained model
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
        if self.is_classifier:
            scoring = 'accuracy'
        else:
            scoring = 'neg_mean_squared_error'
            
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, X_train, y_train,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=cv, scoring=scoring
        )
        
        # Calculate mean and standard deviation
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # For regression, we need to negate the scores since they're negative MSE
        if not self.is_classifier:
            train_mean = -train_mean
            train_std = train_std
            test_mean = -test_mean
            test_std = test_std
        
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        plt.grid()
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
        plt.title('Learning Curve')
        plt.xlabel('Training Size')
        
        if self.is_classifier:
            plt.ylabel('Accuracy')
        else:
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
        
        if self.is_classifier:
            # Calculate classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)
            
            # Create metrics dictionary
            metrics = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': cm
            }
            
            # Print metrics
            print("\nGradient Boosting Classification Model Evaluation Metrics:")
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Visualize confusion matrix
            self.visualize_confusion_matrix(cm)
            
            # Save metrics to file
            with open('output/gradient_boosting/classification_metrics.txt', 'w') as f:
                f.write("=== Gradient Boosting Classification Model Evaluation Metrics ===\n\n")
                f.write(f"Accuracy: {accuracy:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(classification_report(y_test, y_pred))
            
            print("Saved classification metrics to output/gradient_boosting/classification_metrics.txt")
        else:
            # Calculate regression metrics
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
        plt.savefig('output/gradient_boosting/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved confusion matrix visualization to output/gradient_boosting/confusion_matrix.png")
    
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
    gb_model = GradientBoostingModel(is_classifier=False)
    gb_model.build_model()
    gb_model.train(X_train_dense, y_train)
    
    # Evaluate model
    metrics = gb_model.evaluate(X_test_dense, y_test)
    
    print("\nGradient Boosting model training and evaluation completed successfully!") 