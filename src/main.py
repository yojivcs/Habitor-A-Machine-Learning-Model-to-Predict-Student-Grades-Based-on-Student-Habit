import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import argparse

# Import all our modules
from data_collection import DataCollector
from data_visualization import DataVisualizer
from preprocessing import DataPreprocessor, convert_cgpa_to_letter_grade
from count_vectorization import FeatureVectorizer
from bilstm_model import BiLSTMModel
from gradient_boosting_model import GradientBoostingModel
from future_prediction import FuturePredictor
from model_evaluation import ModelEvaluator

class HabitorPipeline:
    def __init__(self, data_path, output_dir="output", classification_mode=False):
        """
        Initialize the Habitor Pipeline
        
        Args:
            data_path (str): Path to the student habits dataset
            output_dir (str): Path to the output directory
            classification_mode (bool): Whether to use classification mode
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.classification_mode = classification_mode
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.collector = None
        self.visualizer = None
        self.preprocessor = None
        self.vectorizer = None
        self.bilstm_model = None
        self.gb_model = None
        self.predictor = None
        self.evaluator = None
        
        # Data containers
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_vectorized = None
        self.X_test_vectorized = None
        
        print(f"Habitor Pipeline initialized with data path: {data_path}")
        print(f"Mode: {'Classification' if self.classification_mode else 'Regression'}")
    
    def collect_data(self):
        """
        Collect and load data
        """
        print("\n=== Step 1: Data Collection ===")
        
        self.collector = DataCollector(self.data_path)
        self.data = self.collector.load_data()
        self.collector.get_data_summary()
        
        return self.data
    
    def visualize_data(self):
        """
        Visualize the data
        """
        print("\n=== Step 2: Data Visualization ===")
        
        if self.data is None:
            self.collect_data()
        
        self.visualizer = DataVisualizer(self.data)
        self.visualizer.visualize_all()
        
        return self.visualizer
    
    def preprocess_data(self):
        """
        Preprocess the data
        """
        print("\n=== Step 3: Data Preprocessing ===")
        
        if self.data is None:
            self.collect_data()
        
        self.preprocessor = DataPreprocessor(self.data, classification_mode=self.classification_mode)
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocessor.preprocess_data()
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def vectorize_features(self):
        """
        Vectorize features
        """
        print("\n=== Step 4: Count Vectorization ===")
        
        if self.X_train is None:
            self.preprocess_data()
        
        self.vectorizer = FeatureVectorizer(self.X_train, self.X_test)
        self.X_train_vectorized, self.X_test_vectorized = self.vectorizer.vectorize_and_combine()
        
        return self.X_train_vectorized, self.X_test_vectorized
    
    def train_bilstm_model(self, epochs=50):
        """
        Train the BI-LSTM model
        
        Args:
            epochs (int): Number of training epochs
        """
        print("\n=== Step 5: BI-LSTM Model Training ===")
        
        if self.X_train_vectorized is None:
            self.vectorize_features()
        
        # Convert to dense arrays for LSTM
        X_train_dense = self.X_train_vectorized.toarray() if hasattr(self.X_train_vectorized, 'toarray') else self.X_train_vectorized
        X_test_dense = self.X_test_vectorized.toarray() if hasattr(self.X_test_vectorized, 'toarray') else self.X_test_vectorized
        
        # Build and train BI-LSTM model
        if self.classification_mode:
            # Get number of classes for classification
            num_classes = len(np.unique(self.y_train))
            print(f"Training classification model with {num_classes} classes")
            
            self.bilstm_model = BiLSTMModel(X_train_dense.shape[1], is_classifier=True, num_classes=num_classes)
        else:
            self.bilstm_model = BiLSTMModel(X_train_dense.shape[1])
            
        self.bilstm_model.build_model()
        self.bilstm_model.train(X_train_dense, self.y_train, X_test_dense, self.y_test, epochs=epochs)
        
        # Evaluate model
        metrics = self.bilstm_model.evaluate(X_test_dense, self.y_test)
        
        return self.bilstm_model, metrics
    
    def train_gradient_boosting_model(self, hyperparameter_tuning=False):
        """
        Train the Gradient Boosting model
        
        Args:
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        """
        print("\n=== Step 6: Gradient Boosting Model Training ===")
        
        if self.X_train_vectorized is None:
            self.vectorize_features()
        
        # Convert to dense arrays if needed
        X_train_dense = self.X_train_vectorized.toarray() if hasattr(self.X_train_vectorized, 'toarray') else self.X_train_vectorized
        X_test_dense = self.X_test_vectorized.toarray() if hasattr(self.X_test_vectorized, 'toarray') else self.X_test_vectorized
        
        # Build and train Gradient Boosting model
        self.gb_model = GradientBoostingModel(is_classifier=self.classification_mode)
        
        if hyperparameter_tuning:
            self.gb_model.hyperparameter_tuning(X_train_dense, self.y_train)
        else:
            self.gb_model.build_model()
            
        self.gb_model.train(X_train_dense, self.y_train)
        
        # Evaluate model
        metrics = self.gb_model.evaluate(X_test_dense, self.y_test)
        
        return self.gb_model, metrics
    
    def future_prediction(self):
        """
        Make future predictions with trained models
        """
        print("\n=== Step 7: Future Prediction ===")
        
        if self.bilstm_model is None or self.gb_model is None:
            print("Models not trained yet. Training models...")
            self.train_bilstm_model(epochs=10)
            self.train_gradient_boosting_model()
        
        # Convert to dense arrays if needed
        X_test_dense = self.X_test_vectorized.toarray() if hasattr(self.X_test_vectorized, 'toarray') else self.X_test_vectorized
        
        # Create future predictor
        self.predictor = FuturePredictor(
            bilstm_model=self.bilstm_model.model,
            gradient_model=self.gb_model.model,
            is_classifier=self.classification_mode
        )
        
        # Make predictions
        predictions = self.predictor.predict_all(X_test_dense, self.y_test)
        
        # Generate prediction report
        self.predictor.generate_prediction_report()
        
        return self.predictor, predictions
    
    def model_evaluation(self):
        """
        Evaluate all models
        """
        print("\n=== Step 8: Model Evaluation ===")
        
        if self.bilstm_model is None or self.gb_model is None:
            print("Models not trained yet. Training models...")
            self.train_bilstm_model(epochs=10)
            self.train_gradient_boosting_model()
        
        # Convert to dense arrays if needed
        X_test_dense = self.X_test_vectorized.toarray() if hasattr(self.X_test_vectorized, 'toarray') else self.X_test_vectorized
        
        # Create model evaluator
        self.evaluator = ModelEvaluator()
        
        # Add models to evaluator
        self.evaluator.add_model('BI_LSTM', self.bilstm_model.model, is_classifier=self.classification_mode)
        self.evaluator.add_model('Gradient_Boosting', self.gb_model.model, is_classifier=self.classification_mode)
        
        # Evaluate models
        metrics = self.evaluator.evaluate_all(X_test_dense, self.y_test)
        
        return self.evaluator, metrics
    
    def run_full_pipeline(self, epochs=50, hyperparameter_tuning=False):
        """
        Run the full pipeline
        
        Args:
            epochs (int): Number of training epochs for BI-LSTM
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning for Gradient Boosting
        """
        print("\n============================================")
        print("         HABITOR PIPELINE EXECUTION         ")
        print("============================================")
        print("Starting the full Habitor pipeline for student grade prediction")
        print(f"Mode: {'Classification' if self.classification_mode else 'Regression'}")
        print("--------------------------------------------")
        
        # Record start time
        start_time = time.time()
        
        # Step 1: Data Collection
        self.collect_data()
        
        # Step 2: Data Visualization
        self.visualize_data()
        
        # Step 3: Data Preprocessing
        self.preprocess_data()
        
        # Step 4: Count Vectorization
        self.vectorize_features()
        
        # Step 5: BI-LSTM Model Training
        self.train_bilstm_model(epochs=epochs)
        
        # Step 6: Gradient Boosting Model Training
        self.train_gradient_boosting_model(hyperparameter_tuning=hyperparameter_tuning)
        
        # Step 7: Future Prediction
        self.future_prediction()
        
        # Step 8: Model Evaluation
        self.model_evaluation()
        
        # Record end time
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print summary
        print("\n============================================")
        print("         PIPELINE EXECUTION SUMMARY         ")
        print("============================================")
        print(f"Mode: {'Classification' if self.classification_mode else 'Regression'}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"All outputs saved to: {self.output_dir}/")
        print("--------------------------------------------")
        
        if self.evaluator and self.evaluator.best_model:
            print(f"Best model: {self.evaluator.best_model}")
            
            if self.classification_mode:
                metrics = self.evaluator.metrics[self.evaluator.best_model]
                print(f"Accuracy: {metrics['accuracy']:.4f}")
            else:
                if self.evaluator.best_model == 'BI_LSTM':
                    metrics = self.evaluator.metrics['BI_LSTM']
                    print(f"RMSE: {metrics['rmse']:.4f}")
                    print(f"R2 Score: {metrics['r2']:.4f}")
                else:
                    metrics = self.evaluator.metrics['Gradient_Boosting']
                    print(f"RMSE: {metrics['rmse']:.4f}")
                    print(f"R2 Score: {metrics['r2']:.4f}")
        
        print("\nHabitor pipeline completed successfully!")
        print("============================================")
        
        return {
            "data": self.data,
            "X_train": self.X_train,
            "X_test": self.X_test,
            "y_train": self.y_train,
            "y_test": self.y_test,
            "bilstm_model": self.bilstm_model,
            "gb_model": self.gb_model,
            "predictor": self.predictor,
            "evaluator": self.evaluator,
            "execution_time": total_time
        }


def main():
    """
    Main function to run the Habitor pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Habitor: Predict Student Grades Based on Habits')
    parser.add_argument('--data', type=str, default="student_habits_dataset.csv",
                        help='Path to the student habits dataset')
    parser.add_argument('--output', type=str, default="output",
                        help='Path to the output directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for BI-LSTM training')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning for Gradient Boosting')
    parser.add_argument('--classification', action='store_true',
                        help='Run in classification mode (predict letter grades instead of CGPA)')
    parser.add_argument('--steps', type=str, nargs='+', 
                        choices=['data', 'visualization', 'preprocessing', 'vectorization', 
                                'bilstm', 'gradient', 'prediction', 'evaluation', 'all'],
                        default=['all'],
                        help='Specify which steps to run')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = HabitorPipeline(args.data, args.output, classification_mode=args.classification)
    
    # Run specified steps or full pipeline
    if 'all' in args.steps:
        pipeline.run_full_pipeline(epochs=args.epochs, hyperparameter_tuning=args.tune)
    else:
        if 'data' in args.steps:
            pipeline.collect_data()
        
        if 'visualization' in args.steps:
            pipeline.visualize_data()
        
        if 'preprocessing' in args.steps:
            pipeline.preprocess_data()
        
        if 'vectorization' in args.steps:
            pipeline.vectorize_features()
        
        if 'bilstm' in args.steps:
            pipeline.train_bilstm_model(epochs=args.epochs)
        
        if 'gradient' in args.steps:
            pipeline.train_gradient_boosting_model(hyperparameter_tuning=args.tune)
        
        if 'prediction' in args.steps:
            pipeline.future_prediction()
        
        if 'evaluation' in args.steps:
            pipeline.model_evaluation()
    
    print("Done!")


if __name__ == "__main__":
    main() 