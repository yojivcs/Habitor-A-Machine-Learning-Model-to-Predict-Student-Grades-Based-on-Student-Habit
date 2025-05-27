import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# Define DummyVectorizer class at module level so it can be pickled
class DummyVectorizer:
    def __init__(self, feature_names):
        self.feature_names = feature_names
    
    def get_feature_names_out(self):
        return self.feature_names

class FeatureVectorizer:
    def __init__(self, X_train, X_test):
        """
        Initialize the FeatureVectorizer with training and testing data
        
        Args:
            X_train (pandas.DataFrame): Training features
            X_test (pandas.DataFrame): Testing features
        """
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.vectorizers = {}
        self.X_train_vectorized = None
        self.X_test_vectorized = None
        
        # Create output directory if it doesn't exist
        os.makedirs('output/vectorization', exist_ok=True)
    
    def convert_categorical_to_text(self):
        """
        Convert categorical features to text format for vectorization
        
        Returns:
            tuple: (X_train_text, X_test_text) dictionaries with column names as keys and text data as values
        """
        print("Converting categorical features to text format...")
        
        # Identify categorical columns
        categorical_cols = []
        for col in self.X_train.columns:
            if self.X_train[col].dtype == 'object' or self.X_train[col].nunique() < 10:
                categorical_cols.append(col)
        
        # If no categorical columns found, use all columns
        if not categorical_cols:
            print("No categorical columns found. Using all columns for vectorization.")
            categorical_cols = list(self.X_train.columns)
        
        print(f"Columns for vectorization: {categorical_cols}")
        
        # Convert categorical features to text for train set
        X_train_text = {}
        for col in categorical_cols:
            X_train_text[col] = self.X_train[col].astype(str)
        
        # Convert categorical features to text for test set
        X_test_text = {}
        for col in categorical_cols:
            X_test_text[col] = self.X_test[col].astype(str)
        
        return X_train_text, X_test_text
    
    def vectorize_features(self):
        """
        Apply count vectorization to the features
        
        Returns:
            tuple: (X_train_vectorized, X_test_vectorized) - Vectorized feature matrices
        """
        print("Vectorizing features...")
        
        # Convert categorical features to text
        X_train_text, X_test_text = self.convert_categorical_to_text()
        
        # Initialize dictionaries to store vectorized features
        X_train_vectorized_dict = {}
        X_test_vectorized_dict = {}
        
        # Vectorize each column
        for col in X_train_text:
            print(f"Vectorizing column: {col}")
            
            try:
                # Create and fit vectorizer with min_df=1 (include all terms that appear at least once)
                vectorizer = CountVectorizer(binary=True, min_df=1)
                X_train_col_vectorized = vectorizer.fit_transform(X_train_text[col])
                
                # Transform test data
                X_test_col_vectorized = vectorizer.transform(X_test_text[col])
                
                # Store vectorizer
                self.vectorizers[col] = vectorizer
                
                # Store vectorized features
                X_train_vectorized_dict[col] = X_train_col_vectorized
                X_test_vectorized_dict[col] = X_test_col_vectorized
                
                # Generate visualization
                self.visualize_vectorization(col, vectorizer, X_train_col_vectorized)
            except ValueError as e:
                print(f"Warning: Could not vectorize column {col}. Error: {e}")
                print(f"Using one-hot encoding for column {col} instead.")
                
                # Fallback to one-hot encoding
                X_train_dummies = pd.get_dummies(self.X_train[col], prefix=col)
                X_test_dummies = pd.get_dummies(self.X_test[col], prefix=col)
                
                # Ensure test has same columns as train
                for column in X_train_dummies.columns:
                    if column not in X_test_dummies.columns:
                        X_test_dummies[column] = 0
                
                # Keep only columns in train
                X_test_dummies = X_test_dummies[X_train_dummies.columns]
                
                # Convert to sparse matrices
                from scipy.sparse import csr_matrix
                X_train_col_vectorized = csr_matrix(X_train_dummies.values)
                X_test_col_vectorized = csr_matrix(X_test_dummies.values)
                
                # Store feature names as a custom vectorizer (using class defined at module level)
                dummy_vectorizer = DummyVectorizer(X_train_dummies.columns.tolist())
                self.vectorizers[col] = dummy_vectorizer
                
                # Store vectorized features
                X_train_vectorized_dict[col] = X_train_col_vectorized
                X_test_vectorized_dict[col] = X_test_col_vectorized
                
                # Generate visualization for one-hot encoding
                self.visualize_vectorization(col, dummy_vectorizer, X_train_col_vectorized)
        
        # Save vectorizers for later use
        with open('output/vectorization/vectorizers.pkl', 'wb') as f:
            pickle.dump(self.vectorizers, f)
        
        # Get numerical columns (not vectorized)
        numerical_cols = [col for col in self.X_train.columns if col not in X_train_text]
        
        # Create feature information summary
        with open('output/vectorization/vectorization_summary.txt', 'w') as f:
            f.write("=== Vectorization Summary ===\n\n")
            f.write("Vectorized columns:\n")
            for col in X_train_text:
                f.write(f"- {col}: {X_train_vectorized_dict[col].shape[1]} features\n")
            
            f.write("\nNumerical columns (not vectorized):\n")
            for col in numerical_cols:
                f.write(f"- {col}\n")
        
        print("Saved vectorizers to output/vectorization/vectorizers.pkl")
        print("Saved vectorization summary to output/vectorization/vectorization_summary.txt")
        
        # Store vectorized data
        self.X_train_vectorized = X_train_vectorized_dict
        self.X_test_vectorized = X_test_vectorized_dict
        
        return self.X_train_vectorized, self.X_test_vectorized
    
    def visualize_vectorization(self, column, vectorizer, vectorized_data):
        """
        Visualize the vectorization of a column
        
        Args:
            column (str): Column name
            vectorizer (CountVectorizer): The vectorizer used
            vectorized_data (scipy.sparse.csr_matrix): Vectorized data
        """
        try:
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Count feature occurrences
            feature_counts = np.asarray(vectorized_data.sum(axis=0)).flatten()
            
            # Create a DataFrame for visualization
            features_df = pd.DataFrame({
                'feature': feature_names,
                'count': feature_counts
            })
            
            # Sort by count (descending)
            features_df = features_df.sort_values('count', ascending=False)
            
            # Take top 20 features (or all if less than 20)
            top_n = min(20, len(features_df))
            top_features = features_df.head(top_n)
            
            # Plot
            plt.figure(figsize=(12, 6))
            sns.barplot(x='count', y='feature', data=top_features)
            plt.title(f'Top {top_n} Features from Vectorization of {column}')
            plt.xlabel('Count')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(f'output/vectorization/vectorization_{column}.png')
            plt.close()
            print(f"Saved vectorization visualization for {column} to output/vectorization/vectorization_{column}.png")
        except Exception as e:
            print(f"Warning: Could not create visualization for column {column}. Error: {e}")
    
    def combine_vectorized_features(self):
        """
        Combine all vectorized features into a single matrix
        
        Returns:
            tuple: (X_train_combined, X_test_combined) - Combined feature matrices
        """
        print("Combining vectorized features...")
        
        if self.X_train_vectorized is None:
            print("Features not vectorized yet. Call vectorize_features() first.")
            return None, None
        
        # Extract numerical features from original data
        numerical_cols = [col for col in self.X_train.columns if col not in self.X_train_vectorized]
        
        # Create lists to store features to combine
        X_train_features = []
        X_test_features = []
        
        # Add vectorized features
        for col in self.X_train_vectorized:
            X_train_features.append(self.X_train_vectorized[col])
            X_test_features.append(self.X_test_vectorized[col])
        
        # Add numerical features
        if numerical_cols:
            X_train_numerical = self.X_train[numerical_cols].values
            X_test_numerical = self.X_test[numerical_cols].values
            
            X_train_features.append(X_train_numerical)
            X_test_features.append(X_test_numerical)
        
        # Combine all features into a single matrix
        from scipy.sparse import hstack, csr_matrix
        
        # Convert all features to sparse format
        X_train_features_sparse = [csr_matrix(f) if not hasattr(f, 'toarray') else f for f in X_train_features]
        X_test_features_sparse = [csr_matrix(f) if not hasattr(f, 'toarray') else f for f in X_test_features]
        
        # Combine features
        X_train_combined = hstack(X_train_features_sparse)
        X_test_combined = hstack(X_test_features_sparse)
        
        print(f"Combined training features shape: {X_train_combined.shape}")
        print(f"Combined testing features shape: {X_test_combined.shape}")
        
        # Save combined features
        with open('output/vectorization/X_train_vectorized.pkl', 'wb') as f:
            pickle.dump(X_train_combined, f)
        
        with open('output/vectorization/X_test_vectorized.pkl', 'wb') as f:
            pickle.dump(X_test_combined, f)
        
        print("Saved combined vectorized features to output/vectorization/")
        
        return X_train_combined, X_test_combined
    
    def vectorize_and_combine(self):
        """
        Run the full vectorization pipeline
        
        Returns:
            tuple: (X_train_combined, X_test_combined) - Combined feature matrices
        """
        print("\n=== Starting Feature Vectorization ===")
        
        # Step 1: Vectorize features
        self.vectorize_features()
        
        # Step 2: Combine vectorized features
        X_train_combined, X_test_combined = self.combine_vectorized_features()
        
        print("\nFeature vectorization completed successfully!")
        return X_train_combined, X_test_combined


# For testing the module independently
if __name__ == "__main__":
    from data_collection import DataCollector
    from preprocessing import DataPreprocessor
    
    # Load data
    collector = DataCollector("Reports and Dataset/student_habits_dataset.xlsx")
    data = collector.load_data()
    
    # Preprocess data
    preprocessor = DataPreprocessor(data)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data()
    
    # Vectorize features
    vectorizer = FeatureVectorizer(X_train, X_test)
    X_train_vectorized, X_test_vectorized = vectorizer.vectorize_and_combine()
    
    print("\nCount vectorization completed successfully!") 