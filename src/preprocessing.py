import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Update this function to match the provided grade sheet
def convert_cgpa_to_letter_grade(cgpa):
    """
    Convert CGPA/percentage to letter grade according to provided grade sheet
    
    Args:
        cgpa (float): CGPA or percentage value
        
    Returns:
        str: Letter grade
    """
    # Convert CGPA to percentage scale (assuming CGPA is on 4.0 scale)
    percentage = (cgpa / 4.0) * 100
    
    if percentage >= 80:
        return 'A+'
    elif percentage >= 75:
        return 'A'
    elif percentage >= 70:
        return 'A-'
    elif percentage >= 65:
        return 'B+'
    elif percentage >= 60:
        return 'B'
    elif percentage >= 55:
        return 'B-'
    elif percentage >= 50:
        return 'C+'
    elif percentage >= 45:
        return 'C'
    elif percentage >= 40:
        return 'D'
    else:
        return 'F'

class DataPreprocessor:
    def __init__(self, data, classification_mode=False):
        """
        Initialize the DataPreprocessor with the dataset
        
        Args:
            data (pandas.DataFrame): The student habits dataset
            classification_mode (bool): Whether to use classification mode
        """
        self.data = data.copy()  # Create a copy to avoid modifying the original
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_cols = []
        self.numerical_cols = []
        self.classification_mode = classification_mode
        self.grade_encoder = None
        
        # Create output directory if it doesn't exist
        os.makedirs('output/preprocessing', exist_ok=True)
    
    def identify_column_types(self):
        """
        Identify categorical and numerical columns
        """
        print("Identifying column types...")
        
        # Identify column types
        self.categorical_cols = []
        self.numerical_cols = []
        
        for col in self.data.columns:
            if col == 'Student_ID':
                continue  # Skip ID column
            elif col == 'CGPA':
                continue  # Skip target variable
            elif self.data[col].dtype == 'object' or self.data[col].nunique() < 10:
                # Consider columns with object dtype or few unique values as categorical
                self.categorical_cols.append(col)
            else:
                self.numerical_cols.append(col)
        
        print(f"Categorical columns: {self.categorical_cols}")
        print(f"Numerical columns: {self.numerical_cols}")
        
        return self.categorical_cols, self.numerical_cols
    
    def handle_missing_values(self):
        """
        Handle missing values in the dataset
        """
        print("Handling missing values...")
        
        # Get missing value counts before handling
        missing_before = self.data.isnull().sum()
        
        # Handle missing values (for categorical columns, fill with mode)
        for col in self.categorical_cols:
            if self.data[col].isnull().sum() > 0:
                mode_value = self.data[col].mode()[0]
                self.data[col].fillna(mode_value, inplace=True)
        
        # For numerical columns, fill with median
        for col in self.numerical_cols:
            if self.data[col].isnull().sum() > 0:
                median_value = self.data[col].median()
                self.data[col].fillna(median_value, inplace=True)
        
        # Get missing value counts after handling
        missing_after = self.data.isnull().sum()
        
        # Create summary of missing value handling
        with open('output/preprocessing/missing_values_summary.txt', 'w') as f:
            f.write("=== Missing Values Summary ===\n\n")
            f.write("Missing values before handling:\n")
            for col, count in missing_before.items():
                if count > 0:
                    f.write(f"- {col}: {count} missing values\n")
            
            f.write("\nMissing values after handling:\n")
            for col, count in missing_after.items():
                if count > 0:
                    f.write(f"- {col}: {count} missing values\n")
                else:
                    f.write(f"- {col}: No missing values\n")
        
        print("Saved missing values summary to output/preprocessing/missing_values_summary.txt")
        return self.data
    
    def encode_categorical_features(self):
        """
        Encode categorical features using Label Encoding
        """
        print("Encoding categorical features...")
        
        # Create a dictionary to store encoding mappings
        encoding_maps = {}
        
        # Encode each categorical column
        for col in self.categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
            self.label_encoders[col] = le
            
            # Store encoding mapping
            encoding_maps[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        
        # Save encoders for later use
        with open('output/preprocessing/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # Create summary of encoding mappings
        with open('output/preprocessing/encoding_mappings.txt', 'w') as f:
            f.write("=== Categorical Encoding Mappings ===\n\n")
            for col, mapping in encoding_maps.items():
                f.write(f"{col} encoding:\n")
                for original, encoded in mapping.items():
                    f.write(f"- {original} -> {encoded}\n")
                f.write("\n")
        
        # Visualize encoded categorical features
        self.visualize_encoded_features()
        
        print("Saved encoding mappings to output/preprocessing/encoding_mappings.txt")
        print("Saved label encoders to output/preprocessing/label_encoders.pkl")
        return self.data
    
    def visualize_encoded_features(self):
        """
        Visualize the distribution of encoded categorical features
        """
        if not self.categorical_cols:
            return
        
        plt.figure(figsize=(12, 4 * len(self.categorical_cols)))
        
        for i, col in enumerate(self.categorical_cols):
            plt.subplot(len(self.categorical_cols), 1, i + 1)
            sns.countplot(x=col, data=self.data)
            plt.title(f'Distribution of Encoded {col}')
            plt.xlabel(f'{col} (Encoded)')
            plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('output/preprocessing/encoded_features_distribution.png')
        plt.close()
        print("Saved encoded features distribution to output/preprocessing/encoded_features_distribution.png")
    
    def scale_numerical_features(self):
        """
        Scale numerical features using StandardScaler
        """
        print("Scaling numerical features...")
        
        # Extract numerical features
        numerical_data = self.data[self.numerical_cols]
        
        # Fit scaler and transform data
        scaled_data = self.scaler.fit_transform(numerical_data)
        
        # Replace original data with scaled data
        for i, col in enumerate(self.numerical_cols):
            self.data[col] = scaled_data[:, i]
        
        # Save scaler for later use
        with open('output/preprocessing/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Visualize the effect of scaling
        self.visualize_scaling_effect(numerical_data, scaled_data)
        
        print("Saved scaler to output/preprocessing/scaler.pkl")
        return self.data
    
    def visualize_scaling_effect(self, original_data, scaled_data):
        """
        Visualize the effect of scaling on a sample feature
        
        Args:
            original_data (pandas.DataFrame): Original numerical data
            scaled_data (numpy.ndarray): Scaled numerical data
        """
        if not self.numerical_cols:
            return
        
        # Select first numerical column for visualization
        sample_col = self.numerical_cols[0]
        sample_idx = 0
        
        plt.figure(figsize=(12, 6))
        
        # Original data distribution
        plt.subplot(1, 2, 1)
        sns.histplot(original_data[sample_col], kde=True)
        plt.title(f'Original {sample_col} Distribution')
        plt.xlabel(sample_col)
        
        # Scaled data distribution
        plt.subplot(1, 2, 2)
        sns.histplot(scaled_data[:, sample_idx], kde=True)
        plt.title(f'Scaled {sample_col} Distribution')
        plt.xlabel(f'{sample_col} (Scaled)')
        
        plt.tight_layout()
        plt.savefig('output/preprocessing/scaling_effect.png')
        plt.close()
        print("Saved scaling effect visualization to output/preprocessing/scaling_effect.png")
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        print("Splitting data into training and testing sets...")
        
        # Separate features and target
        X = self.data.drop(['Student_ID', 'CGPA'], axis=1, errors='ignore')
        y = self.data['CGPA']
        
        # If in classification mode, convert CGPA to letter grades
        if self.classification_mode:
            print("Converting CGPA to letter grades for classification...")
            
            # Convert to letter grades
            y_grades = y.apply(convert_cgpa_to_letter_grade)
            
            # Encode letter grades
            self.grade_encoder = LabelEncoder()
            y_encoded = self.grade_encoder.fit_transform(y_grades)
            
            # Save grade encoding mapping
            grade_mapping = dict(zip(self.grade_encoder.classes_, range(len(self.grade_encoder.classes_))))
            with open('output/preprocessing/grade_mapping.txt', 'w') as f:
                f.write("=== Grade Encoding Mapping ===\n\n")
                for grade, code in grade_mapping.items():
                    f.write(f"{grade} -> {code}\n")
            
            # Save grade encoder
            with open('output/preprocessing/grade_encoder.pkl', 'wb') as f:
                pickle.dump(self.grade_encoder, f)
            
            print(f"Encoded letter grades: {grade_mapping}")
            print("Saved grade mapping to output/preprocessing/grade_mapping.txt")
            
            # Use encoded grades as target
            y = y_encoded
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
        
        # Save split data for later use
        train_data = pd.concat([self.X_train, pd.Series(self.y_train, index=self.X_train.index, name='Target')], axis=1)
        test_data = pd.concat([self.X_test, pd.Series(self.y_test, index=self.X_test.index, name='Target')], axis=1)
        
        train_data.to_csv('output/preprocessing/train_data.csv', index=False)
        test_data.to_csv('output/preprocessing/test_data.csv', index=False)
        
        # Visualize train-test split distribution
        self.visualize_train_test_split()
        
        print("Saved training data to output/preprocessing/train_data.csv")
        print("Saved testing data to output/preprocessing/test_data.csv")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def visualize_train_test_split(self):
        """
        Visualize the distribution of target variable in train and test sets
        """
        plt.figure(figsize=(10, 6))
        
        if self.classification_mode:
            # For classification mode, show class distribution
            train_counts = pd.Series(self.y_train).value_counts().sort_index()
            test_counts = pd.Series(self.y_test).value_counts().sort_index()
            
            # Get all unique classes
            all_classes = sorted(set(train_counts.index) | set(test_counts.index))
            
            # Create bar chart for train data
            plt.subplot(1, 2, 1)
            sns.barplot(x=train_counts.index, y=train_counts.values, color='blue')
            plt.title('Training Set Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(range(len(all_classes)), [self.grade_encoder.inverse_transform([c])[0] for c in all_classes])
            
            # Create bar chart for test data
            plt.subplot(1, 2, 2)
            sns.barplot(x=test_counts.index, y=test_counts.values, color='green')
            plt.title('Testing Set Class Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(range(len(all_classes)), [self.grade_encoder.inverse_transform([c])[0] for c in all_classes])
        else:
            # For regression mode, show CGPA distribution
            plt.subplot(1, 2, 1)
            sns.histplot(self.y_train, kde=True, color='blue')
            plt.title('Training Set CGPA Distribution')
            plt.xlabel('CGPA')
            
            plt.subplot(1, 2, 2)
            sns.histplot(self.y_test, kde=True, color='green')
            plt.title('Testing Set CGPA Distribution')
            plt.xlabel('CGPA')
        
        plt.tight_layout()
        plt.savefig('output/preprocessing/train_test_distribution.png')
        plt.close()
        print("Saved train-test distribution visualization to output/preprocessing/train_test_distribution.png")
    
    def preprocess_data(self):
        """
        Run the full preprocessing pipeline
        """
        print("\n=== Starting Data Preprocessing ===")
        print(f"Mode: {'Classification' if self.classification_mode else 'Regression'}")
        
        # Step 1: Identify column types
        self.identify_column_types()
        
        # Step 2: Handle missing values
        self.handle_missing_values()
        
        # Step 3: Encode categorical features
        self.encode_categorical_features()
        
        # Step 4: Scale numerical features
        self.scale_numerical_features()
        
        # Step 5: Split data
        self.split_data()
        
        print("\nPreprocessing completed successfully!")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_preprocessed_data(self):
        """
        Get the preprocessed data
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.X_train is None:
            print("Data not preprocessed yet. Call preprocess_data() first.")
            return None, None, None, None
        
        return self.X_train, self.X_test, self.y_train, self.y_test


# For testing the module independently
if __name__ == "__main__":
    from data_collection import DataCollector
    
    # Load data
    collector = DataCollector("Reports and Dataset/student_habits_dataset.xlsx")
    data = collector.load_data()
    
    # Preprocess data
    preprocessor = DataPreprocessor(data)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data()
    print("\nData preprocessing completed successfully!") 