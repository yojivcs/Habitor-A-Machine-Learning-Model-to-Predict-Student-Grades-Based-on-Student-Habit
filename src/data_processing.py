import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, data_path):
        """
        Initialize the DataProcessor with the path to the dataset
        
        Args:
            data_path (str): Path to the student habits dataset
        """
        self.data_path = data_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self):
        """Load the dataset from the provided path"""
        print(f"Loading data from {self.data_path}")
        self.data = pd.read_excel(self.data_path)
        return self.data
    
    def explore_data(self):
        """Explore the dataset and return basic statistics"""
        if self.data is None:
            self.load_data()
            
        # Basic information
        print("\nDataset Shape:", self.data.shape)
        print("\nColumns:", list(self.data.columns))
        print("\nData Types:")
        print(self.data.dtypes)
        
        # Check for missing values
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(self.data.describe())
        
        return self.data.head()
    
    def visualize_data(self):
        """Create basic visualizations of the dataset"""
        if self.data is None:
            self.load_data()
            
        # Set up the matplotlib figure
        plt.figure(figsize=(15, 10))
        
        # Correlation heatmap
        plt.subplot(2, 2, 1)
        correlation = self.data.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        
        # Study hours vs CGPA
        plt.subplot(2, 2, 2)
        sns.scatterplot(x='Study_Hours_Per_Day', y='CGPA', data=self.data)
        plt.title('Study Hours vs CGPA')
        
        # Sleep hours vs CGPA
        plt.subplot(2, 2, 3)
        sns.scatterplot(x='Sleep_Hours_Per_Day', y='CGPA', data=self.data)
        plt.title('Sleep Hours vs CGPA')
        
        # Stress level vs CGPA
        plt.subplot(2, 2, 4)
        sns.boxplot(x='Stress_Level', y='CGPA', data=self.data)
        plt.title('Stress Level vs CGPA')
        
        plt.tight_layout()
        plt.savefig('visualization.png')
        plt.close()
        
        # Additional visualizations
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Parental_Involvement', y='CGPA', data=self.data)
        plt.title('Parental Involvement vs CGPA')
        plt.savefig('parental_involvement.png')
        plt.close()
        
        print("Visualizations saved to disk")
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess the data for modeling
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        if self.data is None:
            self.load_data()
            
        # Make a copy to avoid modifying the original data
        processed_data = self.data.copy()
        
        # Encode categorical variables
        categorical_cols = ['Parental_Involvement', 'Stress_Level']
        for col in categorical_cols:
            if col in processed_data.columns:
                processed_data[col] = self.label_encoder.fit_transform(processed_data[col])
        
        # Separate features and target
        X = processed_data.drop(['Student_ID', 'CGPA'], axis=1)
        y = processed_data['CGPA']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Testing set: {self.X_test.shape[0]} samples")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_training_data(self):
        """Return the training data"""
        if self.X_train is None:
            raise ValueError("Data not preprocessed yet. Call preprocess_data() first.")
        return self.X_train, self.y_train
    
    def get_testing_data(self):
        """Return the testing data"""
        if self.X_test is None:
            raise ValueError("Data not preprocessed yet. Call preprocess_data() first.")
        return self.X_test, self.y_test 