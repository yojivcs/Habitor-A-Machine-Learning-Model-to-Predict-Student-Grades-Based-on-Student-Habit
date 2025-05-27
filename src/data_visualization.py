import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class DataVisualizer:
    def __init__(self, data):
        """
        Initialize the DataVisualizer with the dataset
        
        Args:
            data (pandas.DataFrame): The student habits dataset
        """
        self.data = data
        
        # Create output directory if it doesn't exist
        os.makedirs('output/visualizations', exist_ok=True)
    
    def visualize_all(self):
        """
        Create all visualizations at once
        """
        self.correlation_heatmap()
        self.feature_distributions()
        self.feature_vs_cgpa()
        self.categorical_feature_analysis()
        self.pairplot_key_features()
        print("\nAll visualizations completed successfully!")
    
    def correlation_heatmap(self):
        """
        Create a correlation heatmap of all numerical features
        """
        print("Creating correlation heatmap...")
        plt.figure(figsize=(12, 10))
        
        # Get numerical columns
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        
        # Create correlation matrix
        corr = self.data[numerical_cols].corr()
        
        # Plot heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', mask=mask)
        plt.title('Correlation Heatmap of Student Habits Features', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/visualizations/correlation_heatmap.png')
        plt.close()
        print("Saved correlation heatmap to output/visualizations/correlation_heatmap.png")
    
    def feature_distributions(self):
        """
        Create histograms of all numerical features
        """
        print("Creating feature distributions...")
        
        # Get numerical columns
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove Student_ID if present
        if 'Student_ID' in numerical_cols:
            numerical_cols.remove('Student_ID')
        
        # Calculate number of rows and columns for subplot grid
        n_features = len(numerical_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten()
        
        # Plot each feature
        for i, col in enumerate(numerical_cols):
            sns.histplot(self.data[col], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/visualizations/feature_distributions.png')
        plt.close()
        print("Saved feature distributions to output/visualizations/feature_distributions.png")
    
    def feature_vs_cgpa(self):
        """
        Create scatter plots of each feature vs CGPA
        """
        print("Creating feature vs CGPA scatter plots...")
        
        # Get numerical columns (excluding CGPA and Student_ID)
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove Student_ID and CGPA if present
        for col in ['Student_ID', 'CGPA']:
            if col in numerical_cols:
                numerical_cols.remove(col)
        
        # Calculate number of rows and columns for subplot grid
        n_features = len(numerical_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten()
        
        # Plot each feature vs CGPA
        for i, col in enumerate(numerical_cols):
            sns.regplot(x=col, y='CGPA', data=self.data, ax=axes[i], scatter_kws={'alpha': 0.5})
            axes[i].set_title(f'{col} vs CGPA')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('CGPA')
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/visualizations/feature_vs_cgpa.png')
        plt.close()
        print("Saved feature vs CGPA scatter plots to output/visualizations/feature_vs_cgpa.png")
    
    def categorical_feature_analysis(self):
        """
        Create bar plots for categorical features vs CGPA
        """
        print("Creating categorical feature analysis...")
        
        # Identify categorical columns (or columns that should be treated as categorical)
        categorical_cols = ['Stress_Level', 'Parental_Involvement']
        
        # Filter only the categorical columns that exist in the dataset
        categorical_cols = [col for col in categorical_cols if col in self.data.columns]
        
        if not categorical_cols:
            print("No categorical features found for analysis")
            return
        
        # Create subplots
        fig, axes = plt.subplots(1, len(categorical_cols), figsize=(15, 6))
        
        # Handle case with only one categorical column
        if len(categorical_cols) == 1:
            axes = [axes]
        
        # Plot each categorical feature
        for i, col in enumerate(categorical_cols):
            sns.boxplot(x=col, y='CGPA', data=self.data, ax=axes[i])
            axes[i].set_title(f'{col} vs CGPA')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('CGPA')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/visualizations/categorical_feature_analysis.png')
        plt.close()
        print("Saved categorical feature analysis to output/visualizations/categorical_feature_analysis.png")
    
    def pairplot_key_features(self):
        """
        Create a pairplot of key features
        """
        print("Creating pairplot of key features...")
        
        # Select key features (modify as needed)
        key_features = ['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Screen_Time_Per_Day', 'CGPA']
        
        # Filter only the key features that exist in the dataset
        key_features = [col for col in key_features if col in self.data.columns]
        
        if len(key_features) < 2:
            print("Not enough key features found for pairplot")
            return
        
        # Create pairplot
        sns.pairplot(self.data[key_features], height=2.5)
        plt.suptitle('Pairplot of Key Features', y=1.02, fontsize=16)
        plt.tight_layout()
        
        # Save figure
        plt.savefig('output/visualizations/pairplot_key_features.png')
        plt.close()
        print("Saved pairplot of key features to output/visualizations/pairplot_key_features.png")


# For testing the module independently
if __name__ == "__main__":
    from data_collection import DataCollector
    
    # Load data
    collector = DataCollector("Reports and Dataset/student_habits_dataset.xlsx")
    data = collector.load_data()
    
    # Create visualizations
    visualizer = DataVisualizer(data)
    visualizer.visualize_all()
    print("\nData visualization completed successfully!") 