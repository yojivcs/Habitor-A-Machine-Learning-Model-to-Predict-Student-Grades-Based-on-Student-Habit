# Habitor - Student Grade Prediction Based on Habits

Habitor is an intelligent system that predicts student academic performance (CGPA) based on their habits and behaviors. The application uses a Bidirectional LSTM neural network and Gradient Boosting models to analyze the relationship between student habits and academic outcomes, providing personalized recommendations for improvement.

![Habitor Preview](https://images.unsplash.com/photo-1501504905252-473c47e087f8?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80)

## Features

- **CGPA Prediction**: Predict student academic performance based on various habits and behaviors
- **Dual Model Approach**: Utilizes both BI-LSTM neural networks and Gradient Boosting for high accuracy
- **Ensemble Predictions**: Combines multiple model outputs for more robust predictions
- **Personalized Recommendations**: Get tailored suggestions to improve academic performance
- **Interactive Data Visualization**: Explore visualizations showing relationships between habits and performance
- **Department-Specific Analysis**: See how success factors vary across different academic departments
- **Model Evaluation**: Comprehensive performance metrics and comparisons between models
- **Responsive Web Interface**: Clean, modern UI with dark/light mode toggle

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/habitor.git
   cd habitor
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

Run the main ML pipeline:

```
python main.py
```

Or run the Streamlit web application:

```
streamlit run app.py
```

The web application will open in your default web browser at `http://localhost:8501`.

## Project Structure

```
habitor/
├── app.py                      # Main Streamlit web application
├── predict_page.py             # CGPA prediction page
├── analysis_page.py            # Data analysis and visualization page
├── main.py                     # Entry point for ML pipeline execution
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── habitor_project_report.html # Detailed project report
├── flow_chart.txt              # ML pipeline flow chart
├── src/                        # Source code for ML pipeline
│   ├── main.py                 # Main orchestrator for ML pipeline
│   ├── data_collection.py      # Data loading and initial processing
│   ├── data_visualization.py   # Data visualization components
│   ├── preprocessing.py        # Data preprocessing and feature engineering
│   ├── count_vectorization.py  # Feature vectorization for categorical data
│   ├── bilstm_model.py         # Bidirectional LSTM neural network model
│   ├── gradient_boosting_model.py # Gradient Boosting model implementation
│   ├── future_prediction.py    # Prediction module for trained models
│   └── model_evaluation.py     # Model evaluation and comparison tools
├── output/                     # Generated outputs, visualizations, and reports
└── images/                     # Images for web interface and documentation
```

## ML Pipeline Flow

The Habitor ML pipeline follows a structured flow:

1. **Data Collection**: Loading student data from CSV files
2. **Data Visualization**: Creating exploratory visualizations
3. **Preprocessing**: Handling missing values, encoding, scaling, and splitting
4. **Feature Vectorization**: Converting categorical data to numerical format
5. **BI-LSTM Model**: Training a Bidirectional LSTM neural network
6. **Gradient Boosting Model**: Training a Gradient Boosting model
7. **Future Prediction**: Making predictions with both models
8. **Model Evaluation**: Comparing model performance and generating reports

## Usage

### ML Pipeline

Run the ML pipeline for model training and evaluation:

```
python main.py
```

This will execute the full machine learning pipeline and generate output visualizations and reports in the `output` directory.

### Web Interface

1. **Predict CGPA**:
   - Fill out the form with your habits and behaviors
   - Click "Predict My CGPA" to get your predicted academic performance
   - Review personalized recommendations and habit impact analysis

2. **Explore Data Analysis**:
   - Navigate to the "Data Analysis" tab
   - Explore visualizations showing correlations between habits and performance
   - Review department-specific insights and key success factors

## Technology Stack

- **TensorFlow/Keras**: Deep learning framework for BI-LSTM model
- **Scikit-learn**: Machine learning utilities and Gradient Boosting model
- **Pandas & NumPy**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualizations
- **Streamlit**: Web application framework
- **Plotly**: Interactive data visualizations

## Future Enhancements

- Mobile application with habit tracking functionality
- Integration with learning management systems
- Expanded recommendation system with specific action plans
- Collaborative features for peer support and group study planning
- Real-time habit monitoring and ongoing prediction updates

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Southern University Bangladesh for supporting this research
- All students who participated in the data collection process
- Open-source libraries and tools used in this project 