# Student Placement Analysis Dashboard

A comprehensive Streamlit web application for analyzing student placement data and predicting placement outcomes using machine learning.

## Features

### 📊 Data Overview
- Dataset statistics and basic information
- Data quality checks (duplicates, missing values)
- Sample data preview

### 🔍 Exploratory Data Analysis
- Outlier detection with box plots
- Distribution analysis with histograms
- Categorical variable analysis with count plots

### 🎯 Placement Analysis
- Numerical features vs placement status comparison
- Distribution analysis by placement status
- Categorical features impact on placement

### 🔗 Correlation Analysis
- Correlation matrix heatmap
- Identification of highly correlated features
- Feature relationship insights

### 🤖 Machine Learning Models
- Multiple classification algorithms:
  - Random Forest
  - Logistic Regression
  - Decision Tree
  - Support Vector Machine
- Model performance comparison
- Feature importance analysis
- Confusion matrix visualization

### 🔮 Placement Prediction
- Interactive prediction interface
- Real-time placement probability calculation
- User-friendly input forms

## Installation

1. Clone or download the project files
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure `placementdata.csv` is in the same directory as the app
2. Run the Streamlit app:
   ```bash
   streamlit run placement_analysis_app.py
   ```
3. Open your web browser and navigate to the provided local URL (usually http://localhost:8501)

## Data Requirements

The application expects a CSV file named `placementdata.csv` with the following columns:
- StudentID
- CGPA
- Internships
- Projects
- Workshops/Certifications
- AptitudeTestScore
- SoftSkillsRating
- ExtracurricularActivities
- PlacementTraining
- SSC_Marks
- HSC_Marks
- PlacementStatus

## Navigation

Use the sidebar to navigate between different sections:
1. **Data Overview**: Basic dataset information
2. **Exploratory Analysis**: Data distribution and patterns
3. **Placement Analysis**: Placement-specific insights
4. **Correlation Analysis**: Feature relationships
5. **Machine Learning**: Model training and evaluation
6. **Prediction**: Interactive placement prediction

## Notes

- Run the "Machine Learning" section before using the "Prediction" feature
- The app automatically handles data preprocessing and encoding
- All visualizations are interactive and responsive
- Model performance metrics are displayed for comparison

## Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms