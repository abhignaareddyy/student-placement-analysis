import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Student Placement Analysis Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the placement data"""
    try:
        df = pd.read_csv("placementdata.csv")
        return df
    except FileNotFoundError:
        st.error("Please upload the placementdata.csv file to proceed.")
        return None

def data_overview(df):
    """Display data overview section"""
    st.markdown('<h2 class="section-header">📊 Data Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(df))
    
    with col2:
        placed_count = len(df[df['PlacementStatus'] == 'Placed'])
        st.metric("Placed Students", placed_count)
    
    with col3:
        not_placed_count = len(df[df['PlacementStatus'] == 'NotPlaced'])
        st.metric("Not Placed Students", not_placed_count)
    
    with col4:
        placement_rate = (placed_count / len(df)) * 100
        st.metric("Placement Rate", f"{placement_rate:.1f}%")
    
    # Display basic info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Info")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Columns:** {list(df.columns)}")
        
    with col2:
        st.subheader("Data Quality Check")
        duplicates = df.duplicated().sum()
        null_values = df.isnull().sum().sum()
        st.write(f"**Duplicate rows:** {duplicates}")
        st.write(f"**Missing values:** {null_values}")
    
    # Display first few rows
    st.subheader("Sample Data")
    st.dataframe(df.head(), use_container_width=True)

def exploratory_analysis(df):
    """Perform exploratory data analysis"""
    st.markdown('<h2 class="section-header">🔍 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    # Numerical columns analysis
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'StudentID' in numerical_cols:
        numerical_cols.remove('StudentID')
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Outlier Detection
    st.subheader("📈 Outlier Detection")
    
    # Create box plots for numerical columns
    rows = math.ceil(len(numerical_cols) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            sns.boxplot(y=df[col], color='lightblue', ax=axes[i])
            axes[i].set_title(col)
    
    # Hide empty subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Distribution Analysis
    st.subheader("📊 Distribution Analysis")
    
    # Histograms for numerical columns
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            sns.histplot(df[col], kde=True, color="red", ax=axes[i])
            axes[i].set_title(col)
    
    # Hide empty subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Categorical Analysis
    st.subheader("📋 Categorical Variables Analysis")
    
    # Count plots for categorical columns
    categorical_rows = math.ceil(len(categorical_cols) / 3)
    fig, axes = plt.subplots(categorical_rows, 3, figsize=(15, 5 * categorical_rows))
    
    if categorical_rows == 1:
        axes = [axes] if len(categorical_cols) == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            sns.countplot(data=df, x=col, ax=axes[i])
            axes[i].set_title(col)
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for i in range(len(categorical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)

def placement_analysis(df):
    """Analyze placement patterns"""
    st.markdown('<h2 class="section-header">🎯 Placement Analysis</h2>', unsafe_allow_html=True)
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'StudentID' in numerical_cols:
        numerical_cols.remove('StudentID')
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Numerical features vs Placement Status
    st.subheader("📊 Numerical Features vs Placement Status")
    
    rows = math.ceil(len(numerical_cols) / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            sns.boxplot(x="PlacementStatus", y=col, data=df, ax=axes[i])
            axes[i].set_title(col)
    
    # Hide empty subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Distribution by placement status
    st.subheader("📈 Distribution by Placement Status")
    
    fig, axes = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else []
    
    for i, col in enumerate(numerical_cols):
        if i < len(axes):
            sns.histplot(data=df, x=col, hue="PlacementStatus", kde=True, multiple="stack", ax=axes[i])
            axes[i].set_title(col)
    
    # Hide empty subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Categorical features vs Placement Status
    st.subheader("📋 Categorical Features vs Placement Status")
    
    categorical_rows = math.ceil(len(categorical_cols) / 3)
    fig, axes = plt.subplots(categorical_rows, 3, figsize=(15, 5 * categorical_rows))
    
    if categorical_rows == 1:
        axes = [axes] if len(categorical_cols) == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(categorical_cols):
        if i < len(axes):
            sns.countplot(x=col, hue="PlacementStatus", data=df, palette="pastel", ax=axes[i])
            axes[i].set_title(col)
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for i in range(len(categorical_cols), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)

def correlation_analysis(df):
    """Perform correlation analysis"""
    st.markdown('<h2 class="section-header">🔗 Correlation Analysis</h2>', unsafe_allow_html=True)
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'StudentID' in numerical_cols:
        numerical_cols.remove('StudentID')
    
    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, ax=ax)
    plt.title('Correlation Matrix of Numerical Features')
    st.pyplot(fig)
    
    # Feature importance insights
    st.subheader("🔍 Key Insights")
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
    
    if high_corr_pairs:
        st.write("**Highly Correlated Feature Pairs (|correlation| > 0.5):**")
        for feat1, feat2, corr_val in high_corr_pairs:
            st.write(f"- {feat1} ↔ {feat2}: {corr_val:.3f}")
    else:
        st.write("No highly correlated feature pairs found.")

def save_model_and_encoders(model, encoders, feature_columns, model_name):
    """Save model and encoders to pickle files"""
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save model
    with open(f'models/{model_name.lower().replace(" ", "_")}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save encoders
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    # Save feature columns
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)

def load_model_and_encoders():
    """Load model and encoders from pickle files"""
    try:
        # Find the best model file
        model_files = [f for f in os.listdir('models') if f.endswith('_model.pkl')]
        if not model_files:
            return None, None, None, None
        
        # Load the first available model (you can modify this logic)
        model_file = model_files[0]
        model_name = model_file.replace('_model.pkl', '').replace('_', ' ').title()
        
        with open(f'models/{model_file}', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/label_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        return model, encoders, feature_columns, model_name
    except:
        return None, None, None, None

def machine_learning_models(df):
    """Build and evaluate machine learning models"""
    st.markdown('<h2 class="section-header">🤖 Machine Learning Models</h2>', unsafe_allow_html=True)
    
    # Check if models already exist
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Retrain Models", type="secondary"):
            # Clear existing models
            if os.path.exists('models'):
                import shutil
                shutil.rmtree('models')
            st.rerun()
    
    with col1:
        # Try to load existing model
        existing_model, existing_encoders, existing_features, existing_name = load_model_and_encoders()
        
        if existing_model is not None:
            st.success(f"✅ Loaded existing {existing_name} model from pickle file")
            return existing_model, existing_encoders, existing_features
    
    # Train new models if none exist
    st.info("🔄 Training new models...")
    
    # Prepare data
    df_ml = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = df_ml.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if col != 'PlacementStatus':
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col])
            label_encoders[col] = le
    
    # Prepare features and target
    X = df_ml.drop(['PlacementStatus', 'StudentID'], axis=1)
    y = LabelEncoder().fit_transform(df_ml['PlacementStatus'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True)  # Added probability=True for predict_proba
    }
    
    # Train and evaluate models
    results = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Performance")
        
        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            
            st.write(f"**{name}:** {accuracy:.3f}")
    
    with col2:
        st.subheader("🏆 Best Model Performance")
        
        # Find best model
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]
        
        st.write(f"**Best Model:** {best_model_name}")
        st.write(f"**Accuracy:** {results[best_model_name]:.3f}")
        
        # Detailed classification report for best model
        y_pred_best = best_model.predict(X_test)
        
        st.subheader("📋 Classification Report")
        report = classification_report(y_test, y_pred_best, target_names=['Not Placed', 'Placed'])
        st.text(report)
    
    # Feature importance (for tree-based models)
    if best_model_name in ['Random Forest', 'Decision Tree']:
        st.subheader("🎯 Feature Importance")
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax)
        plt.title('Top 10 Most Important Features')
        st.pyplot(fig)
    
    # Confusion Matrix
    st.subheader("🔄 Confusion Matrix")
    
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Placed', 'Placed'],
                yticklabels=['Not Placed', 'Placed'], ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)
    
    # Save the best model and encoders
    save_model_and_encoders(best_model, label_encoders, X.columns, best_model_name)
    st.success(f"✅ Saved {best_model_name} model to pickle file")
    
    return best_model, label_encoders, X.columns

def prediction_interface(model, label_encoders, feature_columns, df):
    """Create prediction interface"""
    st.markdown('<h2 class="section-header">🔮 Placement Prediction</h2>', unsafe_allow_html=True)
    
    st.write("Enter student details to predict placement probability:")
    
    # Create input fields
    col1, col2 = st.columns(2)
    
    with col1:
        cgpa = st.slider("CGPA", min_value=5.0, max_value=10.0, value=7.5, step=0.1)
        internships = st.number_input("Number of Internships", min_value=0, max_value=5, value=1)
        projects = st.number_input("Number of Projects", min_value=0, max_value=10, value=2)
        workshops = st.number_input("Workshops/Certifications", min_value=0, max_value=10, value=2)
        aptitude_score = st.slider("Aptitude Test Score", min_value=50, max_value=100, value=75)
    
    with col2:
        soft_skills = st.slider("Soft Skills Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        placement_training = st.selectbox("Placement Training", ["Yes", "No"])
        ssc_marks = st.slider("SSC Marks", min_value=50, max_value=100, value=75)
        hsc_marks = st.slider("HSC Marks", min_value=50, max_value=100, value=75)
    
    if st.button("Predict Placement", type="primary"):
        # Prepare input data
        input_data = {
            'CGPA': cgpa,
            'Internships': internships,
            'Projects': projects,
            'Workshops/Certifications': workshops,
            'AptitudeTestScore': aptitude_score,
            'SoftSkillsRating': soft_skills,
            'ExtracurricularActivities': extracurricular,
            'PlacementTraining': placement_training,
            'SSC_Marks': ssc_marks,
            'HSC_Marks': hsc_marks
        }
        
        # Encode categorical variables
        for col, value in input_data.items():
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform([value])[0]
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.success("🎉 **Prediction: PLACED**")
            else:
                st.error("❌ **Prediction: NOT PLACED**")
        
        with col2:
            st.write("**Prediction Probabilities:**")
            st.write(f"Not Placed: {prediction_proba[0]:.3f}")
            st.write(f"Placed: {prediction_proba[1]:.3f}")

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">🎓 Student Placement Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Data Overview", "Exploratory Analysis", "Placement Analysis", 
         "Correlation Analysis", "Machine Learning", "Prediction"]
    )
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Navigation logic
        if page == "Data Overview":
            data_overview(df)
        
        elif page == "Exploratory Analysis":
            exploratory_analysis(df)
        
        elif page == "Placement Analysis":
            placement_analysis(df)
        
        elif page == "Correlation Analysis":
            correlation_analysis(df)
        
        elif page == "Machine Learning":
            model, encoders, features = machine_learning_models(df)
            # Store in session state for prediction page
            st.session_state['model'] = model
            st.session_state['encoders'] = encoders
            st.session_state['features'] = features
        
        elif page == "Prediction":
            # Try to load model from session state or pickle files
            model, encoders, features = None, None, None
            
            if 'model' in st.session_state:
                model = st.session_state['model']
                encoders = st.session_state['encoders'] 
                features = st.session_state['features']
            else:
                # Try to load from pickle files
                model, encoders, features, _ = load_model_and_encoders()
            
            if model is not None:
                prediction_interface(model, encoders, features, df)
            else:
                st.warning("⚠️ No trained model found. Please run the Machine Learning section first to train the model.")
                if st.button("🚀 Go to Machine Learning Section"):
                    st.switch_page("Machine Learning")
    
    else:
        st.info("Please ensure 'placementdata.csv' is in the same directory as this app.")

if __name__ == "__main__":
    main()