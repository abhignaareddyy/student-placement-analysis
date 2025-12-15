"""
Script to pre-train and save machine learning models as pickle files
Run this script to generate the pkl files before deploying
"""

import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def train_and_save_models():
    """Train models and save them as pickle files"""
    
    # Load data
    try:
        df = pd.read_csv("placementdata.csv")
        print("✅ Data loaded successfully")
    except FileNotFoundError:
        print("❌ Error: placementdata.csv not found")
        return
    
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
        'SVM': SVC(random_state=42, probability=True)
    }
    
    # Train and evaluate models
    results = {}
    trained_models = {}
    
    print("\n🔄 Training models...")
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        trained_models[name] = model
        
        print(f"  ✅ {name}: {accuracy:.3f}")
    
    # Find best model
    best_model_name = max(results, key=results.get)
    best_model = trained_models[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {results[best_model_name]:.3f})")
    
    # Create models directory
    if not os.path.exists('models'):
        os.makedirs('models')
        print("📁 Created models directory")
    
    # Save best model
    model_filename = f'models/{best_model_name.lower().replace(" ", "_")}_model.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"💾 Saved best model: {model_filename}")
    
    # Save all models (optional)
    for name, model in trained_models.items():
        filename = f'models/{name.lower().replace(" ", "_")}_model.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
    print("💾 Saved all models")
    
    # Save encoders
    with open('models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("💾 Saved label encoders")
    
    # Save feature columns
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(X.columns, f)
    print("💾 Saved feature columns")
    
    print("\n✅ All models and encoders saved successfully!")
    print("\nGenerated files:")
    for file in os.listdir('models'):
        print(f"  📄 models/{file}")

if __name__ == "__main__":
    train_and_save_models()