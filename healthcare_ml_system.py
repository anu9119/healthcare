"""
Personalized Healthcare Recommendations - Machine Learning Project
Complete implementation with data generation, preprocessing, model training, and evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# SECTION 1: DATA GENERATION (Since dataset needs to be downloaded)
# ============================================================================

def generate_healthcare_dataset(n_samples=1000):
    """
    Generate synthetic healthcare dataset for demonstration
    """
    np.random.seed(42)
    
    data = {
        'patient_id': range(1, n_samples + 1),
        'age': np.random.randint(18, 85, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'blood_pressure_systolic': np.random.randint(90, 180, n_samples),
        'blood_pressure_diastolic': np.random.randint(60, 120, n_samples),
        'cholesterol': np.random.randint(150, 300, n_samples),
        'heart_rate': np.random.randint(60, 120, n_samples),
        'glucose': np.random.randint(70, 200, n_samples),
        'bmi': np.round(np.random.uniform(18.5, 40, n_samples), 1),
        'smoking_status': np.random.choice(['Non-smoker', 'Former', 'Current'], n_samples, p=[0.6, 0.25, 0.15]),
        'exercise_level': np.random.choice(['Sedentary', 'Light', 'Moderate', 'Active'], n_samples, p=[0.3, 0.3, 0.3, 0.1]),
        'alcohol_consumption': np.random.choice(['None', 'Occasional', 'Regular', 'Heavy'], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
        'family_history': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'stress_level': np.random.randint(1, 11, n_samples),
        'sleep_hours': np.round(np.random.uniform(4, 10, n_samples), 1),
        'hemoglobin': np.round(np.random.uniform(11, 18, n_samples), 1),
        'platelet_count': np.random.randint(150, 450, n_samples),
        'white_blood_cells': np.round(np.random.uniform(4, 11, n_samples), 1),
    }
    
    df = pd.DataFrame(data)
    
    # Generate recommendations based on health indicators
    recommendations = []
    for idx, row in df.iterrows():
        risk_score = 0
        
        # Age risk
        if row['age'] > 60:
            risk_score += 2
        elif row['age'] > 45:
            risk_score += 1
            
        # Blood pressure risk
        if row['blood_pressure_systolic'] > 140 or row['blood_pressure_diastolic'] > 90:
            risk_score += 2
        elif row['blood_pressure_systolic'] > 130:
            risk_score += 1
            
        # Cholesterol risk
        if row['cholesterol'] > 240:
            risk_score += 2
        elif row['cholesterol'] > 200:
            risk_score += 1
            
        # Glucose risk
        if row['glucose'] > 126:
            risk_score += 2
        elif row['glucose'] > 100:
            risk_score += 1
            
        # BMI risk
        if row['bmi'] > 30:
            risk_score += 2
        elif row['bmi'] > 25:
            risk_score += 1
            
        # Lifestyle factors
        if row['smoking_status'] == 'Current':
            risk_score += 2
        if row['exercise_level'] == 'Sedentary':
            risk_score += 1
        if row['family_history'] == 'Yes':
            risk_score += 1
            
        # Assign recommendation based on risk score
        if risk_score >= 8:
            recommendations.append('Immediate Medical Attention')
        elif risk_score >= 5:
            recommendations.append('Medication & Lifestyle Changes')
        elif risk_score >= 3:
            recommendations.append('Lifestyle Changes')
        elif risk_score >= 1:
            recommendations.append('Regular Check-up')
        else:
            recommendations.append('Maintain Current Health')
    
    df['recommendation'] = recommendations
    
    # Add some missing values randomly (realistic scenario)
    missing_cols = ['cholesterol', 'glucose', 'sleep_hours']
    for col in missing_cols:
        missing_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df

# ============================================================================
# SECTION 2: DATA EXPLORATION AND VISUALIZATION
# ============================================================================

def explore_data(df):
    """
    Perform exploratory data analysis
    """
    print("=" * 80)
    print("DATA EXPLORATION")
    print("=" * 80)
    
    print("\n1. Dataset Shape:")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n2. Dataset Info:")
    print(df.info())
    
    print("\n3. First 5 Rows:")
    print(df.head())
    
    print("\n4. Statistical Summary:")
    print(df.describe())
    
    print("\n5. Missing Values:")
    print(df.isnull().sum())
    
    print("\n6. Target Variable Distribution:")
    print(df['recommendation'].value_counts())
    
    return df

def visualize_data(df):
    """
    Create visualizations for data exploration
    """
    print("\n" + "=" * 80)
    print("DATA VISUALIZATION")
    print("=" * 80)
    
    # 1. Target Distribution
    plt.figure(figsize=(10, 6))
    df['recommendation'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Healthcare Recommendations', fontsize=16, fontweight='bold')
    plt.xlabel('Recommendation Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # 2. Age Distribution by Recommendation
    plt.figure(figsize=(12, 6))
    df.boxplot(column='age', by='recommendation', figsize=(12, 6))
    plt.title('Age Distribution by Recommendation Type', fontsize=16, fontweight='bold')
    plt.suptitle('')
    plt.xlabel('Recommendation', fontsize=12)
    plt.ylabel('Age', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(14, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Correlation Matrix of Numeric Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 4. Key Health Indicators
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(df['blood_pressure_systolic'], bins=30, color='salmon', edgecolor='black')
    axes[0, 0].set_title('Systolic Blood Pressure Distribution')
    axes[0, 0].set_xlabel('Blood Pressure (mmHg)')
    
    axes[0, 1].hist(df['cholesterol'], bins=30, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Cholesterol Level Distribution')
    axes[0, 1].set_xlabel('Cholesterol (mg/dL)')
    
    axes[1, 0].hist(df['bmi'], bins=30, color='lightblue', edgecolor='black')
    axes[1, 0].set_title('BMI Distribution')
    axes[1, 0].set_xlabel('BMI')
    
    axes[1, 1].hist(df['glucose'], bins=30, color='wheat', edgecolor='black')
    axes[1, 1].set_title('Glucose Level Distribution')
    axes[1, 1].set_xlabel('Glucose (mg/dL)')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Lifestyle Factors
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    df['smoking_status'].value_counts().plot(kind='bar', ax=axes[0], color='coral', edgecolor='black')
    axes[0].set_title('Smoking Status')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=45)
    
    df['exercise_level'].value_counts().plot(kind='bar', ax=axes[1], color='lightgreen', edgecolor='black')
    axes[1].set_title('Exercise Level')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', rotation=45)
    
    df['alcohol_consumption'].value_counts().plot(kind='bar', ax=axes[2], color='skyblue', edgecolor='black')
    axes[2].set_title('Alcohol Consumption')
    axes[2].set_xlabel('')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# SECTION 3: DATA PREPROCESSING
# ============================================================================

def preprocess_data(df):
    """
    Preprocess the dataset
    """
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING")
    print("=" * 80)
    
    # Create a copy
    df_processed = df.copy()
    
    # Drop patient_id as it's not useful for prediction
    df_processed = df_processed.drop('patient_id', axis=1)
    
    # Separate features and target
    X = df_processed.drop('recommendation', axis=1)
    y = df_processed['recommendation']
    
    # Identify feature types
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumerical Features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")
    
    # Create preprocessing pipelines
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features

# ============================================================================
# SECTION 4: MODEL TRAINING AND EVALUATION
# ============================================================================

def train_multiple_models(X_train, y_train, preprocessor):
    """
    Train multiple machine learning models
    """
    print("\n" + "=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Train the model
        pipeline.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"  Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        trained_models[name] = pipeline
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    results = []
    
    for name, model in models.items():
        print(f"\n{name} Performance:")
        print("-" * 50)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    return results_df

def visualize_model_comparison(results_df):
    """
    Visualize model comparison
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON VISUALIZATION")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'wheat']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        row = idx // 2
        col = idx % 2
        
        axes[row, col].barh(results_df['Model'], results_df[metric], color=color, edgecolor='black')
        axes[row, col].set_xlabel(metric, fontsize=12)
        axes[row, col].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        axes[row, col].set_xlim([0, 1])
        
        # Add value labels
        for i, v in enumerate(results_df[metric]):
            axes[row, col].text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_for_best_model(best_model, X_test, y_test):
    """
    Plot detailed confusion matrix for the best model
    """
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=best_model.classes_, 
                yticklabels=best_model.classes_,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Best Model', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ============================================================================
# SECTION 5: RECOMMENDATION SYSTEM
# ============================================================================

class HealthcareRecommendationSystem:
    """
    Complete Healthcare Recommendation System
    """
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        
    def generate_recommendation(self, patient_data):
        """
        Generate personalized healthcare recommendation
        """
        # Make prediction
        prediction = self.model.predict(patient_data)[0]
        probabilities = self.model.predict_proba(patient_data)[0]
        
        # Get confidence
        confidence = max(probabilities) * 100
        
        # Generate detailed recommendation
        recommendation_details = self._get_recommendation_details(prediction, patient_data)
        
        return {
            'recommendation': prediction,
            'confidence': confidence,
            'details': recommendation_details,
            'probabilities': dict(zip(self.model.classes_, probabilities))
        }
    
    def _get_recommendation_details(self, recommendation, patient_data):
        """
        Provide detailed recommendation based on prediction
        """
        details = {
            'Maintain Current Health': {
                'advice': 'Continue your healthy lifestyle habits',
                'actions': [
                    'Keep up regular exercise routine',
                    'Maintain balanced diet',
                    'Schedule annual check-up',
                    'Monitor stress levels'
                ]
            },
            'Regular Check-up': {
                'advice': 'Schedule a routine medical check-up',
                'actions': [
                    'Visit your primary care physician',
                    'Monitor blood pressure regularly',
                    'Consider lifestyle improvements',
                    'Follow up in 3-6 months'
                ]
            },
            'Lifestyle Changes': {
                'advice': 'Significant lifestyle modifications recommended',
                'actions': [
                    'Increase physical activity to 150 min/week',
                    'Adopt heart-healthy diet',
                    'Reduce stress through meditation/yoga',
                    'Improve sleep quality (7-9 hours)',
                    'Limit alcohol consumption',
                    'Quit smoking if applicable'
                ]
            },
            'Medication & Lifestyle Changes': {
                'advice': 'Medical intervention combined with lifestyle changes',
                'actions': [
                    'Consult with physician for medication',
                    'Regular monitoring of vitals',
                    'Strict dietary modifications',
                    'Structured exercise program',
                    'Monthly follow-up appointments',
                    'Consider specialist consultation'
                ]
            },
            'Immediate Medical Attention': {
                'advice': 'Seek immediate medical attention',
                'actions': [
                    'Schedule urgent appointment with physician',
                    'Comprehensive health screening required',
                    'May need specialist referrals',
                    'Close monitoring of health parameters',
                    'Immediate lifestyle intervention',
                    'Possible hospitalization or intensive treatment'
                ]
            }
        }
        
        return details.get(recommendation, {'advice': 'Consult healthcare professional', 'actions': []})
    
    def display_recommendation(self, result):
        """
        Display recommendation in a formatted way
        """
        print("\n" + "=" * 80)
        print("PERSONALIZED HEALTHCARE RECOMMENDATION")
        print("=" * 80)
        print(f"\nRecommendation: {result['recommendation']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print(f"\nAdvice: {result['details']['advice']}")
        print("\nRecommended Actions:")
        for i, action in enumerate(result['details']['actions'], 1):
            print(f"  {i}. {action}")
        print("\nProbability Distribution:")
        for rec, prob in result['probabilities'].items():
            print(f"  {rec}: {prob*100:.2f}%")
        print("=" * 80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function
    """
    print("=" * 80)
    print("PERSONALIZED HEALTHCARE RECOMMENDATIONS")
    print("Machine Learning Project")
    print("=" * 80)
    
    # Step 1: Generate or Load Dataset
    print("\nStep 1: Loading Dataset...")
    df = generate_healthcare_dataset(n_samples=1000)
    print("Dataset loaded successfully!")
    
    # Step 2: Explore Data
    print("\nStep 2: Exploring Data...")
    df = explore_data(df)
    
    # Step 3: Visualize Data
    print("\nStep 3: Creating Visualizations...")
    visualize_data(df)
    
    # Step 4: Preprocess Data
    print("\nStep 4: Preprocessing Data...")
    X_train, X_test, y_train, y_test, preprocessor, num_features, cat_features = preprocess_data(df)
    
    # Step 5: Train Models
    print("\nStep 5: Training Multiple Models...")
    trained_models = train_multiple_models(X_train, y_train, preprocessor)
    
    # Step 6: Evaluate Models
    print("\nStep 6: Evaluating Models...")
    results_df = evaluate_models(trained_models, X_test, y_test)
    
    # Step 7: Visualize Results
    print("\nStep 7: Visualizing Results...")
    visualize_model_comparison(results_df)
    
    # Step 8: Select Best Model
    best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
    best_model = trained_models[best_model_name]
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {results_df['Accuracy'].max():.4f}")
    
    # Plot confusion matrix for best model
    plot_confusion_matrix_for_best_model(best_model, X_test, y_test)
    
    # Step 9: Create Recommendation System
    print("\nStep 9: Creating Recommendation System...")
    recommendation_system = HealthcareRecommendationSystem(best_model, num_features + cat_features)
    
    # Step 10: Test with Example Patients
    print("\nStep 10: Testing Recommendation System...")
    
    # Example Patient 1: Healthy Individual
    patient1 = pd.DataFrame({
        'age': [35],
        'gender': ['Male'],
        'blood_pressure_systolic': [115],
        'blood_pressure_diastolic': [75],
        'cholesterol': [180],
        'heart_rate': [70],
        'glucose': [90],
        'bmi': [22.5],
        'smoking_status': ['Non-smoker'],
        'exercise_level': ['Active'],
        'alcohol_consumption': ['Occasional'],
        'family_history': ['No'],
        'stress_level': [4],
        'sleep_hours': [8.0],
        'hemoglobin': [14.5],
        'platelet_count': [250],
        'white_blood_cells': [7.0]
    })
    
    print("\n--- Example Patient 1: Healthy Individual ---")
    result1 = recommendation_system.generate_recommendation(patient1)
    recommendation_system.display_recommendation(result1)
    
    # Example Patient 2: High Risk Individual
    patient2 = pd.DataFrame({
        'age': [65],
        'gender': ['Female'],
        'blood_pressure_systolic': [160],
        'blood_pressure_diastolic': [100],
        'cholesterol': [280],
        'heart_rate': [95],
        'glucose': [150],
        'bmi': [32.5],
        'smoking_status': ['Current'],
        'exercise_level': ['Sedentary'],
        'alcohol_consumption': ['Regular'],
        'family_history': ['Yes'],
        'stress_level': [8],
        'sleep_hours': [5.5],
        'hemoglobin': [12.0],
        'platelet_count': [180],
        'white_blood_cells': [9.5]
    })
    
    print("\n--- Example Patient 2: High Risk Individual ---")
    result2 = recommendation_system.generate_recommendation(patient2)
    recommendation_system.display_recommendation(result2)
    
    # Example Patient 3: Moderate Risk
    patient3 = pd.DataFrame({
        'age': [50],
        'gender': ['Male'],
        'blood_pressure_systolic': [135],
        'blood_pressure_diastolic': [85],
        'cholesterol': [220],
        'heart_rate': [80],
        'glucose': [110],
        'bmi': [27.0],
        'smoking_status': ['Former'],
        'exercise_level': ['Light'],
        'alcohol_consumption': ['Occasional'],
        'family_history': ['Yes'],
        'stress_level': [6],
        'sleep_hours': [6.5],
        'hemoglobin': [13.5],
        'platelet_count': [220],
        'white_blood_cells': [7.5]
    })
    
    print("\n--- Example Patient 3: Moderate Risk Individual ---")
    result3 = recommendation_system.generate_recommendation(patient3)
    recommendation_system.display_recommendation(result3)
    
    print("\n" + "=" * 80)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nKey Insights:")
    print(f"  • Total patients analyzed: {len(df)}")
    print(f"  • Best performing model: {best_model_name}")
    print(f"  • Model accuracy: {results_df['Accuracy'].max():.2%}")
    print(f"  • Features used: {len(num_features) + len(cat_features)}")
    print("\nRecommendation System is ready for deployment!")

# Run the complete project
if __name__ == "__main__":
    main()
