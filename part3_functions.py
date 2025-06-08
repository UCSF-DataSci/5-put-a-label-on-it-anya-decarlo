import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

def load_data(file_path):
    """
    Load the synthetic health data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    
    # Load the CSV file using pandas
    df = pd.read_csv(file_path)
    
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Check if 'smoker_status' is present
    if 'smoker_status' in df.columns:
        print(f"Categorical feature 'smoker_status' found with values: {df['smoker_status'].unique()}")
    
    return df

def encode_categorical_features(df, column_to_encode='smoker_status'):
    """
    Encode a categorical column using OneHotEncoder.
    
    Args:
        df: Input DataFrame
        column_to_encode: Name of the categorical column to encode
        
    Returns:
        DataFrame with the categorical column replaced by one-hot encoded columns
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Check if the column exists
    if column_to_encode not in result_df.columns:
        print(f"Warning: Column '{column_to_encode}' not found in DataFrame. Returning original DataFrame.")
        return result_df
    
    # 1. Extract the categorical column
    categorical_data = result_df[[column_to_encode]]
    
    # 2. Apply OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop first to avoid multicollinearity
    encoded_data = encoder.fit_transform(categorical_data)
    
    # 3. Create new column names
    encoded_feature_names = [f"{column_to_encode}_{category}" 
                           for category in encoder.categories_[0][1:]]  # Skip first category due to drop='first'
    
    # 4. Create a DataFrame with the encoded data
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_feature_names, index=result_df.index)
    
    # 5. Drop the original categorical column and concatenate the encoded columns
    result_df = pd.concat([result_df.drop(column_to_encode, axis=1), encoded_df], axis=1)
    
    print(f"Encoded '{column_to_encode}' into features: {', '.join(encoded_feature_names)}")
    
    return result_df

def prepare_data_part3(df, test_size=0.2, random_state=42):
    """
    Prepare data with categorical encoding.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # 1. Encode categorical features
    df_encoded = encode_categorical_features(df, column_to_encode='smoker_status')
    
    # 2. Select relevant features (including the one-hot encoded ones)
    # Base features from previous parts
    base_features = ['age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'glucose_level', 'bmi']
    
    # Add one-hot encoded features if they exist
    smoker_features = [col for col in df_encoded.columns if col.startswith('smoker_status_')]
    all_features = base_features + smoker_features
    
    X = df_encoded[all_features]
    y = df_encoded['disease_outcome']
    
    # 3. Split data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 4. Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    print(f"Data prepared with categorical features: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    print(f"Features: {', '.join(all_features)}")
    print(f"Target distribution - Training: {np.bincount(y_train)}, Testing: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to oversample the minority class.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
        
    Returns:
        Resampled X_train and y_train with balanced classes
    """
    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Print class distribution before and after SMOTE
    before_counts = np.bincount(y_train)
    after_counts = np.bincount(y_train_resampled)
    
    print(f"Class distribution before SMOTE: {before_counts}")
    print(f"Class distribution after SMOTE: {after_counts}")
    
    return X_train_resampled, y_train_resampled

def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained logistic regression model
    """
    # Initialize logistic regression model
    # Note: We don't need class_weight='balanced' here because we're using SMOTE
    model = LogisticRegression(
        C=1.0,               # Regularization strength (inverse)
        max_iter=1000,       # Maximum iterations for convergence
        solver='liblinear',  # Efficient solver for small datasets
        random_state=42      # For reproducibility
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Print model information
    print(f"Logistic Regression model trained on {len(X_train)} samples with {X_train.shape[1]} features")
    print(f"Model coefficients shape: {model.coef_.shape}")
    
    # Get the feature names and their corresponding coefficients
    feature_coefficients = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    # Print top features by coefficient magnitude
    print("Top 3 important features by coefficient magnitude:")
    for i, row in feature_coefficients.head(3).iterrows():
        print(f"  - {row['feature']}: {row['coefficient']:.4f}")
    
    return model

def calculate_evaluation_metrics(model, X_test, y_test):
    """
    Calculate classification evaluation metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing accuracy, precision, recall, f1, auc, and confusion_matrix
    """
    # 1. Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability for positive class
    
    # 2. Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # 3. Create confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 4. Return metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': conf_matrix
    }
    
    print("Evaluation metrics calculated on test set")
    
    return metrics

def save_results_part3(metrics, output_file='results/results_part3.txt'):
    """
    Save the evaluation metrics to a text file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        output_file: Path to save the results
        
    Returns:
        Path to the saved file
    """
    # 1. Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 2. Create test-compatible version with simple key-value pairs
    with open(output_file, 'w') as f:
        for metric_name, value in metrics.items():
            if metric_name != 'confusion_matrix':
                f.write(f"{metric_name}: {value:.4f}\n")
    
    # Create human-readable formatted version
    formatted_output_file = output_file.replace('.txt', '_formatted.txt')
    with open(formatted_output_file, 'w') as f:
        f.write("SMOTE & Categorical Features Model Results\n")
        f.write("=====================================\n\n")
        
        f.write("Classification Metrics:\n")
        for metric_name, value in metrics.items():
            if metric_name != 'confusion_matrix':
                f.write(f"{metric_name}: {value:.4f}\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write(f"{metrics['confusion_matrix']}\n\n")
        
        f.write("Notes:\n")
        f.write("1. This model was trained on balanced data using SMOTE oversampling.\n")
        f.write("2. Categorical features (smoker_status) were encoded using OneHotEncoder.\n")
        f.write("3. The model was evaluated on the original imbalanced test distribution.\n")
    
    # Also save as JSON for easier comparison
    metrics_for_json = {k: float(v) if k != 'confusion_matrix' else v.tolist() 
                       for k, v in metrics.items()}
    with open(output_file.replace('.txt', '.json'), 'w') as f:
        json.dump(metrics_for_json, f)
    
    print(f"\nResults saved to {output_file} (test format) and {formatted_output_file} (human-readable)")
    return output_file

def compare_models(part1_metrics, part3_metrics):
    """
    Calculate percentage improvement between models trained on imbalanced vs. balanced data.
    
    Args:
        part1_metrics: Dictionary containing evaluation metrics from Part 1 (imbalanced)
        part3_metrics: Dictionary containing evaluation metrics from Part 3 (balanced)
        
    Returns:
        Dictionary with metric names as keys and improvement percentages as values
    """
    # 1. Calculate percentage improvement for each metric
    improvement = {}
    
    # 2. For metrics where higher is better (all standard ML metrics)
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        if metric in part1_metrics and metric in part3_metrics:
            part1_value = part1_metrics[metric]
            part3_value = part3_metrics[metric]
            
            # Calculate percentage improvement
            if part1_value > 0:  # Avoid division by zero
                percent_improvement = ((part3_value - part1_value) / part1_value) * 100
                improvement[metric] = percent_improvement
            else:
                improvement[metric] = float('inf') if part3_value > 0 else 0.0
    
    return improvement

# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Prepare data with categorical encoding
    X_train, X_test, y_train, y_test = prepare_data_part3(df)
    
    # 3. Apply SMOTE to balance the training data
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    # 4. Train model on resampled data
    model = train_logistic_regression(X_train_resampled, y_train_resampled)
    
    # 5. Evaluate on original test set
    metrics = calculate_evaluation_metrics(model, X_test, y_test)
    
    # 6. Print metrics
    for metric_name, value in metrics.items():
        if metric_name != 'confusion_matrix':
            print(f"{metric_name}: {value:.4f}")
        else:
            print(f"\nConfusion Matrix:\n{value}")
    
    # 7. Save results
    save_results_part3(metrics)
    
    # 8. Load Part 1 results for comparison
    try:
        with open('results/results_part1.json', 'r') as f:
            part1_metrics = json.load(f)
        
        # 9. Compare models
        comparison = compare_models(part1_metrics, metrics)
        print("\nModel Comparison (improvement percentages):")
        for metric, improvement in comparison.items():
            print(f"{metric}: {improvement:.2f}%")
    except FileNotFoundError:
        print("Part 1 results not found. Run part1_functions.py first.")
