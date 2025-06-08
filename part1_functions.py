import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer

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
    return df

def prepare_data_part1(df, test_size=0.2, random_state=42):
    """
    Prepare data for modeling: select features, split into train/test sets, handle missing values.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # 1. Select relevant features (age, systolic_bp, diastolic_bp, glucose_level, bmi)
    features = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi']
    X = df[features]
    
    # 2. Select target variable (disease_outcome)
    y = df['disease_outcome']
    
    # 3. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 4. Handle missing values using SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    # Print summary
    print(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    print(f"Features: {', '.join(features)}")
    print(f"Target distribution - Training: {np.bincount(y_train)}, Testing: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train, random_state=42, max_iter=1000):
    """
    Train a logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed for reproducibility
        max_iter: Maximum number of iterations for convergence
        
    Returns:
        Trained logistic regression model
    """
    # Initialize logistic regression model with balanced class weights
    # This helps account for any class imbalance in the data
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        class_weight='balanced',  # Adjust weights inversely proportional to class frequencies
        solver='lbfgs'  # Efficient solver for small datasets
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Print model information
    print(f"Logistic Regression model trained with {X_train.shape[1]} features")
    print(f"Model coefficients: {model.coef_}")
    print(f"Model intercept: {model.intercept_}")
    
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
    
    # For metrics like AUC, we need probability predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # 2. Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()  # Convert to list for serialization
    }
    
    # 3. Print metrics summary
    for metric_name, value in metrics.items():
        if metric_name != 'confusion_matrix':
            print(f"{metric_name}: {value:.4f}")
    
    print("Confusion Matrix:")
    print(f"TN: {metrics['confusion_matrix'][0][0]}, FP: {metrics['confusion_matrix'][0][1]}")
    print(f"FN: {metrics['confusion_matrix'][1][0]}, TP: {metrics['confusion_matrix'][1][1]}")
    
    return metrics

def save_results(metrics, output_file='results/results_part1.txt'):
    """
    Save evaluation metrics to a text file.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        output_file: Path to save the results
    """
    # 1. Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 2. Format metrics as strings for output
    with open(output_file, 'w') as f:
        f.write("Classification Model Evaluation Metrics\n")
        f.write("====================================\n\n")
        
        # Write simple metrics first
        for metric_name, value in metrics.items():
            if metric_name != 'confusion_matrix':
                f.write(f"{metric_name}: {value:.4f}\n")
        
        # Write confusion matrix with labels
        f.write("\nConfusion Matrix:\n")
        cm = metrics['confusion_matrix']
        f.write("              Predicted:\n")
        f.write("              Negative  Positive\n")
        f.write(f"Actual: Negative  {cm[0][0]:8d}  {cm[0][1]:8d}\n")
        f.write(f"        Positive  {cm[1][0]:8d}  {cm[1][1]:8d}\n")
    
    # Also save as JSON for easier programmatic comparison
    import json
    # Handle different types of confusion matrix formats safely
    metrics_for_json = {}
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            metrics_for_json[k] = float(v)
        else:
            # Handle different formats of confusion matrix
            if hasattr(v, 'tolist'):
                metrics_for_json[k] = v.tolist()
            else:
                # Already a list or another format
                metrics_for_json[k] = v
                
    json_output_file = output_file.replace('.txt', '.json')
    with open(json_output_file, 'w') as f:
        json.dump(metrics_for_json, f)
    
    print(f"Results saved to {output_file} and {json_output_file}")
    return output_file

def interpret_results(metrics):
    """
    Analyze model performance on imbalanced data.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        
    Returns:
        Dictionary with keys:
        - 'best_metric': Name of the metric that performed best
        - 'worst_metric': Name of the metric that performed worst
        - 'imbalance_impact_score': A score from 0-1 indicating how much
          the class imbalance affected results (0=no impact, 1=severe impact)
    """
    # Extract the metrics we want to compare (excluding confusion_matrix)
    comparable_metrics = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    
    # 1. Determine which metric performed best and worst
    best_metric = max(comparable_metrics, key=comparable_metrics.get)
    worst_metric = min(comparable_metrics, key=comparable_metrics.get)
    
    # 2. Calculate an imbalance impact score
    # Logic: Compare accuracy (which can be misleadingly high on imbalanced data)
    # with more robust metrics like F1 or recall
    imbalance_sensitive_metrics = ['f1', 'recall']
    min_sensitive_value = min(metrics[m] for m in imbalance_sensitive_metrics)
    
    # Calculate the normalized difference between accuracy and the minimum of F1/recall
    # This serves as our imbalance impact score - higher means more impact
    accuracy = metrics['accuracy']
    imbalance_impact_score = abs(accuracy - min_sensitive_value) / max(accuracy, 0.001)
    
    # Clip to ensure it's between 0 and 1
    imbalance_impact_score = min(1.0, max(0.0, imbalance_impact_score))
    
    # 3. Return the results
    interpretation = {
        'best_metric': best_metric,
        'worst_metric': worst_metric,
        'imbalance_impact_score': float(imbalance_impact_score)  # Ensure it's a Python float
    }
    
    # Print interpretation for user feedback
    print(f"Best performing metric: {best_metric} ({metrics[best_metric]:.4f})")
    print(f"Worst performing metric: {worst_metric} ({metrics[worst_metric]:.4f})")
    print(f"Class imbalance impact score: {imbalance_impact_score:.4f} " + 
          f"({'Low' if imbalance_impact_score < 0.3 else 'Moderate' if imbalance_impact_score < 0.6 else 'High'} impact)")
    
    return interpretation

# Main execution block
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Prepare data
    X_train, X_test, y_train, y_test = prepare_data_part1(df)
    
    # 3. Train model
    model = train_logistic_regression(X_train, y_train)
    
    # 4. Evaluate model
    metrics = calculate_evaluation_metrics(model, X_test, y_test)
    
    # 5. Save results
    output_file = save_results(metrics)
    
    # 6. Interpret results
    interpretation = interpret_results(metrics)
    print("\nResults Interpretation:")
    for key, value in interpretation.items():
        print(f"{key}: {value}")
