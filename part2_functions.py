import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer

def load_data(file_path):
    """
    Load the synthetic health data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data with timestamp parsed as datetime
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    
    # Load the CSV file using pandas with timestamp as datetime
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    
    print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def extract_rolling_features(df, window_size_seconds):
    """
    Calculate rolling mean and standard deviation for heart rate.
    
    Args:
        df: DataFrame with timestamp and heart_rate columns
        window_size_seconds: Size of the rolling window in seconds
        
    Returns:
        DataFrame with added hr_rolling_mean and hr_rolling_std columns
    """
    # Make a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # 1. Sort data by timestamp to ensure chronological order
    result_df = result_df.sort_values('timestamp')
    
    # 2. Set timestamp as index for time-based rolling operations
    result_df = result_df.set_index('timestamp')
    
    # 3. Calculate rolling mean and standard deviation
    rolling_window = result_df['heart_rate'].rolling(window=f'{window_size_seconds}s')
    result_df['hr_rolling_mean'] = rolling_window.mean()
    result_df['hr_rolling_std'] = rolling_window.std()
    
    # 4. Reset index to bring timestamp back as a column
    result_df = result_df.reset_index()
    
    # 5. Handle NaN values from rolling calculations
    # Use backward fill first, then forward fill to handle any remaining NaNs
    result_df['hr_rolling_mean'] = result_df['hr_rolling_mean'].fillna(method='bfill').fillna(method='ffill')
    result_df['hr_rolling_std'] = result_df['hr_rolling_std'].fillna(method='bfill').fillna(method='ffill')
    
    # Replace any remaining NaNs in hr_rolling_std with 0 (for cases with constant heart rate)
    result_df['hr_rolling_std'] = result_df['hr_rolling_std'].fillna(0)
    
    print(f"Added rolling features with {window_size_seconds}s window:")
    print(f"  - hr_rolling_mean: mean={result_df['hr_rolling_mean'].mean():.2f}, range=[{result_df['hr_rolling_mean'].min():.2f}, {result_df['hr_rolling_mean'].max():.2f}]")
    print(f"  - hr_rolling_std: mean={result_df['hr_rolling_std'].mean():.2f}, range=[{result_df['hr_rolling_std'].min():.2f}, {result_df['hr_rolling_std'].max():.2f}]")
    
    return result_df

def prepare_data_part2(df_with_features, test_size=0.2, random_state=42):
    """
    Prepare data for modeling with time-series features.
    
    Args:
        df_with_features: DataFrame with original and rolling features
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # 1. Select relevant features including the original features and rolling features
    # For Part 2, we use the same base features as Part 1 plus our time-series features
    features = ['age', 'systolic_bp', 'diastolic_bp', 'glucose_level', 'bmi', 
               'hr_rolling_mean', 'hr_rolling_std']
    X = df_with_features[features]
    
    # 2. Select target variable
    y = df_with_features['disease_outcome']
    
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
    print(f"Data prepared with time-series features: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
    print(f"Features: {', '.join(features)}")
    print(f"Target distribution - Training: {np.bincount(y_train)}, Testing: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of the trees
        random_state: Random seed for reproducibility
        
    Returns:
        Trained Random Forest model
    """
    # Initialize the Random Forest classifier with class balancing
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced',  # Handle class imbalance
        n_jobs=-1  # Use all available cores for parallel training
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Print model information
    print(f"Random Forest model trained with {n_estimators} trees, max depth {max_depth}")
    
    # Get feature importances
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Print top features
    print("Top 3 important features:")
    for i, row in feature_importances.head(3).iterrows():
        print(f"  - {row['feature']}: {row['importance']:.4f}")
    
    return model

def train_xgboost(X_train, y_train, n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42):
    """
    Train an XGBoost classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of boosting rounds
        learning_rate: Boosting learning rate
        max_depth: Maximum depth of a tree
        random_state: Random seed for reproducibility
        
    Returns:
        Trained XGBoost model
    """
    # Calculate scale_pos_weight for handling class imbalance
    # This is an XGBoost-specific approach: ratio of negative to positive samples
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(1, pos_count)  # Avoid division by zero
    
    # Initialize the XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        use_label_encoder=False,  # Avoid deprecated warning
        eval_metric='logloss'  # Standard evaluation metric for binary classification
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Print model information
    print(f"XGBoost model trained with {n_estimators} rounds, learning rate {learning_rate}, max depth {max_depth}")
    
    # Get feature importances
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Print top features
    print("Top 3 important features:")
    for i, row in feature_importances.head(3).iterrows():
        print(f"  - {row['feature']}: {row['importance']:.4f}")
    
    return model

def compare_models(rf_model, xgb_model, X_test, y_test):
    """
    Compare the performance of Random Forest and XGBoost models using AUC.
    
    Args:
        rf_model: Trained Random Forest model
        xgb_model: Trained XGBoost model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with AUC scores
    """
    # Generate probability predictions
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC scores
    rf_auc = roc_auc_score(y_test, rf_probs)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    
    # Print comparison
    print("\nModel Comparison:")
    print(f"Random Forest AUC: {rf_auc:.4f}")
    print(f"XGBoost AUC: {xgb_auc:.4f}")
    
    # Determine the better model
    better_model = "Random Forest" if rf_auc > xgb_auc else "XGBoost"
    improvement = abs(rf_auc - xgb_auc) / min(rf_auc, xgb_auc) * 100
    print(f"The better model is {better_model} by {improvement:.2f}% relative improvement")
    
    # Return results as dictionary
    return {
        'random_forest_auc': rf_auc,
        'xgboost_auc': xgb_auc,
        'better_model': better_model,
        'improvement_percent': improvement
    }

def save_results_part2(comparison_results, output_file='results/results_part2.txt'):
    """
    Save the model comparison results to a text file.
    
    Args:
        comparison_results: Dictionary with AUC scores and comparison
        output_file: Path to save the results
        
    Returns:
        Path to the saved file
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Format the results
    with open(output_file, 'w') as f:
        f.write("Tree-Based Model Comparison Results\n")
        f.write("=================================\n\n")
        f.write(f"Random Forest AUC: {comparison_results['random_forest_auc']:.4f}\n")
        f.write(f"XGBoost AUC: {comparison_results['xgboost_auc']:.4f}\n\n")
        f.write(f"Better Model: {comparison_results['better_model']}\n")
        f.write(f"Relative Improvement: {comparison_results['improvement_percent']:.2f}%\n\n")
        f.write("Time Series Features Impact:\n")
        f.write("These results incorporate time-series features from heart rate data,\n")
        f.write("which may provide predictive signals beyond static measurements.\n")
    
    print(f"Results saved to {output_file}")
    return output_file

# Main execution
if __name__ == "__main__":
    # 1. Load data
    data_file = 'data/synthetic_health_data.csv'
    df = load_data(data_file)
    
    # 2. Extract rolling features
    window_size = 300  # 5 minutes in seconds
    df_with_features = extract_rolling_features(df, window_size)
    
    # 3. Prepare data
    X_train, X_test, y_train, y_test = prepare_data_part2(df_with_features)
    
    # 4. Train models
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # 5. Compare models and calculate AUC scores
    comparison_results = compare_models(rf_model, xgb_model, X_test, y_test)
    
    # 6. Save results
    save_results_part2(comparison_results)
