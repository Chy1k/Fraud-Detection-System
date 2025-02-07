import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_data(file_path, sample_fraction=1.0):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Optional: Use only a fraction of the data for faster tuning
    if sample_fraction < 1.0:
        data = data.sample(frac=sample_fraction, random_state=42)

    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Apply SMOTE to balance classes
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test

# Example usage:
# X_train, X_test, y_train, y_test = preprocess_data('creditcard.csv')