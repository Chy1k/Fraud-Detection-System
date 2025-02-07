# Import necessary libraries
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Define the Random Forest model
    rf_model = RandomForestClassifier(random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100],  # Number of trees in the forest
        'max_depth': [None, 10],  # Max depth of the tree
        'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2],    # Minimum number of samples required to be at a leaf node
        'bootstrap': [True, False]        # Whether to use bootstrap samples for building trees
    }

    # Setup GridSearchCV
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                               cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

    # Fit the model with GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    print("Best Hyperparameters:", grid_search.best_params_)
    best_rf_model = grid_search.best_estimator_

    # Make predictions with the best model
    y_pred_best = best_rf_model.predict(X_test)

    # Evaluate the best model
    accuracy = accuracy_score(y_test, y_pred_best)
    precision = precision_score(y_test, y_pred_best)
    recall = recall_score(y_test, y_pred_best)
    f1 = f1_score(y_test, y_pred_best)

    # Print the evaluation metrics for the best model
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Print the classification report
    print("\nClassification Report (Best Model):")
    print(classification_report(y_test, y_pred_best))

    # Print the confusion matrix
    print("\nConfusion Matrix (Best Model):")
    print(confusion_matrix(y_test, y_pred_best))

    # Visualize the confusion matrix
    cm_best = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix (Best Model)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    best_model = grid_search.best_estimator_

    # Save the best model to a file
    joblib.dump(best_model, 'fraud_detection_model.pkl')
    print("Best model saved as 'fraud_detection_model.pkl'")
