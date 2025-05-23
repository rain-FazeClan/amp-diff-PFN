import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Make sure numpy is imported

MODELS_DIR = 'models'
PREDICTIVE_MODEL_FILE = 'predictive_model.pkl'
RESULTS_DIR = 'results/predictive_evaluation' # Load test data and save plots from here

def evaluate_classifier(model_path, test_features_path, test_labels_path, results_dir):
    """
    Loads a trained classifier and test data, evaluates performance, and saves results.
    """
    try:
        # Load the model
        print(f"Loading classifier model from {model_path}")
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        print("Model loaded successfully.")

        # Load test data
        print(f"Loading test data from {test_features_path} and {test_labels_path}")
        X_test = pd.read_csv(test_features_path)
        y_test = pd.read_csv(test_labels_path).squeeze() # Squeeze to get Series

        # Ensure columns match the training data if necessary (important for some models, though less so for TabPFN on pure numeric)
        # This assumes train_predictive saved the X_test with consistent columns
        print(f"Test data shape: {X_test.shape}")


        # Evaluate
        print("Evaluating Classifier on test data...")
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)[:, 1] # Probability of positive class

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:\n", cm)
        print(f"ROC AUC: {roc_auc:.4f}")

        # Save evaluation results
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'classification_report_test.txt'), 'w') as f:
             f.write(report)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg (0)', 'Pos (1)'], yticklabels=['Neg (0)', 'Pos (1)'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix (Test Set)')
        plt.savefig(os.path.join(results_dir, 'confusion_matrix_test.png'))

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve (Test Set)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(results_dir, 'roc_curve_test.png'))
        print(f"Test evaluation results saved to {results_dir}")


    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Ensure train_predictive.py has been run and saved the model and test data.")
    except ImportError:
         print("Error: Required libraries (TabPFN, scikit-learn) not found.")
    except Exception as e:
        print(f"An error occurred during classifier evaluation: {e}")


if __name__ == '__main__':
    model_path = os.path.join(MODELS_DIR, PREDICTIVE_MODEL_FILE)
    test_features_path = os.path.join(RESULTS_DIR, 'X_test.csv') # Load from saved test data
    test_labels_path = os.path.join(RESULTS_DIR, 'y_test.csv')
    eval_results_dir = RESULTS_DIR

    evaluate_classifier(model_path, test_features_path, test_labels_path, eval_results_dir)