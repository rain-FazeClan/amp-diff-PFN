import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
# TabPFN requires specific imports and potentially setup
# You might need to download the model weights: https://github.com/automl/TabPFN#installation
# from tabpfn import TabPFNClassifier # Make sure tabpfn is installed and model loaded
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns # Optional, for plotting

CLASSIFY_DATA_DIR = 'classify_data'
CLASSIFY_DATA_FILE = 'classify_data.csv'
MODELS_DIR = 'models'
PREDICTIVE_MODEL_FILE = 'predictive_model.pkl'
RESULTS_DIR = 'results/predictive_evaluation' # For evaluation plots/reports

# Set device for TabPFN (it runs on GPU if available)
# TabPFN might have its own device management; check its documentation
# Here we assume it might use CUDA if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Not directly needed for classifier object, but important for TPFN

def train_classifier(data_path, model_output_path, results_dir):
    """
    Loads featurized data, trains a TabPFN classifier, and evaluates it.
    Saves the trained model and evaluation results.
    """
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Assuming 'Sequence' and 'label' are in the dataframe
    # Features are all other columns
    X = df.drop(['Sequence', 'label'], axis=1)
    y = df['label']

    # Handle potential non-numeric columns or NaNs from feature generation
    # Simple handling: drop columns that are not numeric
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    X = X[numeric_cols]
    X = X.fillna(X.mean()) # Simple imputation for NaNs

    print(f"Using {X.shape[1]} features for classification.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # Stratify for imbalanced data

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # Initialize TabPFN Classifier
    # Note: You might need to download weights first based on TabPFN install instructions
    # N_ensemble_configs could be adjusted, default is 1
    # TabPFN is sensitive to dataset size. For very large datasets (>10000), it might sample.
    try:
         from tabpfn import TabPFNClassifier
         # Ensure model weights are downloaded or specified
         # classifier = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu', N_ensemble_configs=1)
         classifier = TabPFNClassifier(device='cpu', N_ensemble_configs=1) # Use cpu for wider compatibility, adjust if needed

         print("Training TabPFN Classifier...")
         classifier.fit(X_train, y_train)
         print("Training complete.")

         # Evaluate
         print("Evaluating Classifier...")
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
         with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
             f.write(report)
         plt.figure(figsize=(8, 6))
         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg (0)', 'Pos (1)'], yticklabels=['Neg (0)', 'Pos (1)'])
         plt.ylabel('Actual')
         plt.xlabel('Predicted')
         plt.title('Confusion Matrix')
         plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))

         plt.figure(figsize=(8, 6))
         plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
         plt.xlim([0.0, 1.0])
         plt.ylim([0.0, 1.05])
         plt.xlabel('False Positive Rate')
         plt.ylabel('True Positive Rate')
         plt.title('Receiver Operating Characteristic (ROC) Curve')
         plt.legend(loc="lower right")
         plt.savefig(os.path.join(results_dir, 'roc_curve.png'))
         print(f"Evaluation results saved to {results_dir}")

         # Save the trained model
         os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
         with open(model_output_path, 'wb') as f:
             pickle.dump(classifier, f)
         print(f"Trained classifier saved to {model_output_path}")

         # Optional: Save test data indices to ensure evaluate_predictive uses the same split
         X_test.to_csv(os.path.join(results_dir, 'X_test.csv'), index=False)
         y_test.to_csv(os.path.join(results_dir, 'y_test.csv'), index=False)
         print(f"Test data saved to {results_dir}")

    except ImportError:
         print("Error: TabPFNClassifier not found. Please install it (`pip install tabpfn`).")
         print("Also, ensure you have downloaded the required model weights as per TabPFN documentation.")
         print("Skipping classifier training.")
    except Exception as e:
         print(f"An error occurred during classifier training: {e}")


if __name__ == '__main__':
    data_path = os.path.join(CLASSIFY_DATA_DIR, CLASSIFY_DATA_FILE)
    model_path = os.path.join(MODELS_DIR, PREDICTIVE_MODEL_FILE)
    eval_results_dir = RESULTS_DIR

    train_classifier(data_path, model_path, eval_results_dir)