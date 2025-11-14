import argparse
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

from feature_extraction import *
from setup_data import *

random_state = 42

# Set random seed for reproducibility
np.random.seed(random_state)


class BrainTumorClassifier:
    def __init__(
        self,
        train_data: tuple[np.ndarray, np.ndarray],
        valid_data: tuple[np.ndarray, np.ndarray],
        classes: set[str],
        cuda: bool = False,
    ):
        """
        Initialize the Brain Tumor Classifier

        Parameters:
        -----------
        train_data : tuple[np.ndarray, np.ndarray]
            tuple of X and y that makes up the training dataset
        valid_data : tuple[np.ndarray, np.ndarray]
            tuple of X and y that makes up the validation dataset
        classes : set[str]
            object class labels
        """
        self.train_data = train_data
        self.valid_data = valid_data
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.classes = classes
        self.cuda = cuda

        # Extract shape and assert it's a valid tuple
        shape = train_data[0].shape
        assert len(shape) >= 2, "Image array must have at least 2 dimensions"
        self.image_size: tuple[int, int] = (shape[0], shape[1])

    def prepare_data(self, feature_type="combined"):
        """
        Prepare data with specified feature extraction

        Parameters:
        -----------
        feature_type : str
            Type of features to extract: 'hog', 'lbp', 'color', or 'combined'
        """
        # Load images
        X_train, y_train = self.train_data

        X_val, y_val = self.valid_data

        # Extract features
        if feature_type == "hog":
            X_train = extract_hog_features(X_train)
            X_val = extract_hog_features(X_val)
        elif feature_type == "lbp":
            X_train = extract_lbp_features(X_train)
            X_val = extract_lbp_features(X_val)
        elif feature_type == "color":
            X_train = extract_color_features(X_train)
            X_val = extract_color_features(X_val)
        elif feature_type == "combined":
            X_train = extract_combined_features(X_train)
            X_val = extract_combined_features(X_val)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        # Encode labels
        y_train = self.label_encoder.fit_transform(y_train)
        y_val = self.label_encoder.transform(y_val)

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        print(f"\nData split:")
        print(f"  Training set: {np.shape(X_train)}")
        print(f"  Validation set: {np.shape(X_val)}")

        return X_train, X_val, y_train, y_val

    def initialize_models(self):
        """Initialize multiple classification models"""
        self.models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            ),
            "Gradient Boosting": xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state,
                device="cuda" if self.cuda else "cpu",
            ),
            "SVM (RBF)": SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True,
                random_state=random_state,
            ),
            "SVM (Linear)": SVC(
                kernel="linear",
                C=1,
                probability=True,
                random_state=random_state,
            ),
            "K-Nearest Neighbors": KNeighborsClassifier(
                n_neighbors=5, weights="distance", n_jobs=-1
            ),
        }
        print(f"\nInitialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")

    def train_models(self, X_train, y_train):
        """Train all models and evaluate on validation set"""
        print("\n" + "=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)

        for name, model in self.models.items():
            train_start = time.time()
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            train_end = time.time()
            self.results[name] = {}
            self.results[name]["training_time"] = train_end - train_start
            print(self.results[name]["training_time"])

    def evaluate_models(self, X_val, y_val):
        """Evaluate all trained models on test set"""
        print("\n" + "=" * 60)
        print("EVALUATING MODELS ON VALIDATION SET")
        print("=" * 60)

        for name, model in self.models.items():
            print(f"\n{name}:")
            print("-" * 40)

            # Predict
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)

            # Accuracy
            accuracy = accuracy_score(y_val, y_pred)
            print(f"Accuracy: {accuracy:.4f}")

            # Classification report
            print("\nClassification Report:")
            print(
                classification_report(
                    y_val, y_pred, target_names=self.label_encoder.classes_
                )
            )

            # Confusion matrix
            cm = confusion_matrix(y_val, y_pred)

            # Calculate AUC-ROC for multi-class
            n_classes = len(self.label_encoder.get_params())
            y_val_bin = label_binarize(y=y_val, classes=range(n_classes))

            if (y_val_bin is None) and y_val_bin.shape[1] > 2:
                auc_roc = roc_auc_score(
                    y_val_bin, y_proba, average="macro", multi_class="ovr"
                )
            else:
                auc_roc = roc_auc_score(
                    y_val, y_proba, average="macro", multi_class="ovr"
                )
            print(f"\nAUC-ROC (macro): {auc_roc:.4f}")

            # Store test results
            self.results[name] = {}
            self.results[name]["test_accuracy"] = accuracy
            self.results[name]["test_predictions"] = y_pred
            self.results[name]["test_probabilities"] = y_proba
            self.results[name]["confusion_matrix"] = cm
            self.results[name]["auc_roc"] = auc_roc

    def plot_results(self, save_dir="results"):
        """Create visualizations of model performance"""
        os.makedirs(save_dir, exist_ok=True)

        # 1. Model comparison bar chart
        names = list(self.results.keys())
        accuracies = [self.results[name]["test_accuracy"] for name in names]
        auc_scores = [self.results[name]["auc_roc"] for name in names]

        x = np.arange(len(names))
        width = 0.35

        _, ax1 = plt.subplots(figsize=(12, 6))

        ax1.bar(
            x - width / 2,
            accuracies,
            width,
            label="Accuracy",
            alpha=0.8,
            color="blue",
        )
        ax1.set_xlabel("Model")
        ax1.set_ylabel("Accuracy", color="blue")
        ax1.tick_params(axis="y", labelcolor="blue")

        ax2 = ax1.twinx()
        ax2.bar(
            x + width / 2, auc_scores, width, label="AUC-ROC", alpha=0.8, color="red"
        )
        ax2.set_ylabel("AUC-ROC Score", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        plt.title("Model Performance Comparison")
        plt.xticks(x, names, rotation=0, ha="center")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        plt.ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/model_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Confusion matrices
        n_models = len(self.results)
        _, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, (name, result) in enumerate(self.results.items()):
            if idx >= len(axes):
                break
            cm = result["confusion_matrix"]
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[idx],
                xticklabels=list(self.classes),
                yticklabels=list(self.classes),
            )
            axes[idx].set_title(f'{name}\nAccuracy: {result["test_accuracy"]:.4f}')
            axes[idx].set_ylabel("True Label")
            axes[idx].set_xlabel("Predicted Label")

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrices.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\nPlots saved to {save_dir}/")

    def save_models(self):
        for name, model in self.models.items():
            filename = name.replace(" ", "_")
            filename = filename.replace(")", "")
            filename = filename.replace("(", "")
            filename = filename + ".pkl"
            print(f"\nSaving model: {name} to {filename}")

            # Save model, scaler, and label encoder
            model_data = {
                "model": model,
                "scaler": self.scaler,
                "label_encoder": self.label_encoder,
                "model_name": name,
                "accuracy": self.results[name]["test_accuracy"],
                "auc_roc": self.results[name]["auc_roc"],
            }

            joblib.dump(model_data, filename)

        return self.save_best_model("best_brain_tumor_model.pkl")

    def save_best_model(self, save_path="best_model.pkl"):
        """Save the best performing model"""
        # Find best model by test accuracy
        best_name = max(
            self.results.keys(), key=lambda x: self.results[x]["test_accuracy"]
        )
        best_model = self.models[best_name]

        # Save model, scaler, and label encoder
        model_data = {
            "model": best_model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "model_name": best_name,
            "accuracy": self.results[best_name]["test_accuracy"],
            "auc_roc": self.results[best_name]["auc_roc"],
        }

        joblib.dump(model_data, save_path)
        print(f"\nBest model ({best_name}) saved to {save_path}")
        print(f"  Accuracy: {self.results[best_name]['test_accuracy']:.4f}")
        print(f"  AUC-ROC: {self.results[best_name]['auc_roc']:.4f}")

        return best_name


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Tumour classification project")

    parser.add_argument("--cuda", action="store_true", help="Enable cuda for xgboost")
    args = parser.parse_args()
    print("=" * 60)
    print(f"BRAIN TUMOR CLASSIFICATION {'(CUDA)' if args.cuda else ''}")
    print("=" * 60)

    dataset_path = download_dataset("deeppythonist/brain-tumor-mri-dataset")

    train_data, valid_data, classes = load_images_and_labels(
        dataset_path=dataset_path, image_size=(128, 128)
    )

    # Initialize classifier
    classifier = BrainTumorClassifier(
        train_data=train_data, valid_data=valid_data, classes=classes, cuda=True
    )

    # Download and prepare data
    X_train, X_val, y_train, y_val = classifier.prepare_data(feature_type="combined")

    # Initialize and train models
    classifier.initialize_models()
    classifier.train_models(X_train, y_train)

    # Evaluate on test set
    classifier.evaluate_models(X_val, y_val)

    # Create visualizations
    classifier.plot_results()

    # Save models
    best_model_name = classifier.save_models()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest Model: {best_model_name}")
    print("\nResults saved to 'results/' directory")
    print("Best model saved to 'best_brain_tumor_model.pkl'")


if __name__ == "__main__":
    main()
