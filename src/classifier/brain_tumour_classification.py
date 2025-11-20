# ruff: noqa: E402

import argparse
import os
import random
import time
from pathlib import Path

random_state = 42
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# For GPU determinism (makes things slower but reproducible)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import xgboost as xgb
from classifier.feature_extraction import (
    extract_color_features,
    extract_combined_features,
    extract_hog_features,
    extract_lbp_features,
)
from classifier.neural_nets import TumourNet, TumourNetWrapper
from classifier.setup_data import download_dataset, load_images_and_labels
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

# Set random seed for reproducibility
random.seed(random_state)
np.random.seed(random_state)
tf.random.set_seed(random_state)
tf.config.experimental.enable_op_determinism()

# Set TensorFlow to only allocate what it needs
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class BrainTumorClassifier:
    def __init__(
        self,
        train_data: tuple[np.ndarray, np.ndarray],
        valid_data: tuple[np.ndarray, np.ndarray],
        classes: set[str],
        model_files: dict[str, Path],
        results_dir: Path,
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
        self.conventional_models = {}
        self.neural_networks = {}
        self.results = {}
        self.classes = classes
        self.cuda = cuda
        self.model_files = model_files
        self.results_dir = results_dir

        # Extract shape and assert it's a valid tuple
        shape = train_data[0].shape[1:]
        assert len(shape) >= 2, "Image array must have at least 2 dimensions"
        self.image_shape: tuple[int, int, int] = shape

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

        print("\nData split:")
        print(f"  Training set: {np.shape(X_train)}")
        print(f"  Validation set: {np.shape(X_val)}")

        return X_train, X_val, y_train, y_val

    def initialize_models(self, random_state):
        self.initialize_conventional_models(random_state)
        self.initialize_neural_networks()
        print(
            f"\nInitialized {len(self.conventional_models) + len(self.neural_networks)} models:"
        )
        for name in self.conventional_models.keys():
            print(f"  - {name}")
        for name in self.neural_networks.keys():
            print(f"  - {name}")

    def initialize_conventional_models(self, random_state):
        """Initialize multiple classification models"""
        self.conventional_models = {
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

    def initialize_neural_networks(self):
        """Initialize neural network model"""
        model = TumourNet()
        dummy_input = np.zeros((1, *self.image_shape), dtype=np.float32)
        _ = model(dummy_input, training=True)
        self.neural_networks["Neural_1"] = TumourNetWrapper(model)
        model.summary()

    def train_models(self, X_train, y_train):
        """Train all models and evaluate"""
        print("\n" + "=" * 60)
        print("TRAINING MODELS")
        print("=" * 60)

        y_train_neural = self.label_encoder.transform(self.train_data[1])

        self.train_conventional_models(X_train, y_train)
        self.train_neural_networks(self.train_data[0], y_train_neural)

    def train_conventional_models(self, X_train, y_train):
        """Train all conventional models"""
        for name, model in self.conventional_models.items():
            train_start = time.time()
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            train_end = time.time()
            self.results[name] = {}
            self.results[name]["training_time"] = train_end - train_start
            print(self.results[name]["training_time"])

    def train_neural_networks(self, X_train, y_train):
        """Train neural network"""
        for name, model in self.neural_networks.items():
            print(f"\nTraining {name}...")
            model.compile()
            model.fit(X_train, y_train, epochs=100)

    def evaluate_models(self, X_val, y_val):
        print("\n" + "=" * 60)
        print("EVALUATING MODELS ON VALIDATION SET")
        print("=" * 60)

        y_val_neural = self.label_encoder.fit_transform(self.valid_data[1])

        self.evaluate_conventional_models(X_val, y_val)
        self.evaluate_neural_networks(self.valid_data[0], y_val_neural)
        return self.determine_best_model()

    def evaluate_model(self, name, model, X_val, y_val):
        """Evaluate model and store results"""
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
            auc_roc = roc_auc_score(y_val, y_proba, average="macro", multi_class="ovr")
        print(f"\nAUC-ROC (macro): {auc_roc:.4f}")

        # Store test results
        self.results[name] = {}
        self.results[name]["test_accuracy"] = accuracy
        self.results[name]["test_predictions"] = y_pred
        self.results[name]["test_probabilities"] = y_proba
        self.results[name]["confusion_matrix"] = cm
        self.results[name]["auc_roc"] = auc_roc

    def evaluate_conventional_models(self, X_val, y_val):
        """Evaluate conventional models"""
        for name, model in self.conventional_models.items():
            self.evaluate_model(name, model, X_val, y_val)

    def evaluate_neural_networks(self, X_val, y_val):
        """Evaluate neural network"""
        for name, model in self.neural_networks.items():
            self.evaluate_model(name, model, X_val, y_val)

    def plot_results(self):
        """Create visualizations of model performance"""

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
        model_comparison_path = self.results_dir / "model_comparison.png"
        plt.savefig(model_comparison_path, dpi=300, bbox_inches="tight")
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
            axes[idx].set_title(f"{name}\nAccuracy: {result['test_accuracy']:.4f}")
            axes[idx].set_ylabel("True Label")
            axes[idx].set_xlabel("Predicted Label")

        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        confusion_matrix_path = self.results_dir / "model_comparison.png"
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\nPlots saved to {self.results_dir}")

    def save_conventional_model(self, name, model):
        filename = self.model_files[name]
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

    def save_neural_network(self, name, model):
        filename = self.model_files[name]
        model.save_weights(filename)

    def save_models(self):
        for name, model in self.conventional_models.items():
            self.save_conventional_model(name, model)
        for name, model in self.neural_networks.items():
            self.save_neural_network(name, model)

    def determine_best_model(self):
        """Save the best performing model"""
        # Find best model by test accuracy
        best_name = max(
            self.results.keys(), key=lambda x: self.results[x]["test_accuracy"]
        )

        print(f"\nBest model ({best_name})")
        print(f"  Accuracy: {self.results[best_name]['test_accuracy']:.4f}")
        print(f"  AUC-ROC: {self.results[best_name]['auc_roc']:.4f}")

        return best_name

    def load_conventional_models(self):
        for name, _ in self.conventional_models.items():
            model_data = joblib.load(self.model_files[name])
            self.conventional_models[name] = model_data["model"]

    def load_neural_networks(self):
        for name, model in self.neural_networks.items():
            model.load_weights(self.model_files[name])

    def load_models(self):
        self.load_conventional_models()
        self.load_neural_networks()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Tumour classification project")

    parser.add_argument("--cuda", action="store_true", help="Enable cuda for xgboost")
    parser.add_argument(
        "--train", action="store_true", help="Train new models or load existing ones"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save models",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save results",
    )

    args = parser.parse_args()
    print("=" * 60)
    print(f"BRAIN TUMOR CLASSIFICATION {'(CUDA)' if args.cuda else ''}")
    print("=" * 60)

    args.models_dir.mkdir(exist_ok=True)
    args.results_dir.mkdir(exist_ok=True)

    model_files: dict[str, Path] = {
        "Random Forest": args.models_dir / "Random_Forest.pkl",
        "SVM (RBF)": args.models_dir / "SVM_RBF.pkl",
        "SVM (Linear)": args.models_dir / "SVM_Linear.pkl",
        "Gradient Boosting": args.models_dir / "Gradient_Boosting.pkl",
        "K-Nearest Neighbors": args.models_dir / "K-Nearest_Neighbors.pkl",
        "Neural_1": args.models_dir / "Neural_1.weights.h5",
    }

    dataset_path = download_dataset("deeppythonist/brain-tumor-mri-dataset")

    train_data, valid_data, classes = load_images_and_labels(
        dataset_path=dataset_path, image_size=(128, 128)
    )

    # Initialize classifier
    classifier = BrainTumorClassifier(
        train_data=train_data,
        valid_data=valid_data,
        classes=classes,
        model_files=model_files,
        results_dir=args.results_dir,
        cuda=args.cuda,
    )

    # Initialize and train models
    classifier.initialize_models(random_state)

    # Download and prepare data
    X_train, X_val, y_train, y_val = classifier.prepare_data(feature_type="combined")

    if args.train:
        classifier.train_models(X_train, y_train)
    else:
        classifier.load_models()

    best_model_name = classifier.evaluate_models(X_val, y_val)

    # Create visualizations
    classifier.plot_results()

    if args.train:
        # Save models
        classifier.save_models()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nBest Model: {best_model_name}")
    print(f"\nResults saved to {args.results_dir} directory")


if __name__ == "__main__":
    main()
