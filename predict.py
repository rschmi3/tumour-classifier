import argparse

import cv2
import joblib
import numpy as np

from feature_extraction import extract_combined_features


def predict(image_path, model_path="best_model.pkl", image_size=(128, 128)):
    """
    Predict class for a new image

    Parameters:
    -----------
    image_path : str
        Path to the image
    model_path : str
        Path to saved model
    """
    # Load model
    model_data = joblib.load(model_path)
    model = model_data["model"]
    scaler = model_data["scaler"]
    label_encoder = model_data["label_encoder"]

    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Predict image is none")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, image_size)

    # Extract features (using combined features)
    features = extract_combined_features(np.array([img]))
    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    predicted_class = label_encoder.inverse_transform([prediction])[0]

    print(f"\nPrediction for {image_path}:")
    print(f"  Class: {predicted_class}")
    print(f"  Confidence: {probabilities[prediction]:.4f}")
    print("\nAll probabilities:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {class_name}: {probabilities[i]:.4f}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Tumour classification project")

    parser.add_argument("--image_path", type=str, help="Path for image to predict on")
    parser.add_argument("--model_path", type=str, help="Path for model to predict with")
    parser.add_argument(
        "--image_size",
        type=int,
        help="dimension to resize images to.  Images resized to square.",
        default=128,
    )
    args = parser.parse_args()

    print(args)

    image_size = (args.image_size, args.image_size)
    predict(args.image_path, args.model_path, image_size)


if __name__ == "__main__":
    main()
