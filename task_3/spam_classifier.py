"""
Spam vs. Non-Spam Email Classifier

This script trains a machine learning model to distinguish between spam
and non-spam emails using TF-IDF features and Logistic Regression.

It can:
- train a model on data/spam.csv
- save the trained model to models/spam_classifier.joblib
- generate evaluation plots under outputs/
- run predictions on custom email text from the command line
"""

import argparse
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)

DATA_PATH = os.path.join("data", "spam.csv")
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(MODEL_DIR, "spam_classifier.joblib")


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the spam dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Make sure data/spam.csv exists with columns 'label' and 'text'."
        )
    df = pd.read_csv(path)
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain 'label' and 'text' columns.")
    return df


def build_pipeline() -> Pipeline:
    """Create a text classification pipeline with TF-IDF and Logistic Regression."""
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=10000,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


def train_model() -> None:
    """Train the spam classifier and save model and visualizations."""
    print("Loading dataset...")
    df = load_dataset()
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    print("Splitting train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Building pipeline...")
    pipeline = build_pipeline()

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating on validation set...")
    y_pred = pipeline.predict(X_val)
    print("\nClassification report:")
    print(classification_report(y_val, y_pred, digits=4))

    # Ensure directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save model
    print(f"\nSaving model to {MODEL_PATH} ...")
    joblib.dump(pipeline, MODEL_PATH)

    # Confusion matrix visualization
    # We explicitly specify the order: non-spam (first), spam (second)
    labels = ["non-spam", "spam"]
    cm = confusion_matrix(y_val, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(colorbar=False)
    plt.title("Spam Classifier – Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # ROC curve visualization (treat spam as positive class)
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        y_proba = pipeline.predict_proba(X_val)[:, 1]
        y_true_bin = (y_val == "spam").astype(int)

        fpr, tpr, _ = roc_curve(y_true_bin, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Spam Classifier – ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"ROC curve saved to {roc_path}")
    else:
        print("Model does not support predict_proba; skipping ROC curve.")


def predict_email(text: str) -> None:
    """Load a trained model and predict spam vs non-spam for a single email."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run with --mode train first."
        )

    print(f"Loading model from {MODEL_PATH} ...")
    pipeline: Pipeline = joblib.load(MODEL_PATH)

    pred = pipeline.predict([text])[0]
    proba = None
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        proba = pipeline.predict_proba([text])[0]

    print("\nPrediction:")
    print(f"  Label: {pred}")
    if proba is not None:
        nonspam_prob, spam_prob = proba
        print(f"  P(non-spam) = {nonspam_prob:.3f}")
        print(f"  P(spam)     = {spam_prob:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and use a spam vs. non-spam email classifier."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        required=True,
        help="Mode: 'train' a new model or 'predict' for a single email.",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Email text to classify when using --mode predict. "
        "If omitted, you will be prompted to paste text.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "train":
        train_model()
    elif args.mode == "predict":
        if args.text is None:
            print("Paste the email text below. Finish with Ctrl+D (Linux/macOS) or Ctrl+Z then Enter (Windows):")
            try:
                text = "".join(iter(input, ""))
            except EOFError:
                text = ""
        else:
            text = args.text

        if not text.strip():
            print("No text provided; aborting prediction.")
            return

        predict_email(text)


if __name__ == "__main__":
    main()
