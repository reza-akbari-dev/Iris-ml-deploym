from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# Find the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Create the model file path
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "iris_model.pkl"


def train_model() -> None:
    """Train an Iris classification model and save it."""

    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=iris.target_names,
        )
    )

    # Create the model directory if it does not exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save the model and useful information together
    model_data = {
        "model": model,
        "target_names": iris.target_names.tolist(),
        "feature_names": iris.feature_names,
    }

    joblib.dump(model_data, MODEL_PATH)

    print(f"Model saved successfully at: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()