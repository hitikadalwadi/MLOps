import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle
import os
import json
import base64

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
HISTORY_PATH = os.path.join(MODEL_DIR, "metrics_history.json")

TARGET = "target"
PRIMARY_METRIC = "accuracy"
# A drop in the primary metric larger than this is reported as a regression.
REGRESSION_TOLERANCE = 0.01


def _encode(obj) -> str:
    """Pickle an object and return it as a base64 string (JSON-safe for XCom)."""
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def _decode(data_b64: str):
    return pickle.loads(base64.b64decode(data_b64))


def load_data():
    """
    Loads the wine training set from CSV, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized DataFrame (JSON-safe).
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "wine_train.csv"))
    print(f"Loaded training data: {df.shape[0]} rows, {df.shape[1]} columns")
    return _encode(df)


def data_preprocessing(data_b64: str):
    """
    Deserializes the training data, drops incomplete rows, and splits it into
    features and labels.
    Returns:
        str: Base64-encoded pickled dict with keys "X" and "y".
    """
    df = _decode(data_b64).dropna()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Preprocessed data: {X.shape[0]} rows, {X.shape[1]} features")
    return _encode({"X": X, "y": y})


def build_save_model(data_b64: str, filename: str):
    """
    Trains a random forest classifier on the preprocessed data and saves it.
    Returns the model hyperparameters (JSON-serializable) so later tasks can log them.
    """
    data = _decode(data_b64)

    params = {"n_estimators": 150, "max_depth": 3, "random_state": 42}
    model = RandomForestClassifier(**params)
    model.fit(data["X"], data["y"])

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, filename), "wb") as f:
        pickle.dump(model, f)

    print(f"Saved model to {os.path.join(MODEL_DIR, filename)}")
    return params


def evaluate_model(filename: str, model_params: dict, run_id: str = "manual"):
    """
    Loads the saved model, scores it on the held-out wine test set, and appends
    the result to the metrics history.
    Returns:
        dict: JSON-safe metrics for this run.
    """
    with open(os.path.join(MODEL_DIR, filename), "rb") as f:
        model = pickle.load(f)

    test_df = pd.read_csv(os.path.join(DATA_DIR, "wine_test.csv")).dropna()
    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET]
    pred = model.predict(X_test)

    metrics = {
        "run_id": run_id,
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "n_test": int(len(y_test)),
        "params": model_params,
    }
    print(f"Evaluation for {run_id}: accuracy={metrics['accuracy']:.4f}, "
          f"f1_macro={metrics['f1_macro']:.4f}")

    history = _read_history()
    # Replace any entry from the same run so task retries don't duplicate it.
    history = [h for h in history if h["run_id"] != run_id]
    history.append(metrics)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    return metrics


def compare_with_previous_run(metrics: dict):
    """
    Compares this run's metrics against the previous run in the history file.
    Returns:
        dict: JSON-safe summary with the deltas and a verdict
        ("baseline", "improved", "unchanged" or "regressed").
    """
    history = _read_history()
    earlier = [h for h in history if h["run_id"] != metrics["run_id"]]

    if not earlier:
        print("No previous run found. This run is the baseline.")
        return {"verdict": "baseline", "current": metrics, "previous": None}

    previous = earlier[-1]
    deltas = {
        name: round(metrics[name] - previous[name], 4)
        for name in ("accuracy", "f1_macro")
    }
    change = deltas[PRIMARY_METRIC]
    if change > REGRESSION_TOLERANCE:
        verdict = "improved"
    elif change < -REGRESSION_TOLERANCE:
        verdict = "regressed"
    else:
        verdict = "unchanged"

    print(f"Previous run {previous['run_id']}: accuracy={previous['accuracy']:.4f}, "
          f"f1_macro={previous['f1_macro']:.4f}")
    print(f"Current run  {metrics['run_id']}: accuracy={metrics['accuracy']:.4f}, "
          f"f1_macro={metrics['f1_macro']:.4f}")
    print(f"Deltas: {deltas} -> {verdict.upper()}")
    return {"verdict": verdict, "deltas": deltas, "current": metrics, "previous": previous}


def _read_history() -> list:
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH) as f:
        return json.load(f)
