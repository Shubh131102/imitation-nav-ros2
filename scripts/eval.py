from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from data_prep import load_dataset

def main():
    model_path = Path("models/bc_mlp.joblib")
    if not model_path.exists():
        raise FileNotFoundError("models/bc_mlp.joblib not found. Train first: python scripts/train.py")

    X, y = load_dataset("data")
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(model_path)
    y_pred = model.predict(X_te)

    mse = mean_squared_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)
    print(f"Eval MSE: {mse:.6f} | R2: {r2:.4f}")

if __name__ == "__main__":
    main()
