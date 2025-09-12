from pathlib import Path
import sys
# allow "from data_prep import load_dataset" when running as a script
sys.path.append(str(Path(__file__).resolve().parent))

import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from data_prep import load_dataset

def main():
    X, y = load_dataset("data")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                         solver="adam", max_iter=500, random_state=42)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    mse = mean_squared_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)

    print(f"Test MSE: {mse:.6f} | R2: {r2:.4f}")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/bc_mlp.joblib")
    print("Saved -> models/bc_mlp.joblib")

if __name__ == "__main__":
    main()
