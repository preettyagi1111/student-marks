import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv("data/student_data.csv")

print("===== DATASET LOADED =====")
print("Total records:", len(df))
print()


# ==============================
# 2. SELECT FEATURES
# ==============================

features = [
    "Attendance",
    "StudyHours",
    "PreviousMarks",
    "AssignmentScore",
    "InternalMarks",
    "PracticalMarks",
    "Backlogs"
]

target = "FinalMarks"


# Check required columns
required_columns = features + [target]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    print("ERROR: Missing columns:")
    print(missing_columns)
    raise SystemExit


X = df[features]
y = df[target]


# ==============================
# 3. TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print()


# ==============================
# 4. CREATE MODELS
# ==============================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}


# ==============================
# 5. TRAIN AND COMPARE MODELS
# ==============================

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# ==============================
# 6. DISPLAY RESULTS
# ==============================

results_df = pd.DataFrame(results)

print("===== MODEL COMPARISON =====")
print()

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print()


# ==============================
# 7. SELECT BEST MODEL
# ==============================

best_result = results_df.sort_values(
    by=["MAE", "RMSE"],
    ascending=[True, True]
).iloc[0]

best_model_name = best_result["Model"]

best_model = models[best_model_name]


print("===== BEST MODEL =====")
print("Selected model:", best_model_name)
print("MAE:", round(best_result["MAE"], 2))
print("RMSE:", round(best_result["RMSE"], 2))
print("R2 Score:", round(best_result["R2"], 2))
print()


# ==============================
# 8. SAVE MODEL
# ==============================

model_data = {
    "model": best_model,
    "features": features,
    "target": target
}

joblib.dump(
    model_data,
    "model/student_model.pkl"
)

print("===== MODEL SAVED =====")
print("File: model/student_model.pkl")
print()
print("Training completed successfully!")
