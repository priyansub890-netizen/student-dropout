import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("../student-mat.csv", sep=";")

data["sex"] = data["sex"].map({"F": 0, "M": 1})
data["schoolsup"] = data["schoolsup"].map({"no": 0, "yes": 1})
data["famsup"] = data["famsup"].map({"no": 0, "yes": 1})
data["internet"] = data["internet"].map({"no": 0, "yes": 1})

data["risk"] = (data["G3"] < 10).astype(int)

X = data[
    [
        "sex",
        "age",
        "schoolsup",
        "famsup",
        "internet",
        "studytime",
        "failures",
        "absences",
        "G1",
        "G2"
    ]
]

y = data["risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "../student_predictor.pkl")

print("Model trained successfully!")
