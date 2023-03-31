import pandas as pd
import joblib

df = pd.read_csv("labb/test_sample.csv")
model = joblib.load("labb/model.pkl")


X_test = df.drop("cardio", axis=1)
y_test = df["cardio"]

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

prediction = {"prediction": y_pred, "probability class 0": y_proba[:,0], "prediction class 1": y_proba[:,1]}
df_prediction = pd.DataFrame(prediction)

df_prediction.to_csv("labb/prediction.csv", index=False)