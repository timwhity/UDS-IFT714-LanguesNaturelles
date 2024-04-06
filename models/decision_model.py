import pandas as pd
import joblib

data = pd.read_csv('../data/features_dataset_1/splits/test.csv')
test_sample = data.sample(30)

model, ref_cols, target = joblib.load("trained/exp_model.pkl")

X_new = test_sample[ref_cols]
y_new = test_sample[target]

prediction = model.predict(X_new)