
import joblib
import sys

try:
    cols = joblib.load('models/feature_columns.pkl')
    with open('features.txt', 'w') as f:
        f.write(str(cols))
    print("Features written to features.txt")
except Exception as e:
    print(e)
