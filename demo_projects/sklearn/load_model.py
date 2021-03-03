import joblib
import sklearn
model = joblib.load("./models/decision_tree1.pkl")
x = [[2, 4, 1, 5],[1, 2, 3, 1], [7, 8, 3, 6], [4, 8, 4, 7], [2, 5, 6, 9]]
print(model.predict(x))