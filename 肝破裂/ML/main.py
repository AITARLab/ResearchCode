import pandas as pd
from train import train
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

data_file = "data/mydata - 副本 (3).xlsx"
model_list = ["SVC", "RandomForest", "KN", "GaussianNB", "AdaBoost", "GradientBoost", "XGBoost"]

# for model in model_list:
#     train(model, data_file, task="liver_surgery")
df = pd.read_excel(data_file).drop([0])
X = df["patient"]
y = df["surgery"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)
print(X_test,y_test)