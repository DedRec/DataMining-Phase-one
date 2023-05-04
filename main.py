from Project import *
from algorithms import *


df = pd.read_csv("breast-cancer.csv")
df = clean_data(df)
features = df.iloc[:, 2:]
target = df.iloc[:, 1]


logistic_regression(features, target)

#decision_tree(features, target)

#naive_bayes(features, target)

#support_vector_machines(features, target)
