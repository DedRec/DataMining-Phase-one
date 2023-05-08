from Project import *
from algorithms import *


df = pd.read_csv("breast-cancer.csv")
df = clean_data(df)
features = df.iloc[:, 2:]
target = df.iloc[:, 1]
features_clean_z, target_clean_z = remove_outliers_Zscore(features, target)
features_clean_box, target_clean_box = Box_plot_outliers(features, target, df.columns.values)
x, y = pd.DataFrame(features_clean_box), pd.DataFrame(features_clean_z)
y = y.set_axis(df.columns.values[2:], axis=1, copy=False)
target_clean = target_clean_z[target_clean_z.index.isin(target_clean_box.index)]
features_clean = x[x.radius_mean.isin(y.radius_mean)]

#first: no scaling, second: full preprocessing, third: no preprocessing
X_train, X_test, y_train, y_test = preprocess()
X_train2, X_test2, y_train2, y_test2 =train_test_split(features, target, test_size=0.2, random_state=42)



#logistic_regression(X_train, X_test, y_train, y_test)
#logistic_regression(X_train2, X_test2, y_train2, y_test2)

# decision_tree(X_train, X_test, y_train, y_test)
# decision_tree(X_train2, X_test2, y_train2, y_test2)

#naive_bayes(X_train, X_test, y_train, y_test)
#naive_bayes(X_train2, X_test2, y_train2, y_test2)

#support_vector_machines(X_train, X_test, y_train, y_test)
#support_vector_machines(X_train2, X_test2, y_train2, y_test2)

# KNN(X_train, X_test, y_train, y_test)
# KNN(X_train2, X_test2, y_train2, y_test2)