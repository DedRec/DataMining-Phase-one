from preprocessing import *
from algorithms import *


df = pd.read_csv("breast-cancer.csv")

df = clean_data(df)
features = df.iloc[:, 2:]
target = df.iloc[:, 1]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_clean_z, target_clean_z = remove_outliers_Zscore(features_scaled, target)
features_clean_box, target_clean_box = Box_plot_outliers(features_scaled, target, df.columns.values)
x, y = pd.DataFrame(features_clean_box), pd.DataFrame(features_clean_z)
y = y.set_axis(df.columns.values[2:], axis=1, copy=False)
target_clean = target_clean_z[target_clean_z.index.isin(target_clean_box.index)]
features_clean = x[x.radius_mean.isin(y.radius_mean)]


#first: no scaling, second: full preprocessing, third: no preprocessing

X_train2, X_test2, y_train2, y_test2 =train_test_split(features, target, test_size=0.2, random_state=42)
X_train3, X_test3, y_train3, y_test3 =train_test_split(features_clean, target_clean, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = preprocess()

print("+++++++++++++++++++++++++++++++NO Preprocessing++++++++++++++++++++++++++++++++++++++++")
kmediods(X_train2, X_test2, y_train2, y_test2)
print("+++++++++++++++++++++++++++++++Preprocessing but no PCA (Scaling)++++++++++++++++++++++++++++++++++++++++")
kmediods(X_train3, X_test3, y_train3, y_test3)
print("+++++++++++++++++++++++++++++++Preprocessing++++++++++++++++++++++++++++++++++++++++")
kmediods(X_train, X_test, y_train, y_test)

print("+++++++++++++++++++++++++++++++NO Preprocessing++++++++++++++++++++++++++++++++++++++++")
kmeans(X_train2, X_test2, y_train2, y_test2)
print("+++++++++++++++++++++++++++++++Preprocessing but no PCA (Scaling)++++++++++++++++++++++++++++++++++++++++")
kmeans(X_train3, X_test3, y_train3, y_test3)
print("+++++++++++++++++++++++++++++++Preprocessing++++++++++++++++++++++++++++++++++++++++")
kmeans(X_train, X_test, y_train, y_test)

print("+++++++++++++++++++++++++++++++NO Preprocessing++++++++++++++++++++++++++++++++++++++++")
DBscan(X_train2, X_test2, y_train2, y_test2)
print("+++++++++++++++++++++++++++++++Preprocessing but no PCA (Scaling)++++++++++++++++++++++++++++++++++++++++")
DBscan(X_train3, X_test3, y_train3, y_test3)
print("+++++++++++++++++++++++++++++++Preprocessing++++++++++++++++++++++++++++++++++++++++")
DBscan(X_train, X_test, y_train, y_test)

print("+++++++++++++++++++++++++++++++NO Preprocessing++++++++++++++++++++++++++++++++++++++++")
agglomerative_clustering(X_train2, X_test2, y_train2, y_test2)
print("+++++++++++++++++++++++++++++++Preprocessing but no PCA (Scaling)++++++++++++++++++++++++++++++++++++++++")
agglomerative_clustering(X_train3, X_test3, y_train3, y_test3)
print("+++++++++++++++++++++++++++++++Preprocessing++++++++++++++++++++++++++++++++++++++++")
agglomerative_clustering(X_train, X_test, y_train, y_test)


print("+++++++++++++++++++++++++++++++NO Preprocessing++++++++++++++++++++++++++++++++++++++++")
KNN(X_train2, X_test2, y_train2, y_test2)
print("+++++++++++++++++++++++++++++++Preprocessing but no PCA (Scaling)++++++++++++++++++++++++++++++++++++++++")
KNN(X_train3, X_test3, y_train3, y_test3)
print("+++++++++++++++++++++++++++++++Preprocessing++++++++++++++++++++++++++++++++++++++++")
KNN(X_train, X_test, y_train, y_test)

print("+++++++++++++++++++++++++++++++NO Preprocessing++++++++++++++++++++++++++++++++++++++++")
LinearReg(X_train2, X_test2, y_train2, y_test2)
print("+++++++++++++++++++++++++++++++Preprocessing but no PCA (Scaling)++++++++++++++++++++++++++++++++++++++++")
LinearReg(X_train3, X_test3, y_train3, y_test3)
print("+++++++++++++++++++++++++++++++Preprocessing++++++++++++++++++++++++++++++++++++++++")
LinearReg(X_train, X_test, y_train, y_test)

print("+++++++++++++++++++++++++++++++NO Preprocessing++++++++++++++++++++++++++++++++++++++++")
RandomForest(X_train2, X_test2, y_train2, y_test2)
print("+++++++++++++++++++++++++++++++Preprocessing but no PCA (Scaling)++++++++++++++++++++++++++++++++++++++++")
RandomForest(X_train3, X_test3, y_train3, y_test3)
print("+++++++++++++++++++++++++++++++Preprocessing++++++++++++++++++++++++++++++++++++++++")
RandomForest(X_train, X_test, y_train, y_test)