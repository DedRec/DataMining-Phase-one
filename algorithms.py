from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve,  auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
#---------------------------------------------Clustering----------------------------------------------

#           kmeans - kmediod

#           DBscan

#           Hierarchical clustering

#           Gaussian Mixture Models (NEW)

#-------------------------------------------Classification---------------------------------------------

#           k-Nearest Neighbors (KNN)
def KNN(X_train, X_test, y_train, y_test):
    #an empty list to store outputs of trying different k's on data
    KNN_score = []

    # Looping over different values of K
    for i in range(2, 10):
        KNN = KNeighborsClassifier(n_neighbors=i)
        KNN.fit(X_train, y_train)
        KNN_score.append(KNN.score(X_train, y_train))

    highest_KNN = KNN_score.index(max(KNN_score)) + 2

    KNN = KNeighborsClassifier(n_neighbors=highest_KNN)
    #y_pred = KNN.predict(X_test)
    KNN.fit(X_train, y_train)
    test_score = KNN.score(X_test, y_test)

    plt.figure(figsize=(10, 6))
    x = [2,3,4,5,6,7,8,9]

    #print(KNN.classes_)
    #plotting different values of K and the resulted accuracy
    plt.plot(x, KNN_score, color='blue', linestyle='dashed', marker='o',
             markerfacecolor='red', markersize=10)
    plt.title(f'Score vs. number of neighbours for breast cancer dataset')
    plt.xlabel('K')
    plt.ylabel('Score')
    print(f"Maximum score for breast cancer dataset : ",  KNN_score[highest_KNN], "at K =", highest_KNN)
    print(f"Training accuracy for breast cancer dataset: ", test_score * 100, '%')
    plt.show()

    y_pred = KNN.predict(X_test)

    # Convert class labels from integer to string format
    y_true_str = ["Benign" if label == 0 else "Malignant" for label in y_test]
    y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_pred]

    # Evaluate the performance of the classifier
    print("Accuracy:", accuracy_score(y_true_str, y_pred_str))

    # Generate a confusion matrix plot
    cm = confusion_matrix(y_true_str, y_pred_str, labels=["Benign", "Malignant"])
    cm_labels = {"Benign": "Benign", "Malignant": "Malignant"}
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels.values(), yticklabels=cm_labels.values(),
                cbar=False, annot_kws={"fontsize": 12}, linewidths=.5, linecolor='lightgray')
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.ax.tick_params(labelsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.tight_layout()
    plt.show()

    #SCATTER PLOT
    pca = PCA(n_components=1)

    X_train_plt = pca.fit_transform(X_train)
    X_test_plt = pca.fit_transform(X_test)
    knn_plt = KNeighborsClassifier(n_neighbors=2).fit(X_train_plt,y_train)
    y_pred_plot = knn_plt.predict(X_test_plt)

    # Generate a scatter plot graph for
    plt.scatter(X_test_plt[y_pred_plot == 0], y_pred_plot[y_pred_plot == 0], s=3, c='r')
    plt.scatter(X_test_plt[y_pred_plot == 1], y_pred_plot[y_pred_plot == 1], s=3, c='b')
    plt.show()



#           Support Vector Machines (SVM)

#           Naive Bayes

#           Decision Trees


def decision_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(random_state=0)

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Test the classifier on the test data
    y_pred = clf.predict(X_test)

    # Evaluate the performance of the classifier
    print("Accuracy:", accuracy_score(y_test, y_pred))

#           Linear Regression (not the best for our data so we need to point that)
def LinearReg(X_train, X_test, y_train, y_test):
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    #print(X_train.shape)
    intr = regr.intercept_
    print(f'inspect the intercept : {intr}')
    slope = regr.coef_
    print(f'retrieving the slope : {slope}')
    print(f'regression score : {regr.score(X_test, y_test)}')

    y_pred = regr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.2f}')
    print(f'Mean squared error: {mse:.2f}')
    print(f'Root mean squared error: {rmse:.2f}')

    # Convert class labels from integer to string format
    y_true_str = ["Benign" if label == 0 else "Malignant" for label in y_test]
    y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_pred]

    # Evaluate the performance of the classifier
    print("Accuracy:", accuracy_score(y_true_str, y_pred_str)*100)

    # Generate a confusion matrix plot
    cm = confusion_matrix(y_true_str, y_pred_str, labels=["Benign", "Malignant"])
    cm_labels = {"Benign": "Benign", "Malignant": "Malignant"}
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels.values(), yticklabels=cm_labels.values(),
                cbar=False, annot_kws={"fontsize": 12}, linewidths=.5, linecolor='lightgray')
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.ax.tick_params(labelsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Generate a feature importance plot (for models with coefficients available)
    if hasattr(regr, 'coef_'):
        coefs = np.abs(regr.coef_.ravel())
        names = range(1, len(coefs) + 1)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(names, coefs, height=0.7, color=plt.cm.RdBu(np.sign(coefs)))
        ax.set_yticks(names)
        ax.set_xlabel('Coefficient', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Feature Importance', fontsize=14)
        plt.show()

#           Random Forests (NEW)
def RandomForest(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


#           Logistic Regression (NEW)

def logistic_regression(X_train, X_test, y_train, y_test):

    # Create a logistic regression classifier
    clf = LogisticRegression(random_state=0)

    # Train the classifier on the training data
    clf.fit(X_train, y_train)

    # Test the classifier on the test data
    y_pred = clf.predict(X_test)

    # Convert class labels from integer to string format
    y_true_str = ["Benign" if label == 0 else "Malignant" for label in y_test]
    y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_pred]

    # Evaluate the performance of the classifier
    print("Accuracy:", accuracy_score(y_true_str, y_pred_str))

    # Generate a confusion matrix plot
    cm = confusion_matrix(y_true_str, y_pred_str, labels=["Benign", "Malignant"])
    cm_labels = {"Benign": "Benign", "Malignant": "Malignant"}
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels.values(), yticklabels=cm_labels.values(),
                cbar=False, annot_kws={"fontsize": 12}, linewidths=.5, linecolor='lightgray')
    cbar = ax.figure.colorbar(ax.collections[0])
    cbar.ax.tick_params(labelsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted label', fontsize=12)
    plt.ylabel('True label', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Generate an ROC curve plot
    y_prob = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    # Generate a feature importance plot (for models with coefficients available)
    if hasattr(clf, 'coef_'):
        coefs = np.abs(clf.coef_.ravel())
        names = range(1, len(coefs) + 1)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(names, coefs, height=0.7, color=plt.cm.RdBu(np.sign(coefs)))
        ax.set_yticks(names)
        ax.set_xlabel('Coefficient', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title('Feature Importance', fontsize=14)
        plt.show()

#-------------------------------------------Visualization----------------------------------------------

#           Scatterplots

#           Dendrogram

#           Confusion Matrix (NEW)

#           Receiver Operating Characteristic (ROC) curve (NEW)

#           Heatmap (NEW)

#           Silhouette Plot (NEW)





