from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, \
    mean_absolute_error, mean_squared_error

from preprocessing import *
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

show_graphs = True
# Read in the CSV file
df = pd.read_csv('breast-cancer.csv')

# Get the column names (labels)
labels = list(df.columns)
labels.pop(0)
labels.pop(0)

#---------------------------------------------Clustering----------------------------------------------
#          k mediod
def kmediods(X_train, X_test, y_train, y_test):
    print("#---------------------------------------------k mediod----------------------------------------------")
    kmedoids = KMedoids(n_clusters=2).fit(X_train)
    # Get labels for each point in training data
    y_predict = np.array(kmedoids.labels_)
    # View final mediods
    print(kmedoids.cluster_centers_)
    # Flip the 0s to 1s and vice versa
    y_predict_flipped = np.where((y_predict == 0) | (y_predict == 1), y_predict ^ 1, y_predict)
    # Test the model on the test data
    y_predict_test = KMedoids.predict(kmedoids, X_test)
    # Flip the 0s to 1s and vice versa
    y_predict_test_flipped = np.where((y_predict_test == 0) | (y_predict_test == 1), y_predict_test ^ 1, y_predict_test)

    # Evaluate the performance of the model on both training and testing data
    train_acc = max(accuracy_score(y_train, y_predict), accuracy_score(y_train, y_predict_flipped))
    test_acc = max(accuracy_score(y_test, y_predict_test), accuracy_score(y_test, y_predict_test_flipped))

    print("Training data labels: ")
    print(kmedoids.labels_)
    print("---------------------------------------------------------------------")
    print("Cluster Centers")
    print(kmedoids.cluster_centers_)
    print("---------------------------------------------------------------------")
    print("Training Accuracy: ", "{:.4f}".format(train_acc * 100), "%")
    print("Testing Accuracy: ", "{:.4f}".format(test_acc * 100), "%")

    # Reduce training data to 2 features using pca to plot results
    pca = PCA(n_components=2)

    X_train_plt = pca.fit_transform(X_train)

    kmediods_plt = KMedoids(n_clusters=2).fit(X_train_plt)

    # Generate a scatter plot graph for 2 clusters and their centers
    plt.scatter(X_train_plt[:, 0][kmediods_plt.labels_ == 0], X_train_plt[:, 1][kmediods_plt.labels_ == 0], s=3, c='r')
    plt.scatter(X_train_plt[:, 0][kmediods_plt.labels_ == 1], X_train_plt[:, 1][kmediods_plt.labels_ == 1], s=3, c='b')
    plt.plot(kmediods_plt .cluster_centers_[0][0], kmediods_plt.cluster_centers_[0][1], marker="x", markersize=10,
             markeredgecolor="black")
    plt.plot(kmediods_plt .cluster_centers_[1][0], kmediods_plt.cluster_centers_[1][1], marker="x", markersize=10,
             markeredgecolor="black")

    plt.show()

    # Convert class labels from integer to string format
    y_true_str = ["Benign" if label == 0 else "Malignant" for label in y_test]
    if accuracy_score(y_test, y_predict_test) > accuracy_score(y_test, y_predict_test_flipped):
        y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_predict_test]
    else:
        y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_predict_test_flipped]

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


#           kmeans
def kmeans(X_train, X_test, y_train, y_test, n_clusters=2):
    print("#---------------------------------------------kmeans----------------------------------------------")

    # Create a kmeans clustering model and fit on the training data
    kmeans = KMeans(n_clusters=n_clusters).fit(X_train)

    # Get labels for each point in training data
    y_predict = np.array(kmeans.labels_)

    # Flip the 0s to 1s and vice versa
    y_predict_flipped = np.where((y_predict==0)|(y_predict==1), y_predict^1, y_predict)

    # Test the model on the test data
    y_predict_test = KMeans.predict(kmeans, X_test)

    # Flip the 0s to 1s and vice versa
    y_predict_test_flipped = np.where((y_predict_test==0)|(y_predict_test==1), y_predict_test^1, y_predict_test)

    # Evaluate the performance of the model on both training and testing data
    train_acc = max(accuracy_score(y_train, y_predict), accuracy_score(y_train, y_predict_flipped))
    test_acc = max(accuracy_score(y_test, y_predict_test),accuracy_score(y_test, y_predict_test_flipped))

    print("Training data labels: ")
    print(kmeans.labels_)
    print("---------------------------------------------------------------------")
    print("Cluster Centers")
    print(kmeans.cluster_centers_)
    print("---------------------------------------------------------------------")
    print("Training Accuracy: ", "{:.4f}".format(train_acc*100), "%")
    print("Testing Accuracy: ", "{:.4f}".format(test_acc*100), "%")

    # Reduce training data to 2 features using pca to plot results
    pca = PCA(n_components=2)

    X_train_plt = pca.fit_transform(X_train)

    kmeans_plt = KMeans(n_clusters=n_clusters).fit(X_train_plt)

    # Generate a scatter plot graph for 2 clusters and their centers
    if(show_graphs):
        plt.scatter(X_train_plt[:, 0][kmeans_plt.labels_ == 0], X_train_plt[:, 1][kmeans_plt.labels_ == 0], s=3, c='r')
        plt.scatter(X_train_plt[:, 0][kmeans_plt.labels_ == 1], X_train_plt[:, 1][kmeans_plt.labels_ == 1], s=3, c='b')
        plt.plot(kmeans_plt.cluster_centers_[0][0], kmeans_plt.cluster_centers_[0][1], marker="x", markersize=10, markeredgecolor="black")
        plt.plot(kmeans_plt.cluster_centers_[1][0], kmeans_plt.cluster_centers_[1][1], marker="x", markersize=10, markeredgecolor="black")

        plt.show()

        # Convert class labels from integer to string format
        y_true_str = ["Benign" if label == 0 else "Malignant" for label in y_test]
        if accuracy_score(y_test, y_predict_test) > accuracy_score(y_test, y_predict_test_flipped):
            y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_predict_test]
        else:
            y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_predict_test_flipped]

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

#           DBscan
def DBscan(X_train, X_test, y_train, y_test):
    print("#---------------------------------------------DBscan----------------------------------------------")

    acc = []
    for i in range(1, 10):
        # Create a dbscan clustering model and fit on the training data with different maximum distance
        dbscan = DBSCAN(eps=(i/2), min_samples=3).fit(X_train)

        y_predict = np.array(dbscan.labels_)
        y_predict_flipped = np.where((y_predict == 0) | (y_predict == 1), y_predict ^ 1, y_predict)
        train_acc = max(accuracy_score(y_train, y_predict), accuracy_score(y_train, y_predict_flipped))

        acc.append(train_acc)

    ep_highest = (acc.index(max(acc)) + 1)/2

    # Create a dbscan clustering model and fit on the training data with different maximum distance
    highest_dbscan = DBSCAN(eps=ep_highest+0.5, min_samples=3).fit(X_train)  # highest ep shows overfitting so we'll take second highest value which mostly solves this issue

    # Get labels for each point in training data
    y_predict = highest_dbscan.labels_

    # Flip the 0s to 1s and vice versa
    y_predict_flipped = np.where((y_predict == 0) | (y_predict == 1), y_predict ^ 1, y_predict)

    # Evaluate the performance of the model on training data
    train_acc = max(accuracy_score(y_train, y_predict), accuracy_score(y_train, y_predict_flipped))

    print("Training data labels: ")
    print(highest_dbscan.labels_)
    print("---------------------------------------------------------------------")

    # Test the model on the test data
    y_predict_test = DBSCAN.fit_predict(highest_dbscan, X_test)

    # Flip the 0s to 1s and vice versa
    y_predict_test_flipped = np.where((y_predict_test == 0) | (y_predict_test == 1), y_predict_test ^ 1, y_predict_test)

    # Evaluate the performance of the model on testing data
    test_acc = max(accuracy_score(y_test, y_predict_test), accuracy_score(y_test, y_predict_test_flipped))

    print("Test data labels: ")
    print(highest_dbscan.labels_)
    print("---------------------------------------------------------------------")
    print("Training Accuracy: ", "{:.4f}".format(train_acc*100), "%")
    print("Testing Accuracy: ", "{:.4f}".format(test_acc * 100), "%")

    # Convert class labels from integer to string format
    y_true_str = ["Benign" if label == 0 else "Malignant" for label in y_test]
    if accuracy_score(y_test, y_predict_test) > accuracy_score(y_test, y_predict_test_flipped):
        y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_predict_test]
    else:
        y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_predict_test_flipped]

    # Generate a confusion matrix plot
    if (show_graphs):
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

#           Hierarchical clustering
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def agglomerative_clustering(X_train, X_test, y_train, y_test):
    print("#---------------------------------------------agglomerative_clustering----------------------------------------------")

    # Create agglomerative clustering model and fit on the training data
    Agglomerative = AgglomerativeClustering().fit(X_train)

    # Create agglomerative clustering model and fit on the training data to plot dendrogram
    Agglomerative_plt = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(X_train)

    # Get labels for each point in training data
    y_predict = Agglomerative.labels_

    # Flip the 0s to 1s and vice versa
    y_predict_flipped = np.where((y_predict == 0) | (y_predict == 1), y_predict ^ 1, y_predict)

    # Evaluate the performance of the model on training data
    train_acc = max(accuracy_score(y_train, y_predict), accuracy_score(y_train, y_predict_flipped))

    print("Training data labels: ")
    print(Agglomerative.labels_)
    print("---------------------------------------------------------------------")

    # Test the model on the test data
    y_predict_test = AgglomerativeClustering().fit_predict(X_test)

    # Flip the 0s to 1s and vice versa
    y_predict_test_flipped = np.where((y_predict_test == 0) | (y_predict_test == 1), y_predict_test ^ 1, y_predict_test)

    # Evaluate the performance of the model on testing data
    test_acc = max(accuracy_score(y_test, y_predict_test), accuracy_score(y_test, y_predict_test_flipped))

    print("Test data labels: ")
    print(y_predict_test)
    print("---------------------------------------------------------------------")
    print("Training Accuracy: ", "{:.4f}".format(train_acc*100), "%")
    print("Testing Accuracy: ", "{:.4f}".format(test_acc*100), "%")

    if (show_graphs):
        plt.title("Hierarchical Clustering Dendrogram")
        # plot the top three levels of the dendrogram
        plot_dendrogram(Agglomerative_plt, truncate_mode="level", p=3)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.show()

        # Convert class labels from integer to string format
        y_true_str = ["Benign" if label == 0 else "Malignant" for label in y_test]
        if accuracy_score(y_test, y_predict_test) > accuracy_score(y_test, y_predict_test_flipped):
            y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_predict_test]
        else:
            y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_predict_test_flipped]

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


#-------------------------------------------Classification---------------------------------------------

#           k-Nearest Neighbors (KNN)
def KNN(X_train, X_test, y_train, y_test):
    print("#---------------------------------------------KNN----------------------------------------------")

    #an empty list to store outputs of trying different k's on data
    KNN_score = []

    # Looping over different values of K
    for i in range(2, 11):
        KNN = KNeighborsClassifier(n_neighbors=i)
        KNN.fit(X_train, y_train)
        KNN_score.append(KNN.score(X_train, y_train))

    highest_KNN = KNN_score.index(max(KNN_score)) + 2

    KNN = KNeighborsClassifier(n_neighbors=highest_KNN)
    #y_pred = KNN.predict(X_test)
    KNN.fit(X_train, y_train)
    test_score = KNN.score(X_test, y_test)

    y_pred = KNN.predict(X_test)
    print(f"Maximum score for breast cancer dataset : ", KNN_score[highest_KNN], "at K =", highest_KNN)
    print(f"Training accuracy for breast cancer dataset: ", test_score * 100, '%')

    # Convert class labels from integer to string format
    y_true_str = ["Benign" if label == 0 else "Malignant" for label in y_test]
    y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_pred]

    # Evaluate the performance of the classifier
    print("Accuracy:", accuracy_score(y_true_str, y_pred_str))

    if (show_graphs):
        plt.figure(figsize=(10, 6))
        x = [2,3,4,5,6,7,8,9]

        #print(KNN.classes_)
        #plotting different values of K and the resulted accuracy
        plt.plot(x, KNN_score, color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)
        plt.title(f'Score vs. number of neighbours for breast cancer dataset')
        plt.xlabel('K')
        plt.ylabel('Score')

        plt.show()



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

#           Decision Trees
def decision_tree(X_train, X_test, y_train, y_test):
    print("#---------------------------------------------Decision_tree----------------------------------------------")
    model = DecisionTreeClassifier(random_state=0)

    # Train the classifier on the training data
    model.fit(X_train, y_train)

    # Test the classifier on the test data
    y_pred = model.predict(X_test)

    # Evaluate the performance of the classifier
    print("Accuracy:", accuracy_score(y_test, y_pred))

    if (show_graphs):
        # Generate a confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        cm_labels = {"Benign": "Benign", "Malignant": "Malignant"}
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels.values(), yticklabels=cm_labels.values(),
                    cbar=False, annot_kws={"fontsize": 10}, linewidths=.5, linecolor='lightgray')
        cbar = ax1.figure.colorbar(ax1.collections[0])
        cbar.ax.tick_params(labelsize=10)
        plt.title('Confusion Matrix', fontsize=12)
        plt.xlabel('Predicted label', fontsize=10)
        plt.ylabel('True label', fontsize=10)

        # Generate a tree diagram plot
        fig2, ax2 = plt.subplots(figsize=(16, 8))
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns
        else:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        plot_tree(model, filled=True, feature_names=feature_names, class_names=["Benign", "Malignant"], fontsize=6,
                  ax=ax2)
        plt.title('Decision Tree', fontsize=12)
        plt.show()

#           Linear Regression (not the best for our data so we need to point that)
def LinearReg(X_train, X_test, y_train, y_test):
    print("#---------------------------------------------Linear_regression----------------------------------------------")

    regr = LinearRegression()
    regr.fit(X_train, y_train)
    #print(X_train.shape)
    intr = regr.intercept_
    print(f'inspect the intercept : {intr}')
    slope = regr.coef_
    print(f'retrieving the slope : {slope}')
    print(f'regression score : {regr.score(X_test, y_test)}')

    y_pred = regr.predict(X_test)

    y_pred = np.where(y_pred > 0.5, 1, 0)

    y_pred_flipped = np.where((y_pred == 0) | (y_pred == 1), y_pred ^ 1, y_pred)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Mean absolute error: {mae:.4f}')
    print(f'Mean squared error: {mse:.4f}')
    print(f'Root mean squared error: {rmse:.4f}')

    # Convert class labels from integer to string format
    y_true_str = ["Benign" if label == 0 else "Malignant" for label in y_test]
    if accuracy_score(y_test, y_pred) > accuracy_score(y_test, y_pred_flipped):
        y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_pred]
    else:
        y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_pred_flipped]

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

    # Generate an ROC curve plot

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Generate a feature importance plot (for models with coefficients available)
    if hasattr(regr, 'coef_'):
        coefs = regr.coef_.ravel()
        if len(coefs) == 5:
            names = ['symmetry_se', 'smoothness_mean', 'texture_se', 'symmetry_worst', 'compactness_se']
        else:
            names = labels
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ax3.barh(names, coefs, height=0.7, color=plt.cm.RdBu(np.sign(coefs)))
        ax3.set_yticks(names)
        ax3.set_xlabel('Coefficient', fontsize=12)
        ax3.set_ylabel('Feature', fontsize=12)
        ax3.set_title('Feature Importance', fontsize=14)

    # Show all plots
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels.values(), yticklabels=cm_labels.values(),
                cbar=False, annot_kws={"fontsize": 12}, linewidths=.5, linecolor='lightgray', ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=14)
    ax1.set_xlabel('Predicted label', fontsize=12)
    ax1.set_ylabel('True label', fontsize=12)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax2.legend(loc="lower right")
    plt.show()



#           Random Forests (NEW)
def RandomForest(X_train, X_test, y_train, y_test):
    print("#---------------------------------------------Random_forest----------------------------------------------")

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)


#           Support Vector Machines (SVM)
def support_vector_machines(X_train, X_test, y_train, y_test):
    print("#---------------------------------------------SVM----------------------------------------------")

    # Create a support vector machine classifier
    model = svm.SVC(kernel='linear', probability=True, random_state=0)

    # Train the classifier on the training data
    model.fit(X_train, y_train)

    # Test the classifier on the test data
    y_pred = model.predict(X_test)

    # Convert class labels from integer to string format
    y_true_str = ["Benign" if label == 0 else "Malignant" for label in y_test]
    y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_pred]

    # Evaluate the performance of the classifier
    print("Accuracy:", accuracy_score(y_true_str, y_pred_str))

    if (show_graphs):
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

        # Generate an ROC curve plot
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        # Generate a feature importance plot (for models with coefficients available)
        if hasattr(model, 'coef_'):
            coefs = model.coef_.ravel()
            if len(coefs) == 5:
                names = ['symmetry_se', 'smoothness_mean', 'texture_se', 'symmetry_worst', 'compactness_se']
            else:
                names = labels
            fig3, ax3 = plt.subplots(figsize=(7, 5))
            ax3.barh(names, coefs, height=0.7, color=plt.cm.RdBu(np.sign(coefs)))
            ax3.set_yticks(names)
            ax3.set_xlabel('Coefficient', fontsize=12)
            ax3.set_ylabel('Feature', fontsize=12)
            ax3.set_title('Feature Importance', fontsize=14)
            fig3, ax3 = plt.subplots(figsize=(7, 5))
            ax3.barh(names, coefs, height=0.7, color=plt.cm.RdBu(np.sign(coefs)))
            ax3.set_yticks(names)
            ax3.set_xlabel('Coefficient', fontsize=12)
            ax3.set_ylabel('Feature', fontsize=12)
            ax3.set_title('Feature Importance', fontsize=14)

        # Show all plots
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels.values(), yticklabels=cm_labels.values(),
                    cbar=False, annot_kws={"fontsize": 12}, linewidths=.5, linecolor='lightgray', ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=14)
        ax1.set_xlabel('Predicted label', fontsize=12)
        ax1.set_ylabel('True label', fontsize=12)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax2.legend(loc="lower right")
        plt.show()

#           Naive Bayes
def naive_bayes(X_train, X_test, y_train, y_test):
    print("#---------------------------------------------Naive_Bayes----------------------------------------------")

    # Create a Gaussian Naive Bayes classifier
    model = GaussianNB()

    # Train the classifier on the training data
    model.fit(X_train, y_train)

    # Test the classifier on the test data
    y_pred = model.predict(X_test)

    # Convert class labels from integer to string format
    y_true_str = ["Benign" if label == 0 else "Malignant" for label in y_test]
    y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_pred]

    # Evaluate the performance of the classifier
    print("Accuracy:", accuracy_score(y_true_str, y_pred_str))

    if (show_graphs):
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
        plt.show()

        # Generate an ROC curve plot
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='darkorange')
        ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.show()

        # Generate a Class Distribution Plot
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x=y_test, data=pd.DataFrame({'class': y_test}))
        plt.xlabel('Class Label', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title('Class Distribution', fontsize=14)
        plt.show()

        # Generate a Feature Importance Plot
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            feature_names = X_train.columns.values
            features_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
            features_df = features_df.sort_values(by='Importance', ascending=False)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(x='Importance', y='Feature', data=features_df)
            plt.xlabel('Feature Importance', fontsize=12)
            plt.ylabel('Feature Name', fontsize=12)
            plt.title('Feature Importance', fontsize=14)
            plt.show()

        # Compute the calibration curve
        # Predict probabilities for the test data
        y_prob = model.predict_proba(X_test)[:, 1]
        empirical_probs, predicted_probs = calibration_curve(y_test, y_prob, n_bins=10)

        # Plot the calibration curve
        plt.plot(predicted_probs, empirical_probs, 's-', label='%s' % 'Gaussian Naive Bayes')

        # Add diagonal line representing perfect calibration
        plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly calibrated')

        # Customize the plot
        plt.xlabel('Predicted Probability')
        plt.ylabel('Empirical Probability')
        plt.title('Calibration Curve')
        plt.legend(loc='lower right')
        plt.show()

#           Logistic Regression (NEW)
def logistic_regression(X_train, X_test, y_train, y_test):
    print("#---------------------------------------------Logistic_regression----------------------------------------------")

    # Create a logistic regression classifier
    model = LogisticRegression(random_state=0,max_iter=10000)

    # Train the classifier on the training data
    model.fit(X_train, y_train)

    # Test the classifier on the test data
    y_pred = model.predict(X_test)

    # Convert class labels from integer to string format
    y_true_str = ["Benign" if label == 0 else "Malignant" for label in y_test]
    y_pred_str = ["Benign" if label == 0 else "Malignant" for label in y_pred]

    # Evaluate the performance of the classifier
    print("Accuracy:", accuracy_score(y_true_str, y_pred_str))

    if (show_graphs):
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

        # Generate an ROC curve plot
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)

        # Generate a feature importance plot (for models with coefficients available)
        if hasattr(model, 'coef_'):
            coefs = model.coef_.ravel()
            if len(coefs) == 5:
                names = ['symmetry_se', 'smoothness_mean', 'texture_se', 'symmetry_worst', 'compactness_se']
            else:
                names = labels
            fig3, ax3 = plt.subplots(figsize=(7, 5))
            ax3.barh(names, coefs, height=0.7, color=plt.cm.RdBu(np.sign(coefs)))
            ax3.set_yticks(names)
            ax3.set_xlabel('Coefficient', fontsize=12)
            ax3.set_ylabel('Feature', fontsize=12)
            ax3.set_title('Feature Importance', fontsize=14)

        # Show all plots
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=cm_labels.values(), yticklabels=cm_labels.values(),
                    cbar=False, annot_kws={"fontsize": 12}, linewidths=.5, linecolor='lightgray', ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=14)
        ax1.set_xlabel('Predicted label', fontsize=12)
        ax1.set_ylabel('True label', fontsize=12)
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax2.legend(loc="lower right")
        plt.show()


