from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn import svm
from Project import *
import numpy as np
#---------------------------------------------Clustering----------------------------------------------

#           kmeans - kmediod

#           DBscan

#           Hierarchical clustering

#           Gaussian Mixture Models (NEW)

#-------------------------------------------Classification---------------------------------------------

#           Support Vector Machines (SVM)
def support_vector_machines(features,target):
    #preprocessing
    X_train, X_test, y_train, y_test = preprocess()

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
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Generate a feature importance plot (for models with coefficients available)
    if hasattr(model, 'coef_'):
        coefs = np.abs(model.coef_.ravel())
        names = range(1, len(coefs) + 1)
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
def naive_bayes(features,target):
    #preprocessing
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

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
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='darkorange')
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.show()

    # Generate a Precision-Recall Curve plot
    precision, recall, _ = precision_recall_curve(y_test, y_score, pos_label=1)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color='darkorange')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
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



#           Decision Trees

def decision_tree(features, target):
    #preprocessing
    X_train, X_test, y_train, y_test = preprocess()

    model = DecisionTreeClassifier(random_state=0)

    # Train the classifier on the training data
    model.fit(X_train, y_train)

    # Test the classifier on the test data
    y_pred = model.predict(X_test)

    # Evaluate the performance of the classifier
    print("Accuracy:", accuracy_score(y_test, y_pred))

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

#           Random Forests (NEW)

#           Logistic Regression (NEW)

def logistic_regression(features, target):
    #preprocess
    X_train, X_test, y_train, y_test = preprocess()

    # Create a logistic regression classifier
    model = LogisticRegression(random_state=0)

    # Train the classifier on the training data
    model.fit(X_train, y_train)

    # Test the classifier on the test data
    y_pred = model.predict(X_test)

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

    # Generate an ROC curve plot
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Generate a feature importance plot (for models with coefficients available)
    if hasattr(model, 'coef_'):
        coefs = np.abs(model.coef_.ravel())
        names = range(1, len(coefs) + 1)
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







