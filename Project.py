# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import KFold

matplotlib.use('TkAgg')


def remove_outliers_Zscore(features_scaled, target):
    z_scores = np.abs(features_scaled)
    threshold = 3
    features_clean = features_scaled[(z_scores < threshold).all(axis=1)]
    target_clean = target[(z_scores < threshold).all(axis=1)]
    return features_clean, target_clean


def Box_plot_outliers(features_scaled, target, headers):
    df = pd.DataFrame(features_scaled)
    seq = pd.Series(target, index=df.index, name='diagnosis')
    df = pd.concat([seq, df], axis=1)
    df = df.set_axis(headers[1:], axis=1, copy=False)

    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.boxplot(features_scaled, labels=headers[2:])
    #plt.setp(ax.get_xticklabels(), rotation=90)
    # plt.show()

    new_df = df
    for att in headers[1:]:
        Q1 = new_df.sort_values(by=att, ascending=True)[att].quantile(0.25)
        Q3 = new_df.sort_values(by=att, ascending=True)[att].quantile(0.75)
        IQR = Q3 - Q1
        new_df = new_df[(new_df[att] > Q1 - 1.5 * IQR) & (new_df[att] < Q3 + 1.5 * IQR)]

    return new_df.iloc[:, 1:], new_df['diagnosis']


def Kfolds(features, target, no_of_splits):
    kf = KFold(n_splits=no_of_splits)
    target = np.array(target).reshape(-1, 1)
    data = pd.DataFrame(features)
    data['target'] = target
    data_csv_file = {"train": [], "test": []}
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        train = data.iloc[train_index]
        test = data.iloc[test_index]

        data_csv_file['train'].append(f'train_fold_{i}.csv')
        data_csv_file['test'].append(f'test_fold_{i}.csv')
        train.to_csv(f'train_fold_{i}.csv', index=False)
        test.to_csv(f'test_fold_{i}.csv', index=False)

    return data_csv_file


def clean_data(df):
    nominal_cols = df.select_dtypes(exclude=['float', 'int']).columns
    numerical_values = df.select_dtypes(exclude='object').columns.tolist()
    label_encoder = preprocessing.LabelEncoder()

    # Encode nominal columns
    if len(nominal_cols) != 0:
        for col in nominal_cols:
            df[col] = label_encoder.fit_transform(df[col])

    # Checking and replacing any NULL values
    if df.isnull().values.any():
        dataframe_n = df.drop('diagnosis', axis=1)[numerical_values]
        num_imput = SimpleImputer(strategy='mean', missing_values=np.nan)
        dataframe_n = num_imput.fit_transform(dataframe_n)
        df[numerical_values] = dataframe_n

    return df


def preprocess():
    # Load the data
    df = pd.read_csv("breast-cancer.csv")

    # Cleaning our data
    df = clean_data(df)

    # Compute the correlation matrix for all columns except the first one
    corr_matrix = df.iloc[:, 1:].corr()

    # Print the correlation matrix
    # print(corr_matrix.sort_values('diagnosis', ascending=False))

    # Create a heatmap of the correlation matrix
    #plt.figure(figsize=(20, 15))
    #sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".1g")
    # plt.show()

    #plt.figure(figsize=(20, 15))
    #sns.heatmap(df.corr()[['diagnosis']].sort_values(by='diagnosis', ascending=False), linewidths=1, annot=True,
    #            cmap="coolwarm")
    # plt.show()

    # Split the dataset into features and target
    features = df.iloc[:, 2:]
    target = df.iloc[:, 1]

    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Remove outliers
    features_clean_z, target_clean_z = remove_outliers_Zscore(features_scaled, target)
    features_clean_box, target_clean_box = Box_plot_outliers(features_scaled, target, df.columns.values)

    x, y = pd.DataFrame(features_clean_box), pd.DataFrame(features_clean_z)
    y = y.set_axis(df.columns.values[2:], axis=1, copy=False)
    # print("Boxplot:", x.shape)
    # print("zscore:", y.shape)

    target_clean = target_clean_z[target_clean_z.index.isin(target_clean_box.index)]
    features_clean = x[x.radius_mean.isin(y.radius_mean)]

    # Split the dataset into training and testing sets

    files = Kfolds(features_clean, target_clean, 5)

    X_train, X_test, y_train, y_test = train_test_split(features_clean, target_clean, test_size=0.2, random_state=42)

    # Check the shapes of the resulting datasets
    # print("Training set shape:", X_train.shape, y_train.shape)
    # print("Testing set shape:", X_test.shape, y_test.shape)

    # Resample the training set
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    # Apply PCA to reduce dimensionality
    pca = PCA()
    pca.fit(X_train_resampled)

    # Select top principal components that explain at least 80% of the variance
    explained_variance_ratios = pca.explained_variance_ratio_
    cumulative_variance_ratios = np.cumsum(explained_variance_ratios)
    n_components = np.argmax(cumulative_variance_ratios >= 0.8) + 1

    # Print the selected feature names
    selected_features = []
    for i in range(n_components):
        component_index = np.argmax(pca.components_[i])
        selected_features.append(features.columns[component_index])

    # print("Selected features:", selected_features)

    # Transform training and testing sets with PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_resampled)
    X_test_pca = pca.transform(X_test)
    # print("Number of principal components selected: ", n_components)
    # print("Explained variance ratios:", pca.explained_variance_ratio_)

    return X_train_pca, X_test_pca, y_train_resampled, y_test
