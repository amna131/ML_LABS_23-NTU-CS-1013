# Name: Amna
# Reg no: 23-NTU-CS-1013
# Activity 6:  Dataset Exploration & Preprocessing
# Explore Breast Cancer dataset: handle missing values and imbalance, detect outliers,
# apply log transform, scale features, reduce with PCA, and compare classifiers

# Subtask no.1 ---------- Load and Explore Dataset ----------
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = load_breast_cancer()

print(data.keys())
print(data.target_names)

n_samples, n_features = data.data.shape

print("Number of samples:", n_samples)
print("Number of features:", n_features)
print("Feature Name:", data.feature_names)
print("Dimension of Input:", data.data.shape)
print("Dimension of Output:", data.target.shape)
print()


# Subtask no.2 ---------- Handle Missing Values ----------
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Check missing values
print("Missing values before:", df.isnull().sum().sum())

# Add 5 random missing values in mean radius
np.random.seed(42)
random_indices = np.random.choice(df.index, 5, replace=False)
df.loc[random_indices, "mean radius"] = np.nan

print("Missing values after adding NaNs:", df["mean radius"].isnull().sum())
print()
df["mean radius"] = df["mean radius"].fillna(df["mean radius"].median())

print("Median was used instead of mean because 'mean radius' has some skew,")
print("So mean could be affected by unusually large values. Median is more reliable here.")
print()
print("Missing values after filling:", df["mean radius"].isnull().sum())
print()


#Subtask no.3 ---------- Check Class Distribution and Apply SMOTE ----------
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X = df.drop('target', axis=1)
y = df['target']

print("Class distribution (0 = malignant, 1 = benign):")
print(y.value_counts())
print("The dataset is mildly imbalanced (357 benign vs 212 malignant).")

# Split into train/test first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Oversampling (SMOTE)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE (oversampling):\n", y_train_smote.value_counts())
print()


#Subtask no.4 ---------- Detect Outliers using IQR Method ----------
features = ["mean radius", "mean texture"]

for feature in features:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[(df[feature] < lower_limit) | (df[feature] > upper_limit)]

    print(feature, "(number of outliers):", len(outliers))

# Boxplot for both features
df[["mean radius", "mean texture"]].boxplot()
plt.title("Boxplot of Mean Radius and Mean Texture")
plt.ylabel("Values")
plt.show()
print()


# Subtask no.5 ---------- Apply log transformation to mean area ----------
df["log mean area"] = np.log(df["mean area"])

# Compare distribution before and after
plt.hist(df["mean area"], bins=20)
plt.title("Mean Area Before Log Transformation")
plt.xlabel("Mean Area")
plt.ylabel("Frequency")
plt.show()

plt.hist(df["log mean area"], bins=20)
plt.title("Mean Area After Log Transformation")
plt.xlabel("Log Mean Area")
plt.ylabel("Frequency")
plt.show()
print("Log Transformation to mean area")
print("Before: right-skewed. After: more symmetric.")
print()


#Subtask no.6 ---------- Apply Feature Scaling ----------
from sklearn.preprocessing import StandardScaler

# Scale all numeric features
numeric_features = df.drop(['target', 'log mean area'], axis=1)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(numeric_features)

print("Scaling makes all features the same range, so no single feature dominates in KNN or SVM.")
print()


#Subtask no.7 ---------- Reduce Dimensions with PCA ----------
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['target'])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Breast Cancer Dataset")
plt.colorbar(label="Target (0 = malignant, 1 = benign)")
plt.show()
print("The two classes show fairly good separation, with some overlap in the middle.")
print()


#Subtask no.8 ---------- Train and Compare Classifiers ----------
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Split scaled data into train and test sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    scaled_features, df['target'], test_size=0.2, random_state=42, stratify=df['target']
)

# Train SVM
svm_model = SVC()
svm_model.fit(X_train2, y_train2)
svm_pred = svm_model.predict(X_test2)
svm_accuracy = accuracy_score(y_test2, svm_pred)

# Train KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train2, y_train2)
knn_pred = knn_model.predict(X_test2)
knn_accuracy = accuracy_score(y_test2, knn_pred)

# Compare accuracy
print("SVM Accuracy:", svm_accuracy)
print("KNN Accuracy:", knn_accuracy)
print()

#Subtask no.9 ---------- Conclusion ----------
print("Scaling had the biggest impact on this dataset. Features like mean area")
print("have much bigger numbers than features like mean smoothness. Without scaling, models")
print("like SVM and KNN would rely too much on the bigger numbers and ignore the smaller ones.")