# Name: Amna
# Reg no: 23-NTU-CS-1013
# Lab no: 5
# Activity 2: SVM for Brain Tumor Detection

# Detect brain tumors from MRI images using an SVM classifier:
# preprocess the images (grayscale, resize to 64x64, flatten), split the data 80:20,
# scale the features, train an SVM (RBF kernel, C=10, gamma='scale'), evaluate using
# accuracy, classification report and confusion matrix,
# display test images with their predicted and actual labels

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#1. ---Load and preprocess all MRI images---
dataset_path = "brain_tumor_dataset"
categories = ['no', 'yes']

images = []
labels = []

for label, category in enumerate(categories):
    folder = os.path.join(dataset_path, category)
    for file_name in os.listdir(folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = Image.open(os.path.join(folder, file_name))

            # Convert to grayscale
            img = img.convert('L')

            # Resize to 64x64
            img = img.resize((64, 64))

            images.append(np.array(img))
            labels.append(label)

X = np.array(images)
y = np.array(labels)
print("Images shape:", X.shape)
print("No Tumor images:", np.sum(y == 0))
print("Tumor images:", np.sum(y == 1))
print()

# Flatten each image (64 x 64 = 4096 features)
X = X.reshape(len(X), -1)
print("Flattened shape:", X.shape)
print()

#2. ---Split the data into training and testing sets (80:20)---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Keep the test images in 64x64 form for displaying later
test_images = X_test.reshape(-1, 64, 64)

#3. ---Normalize features using StandardScaler---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#4. ---Train SVM (RBF kernel, C=10, gamma='scale')---
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

#5. ---Evaluate the model---
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Tumor', 'Tumor']))

# Confusion matrix
plt.figure(figsize=(8, 6))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Tumor', 'Tumor'],
            yticklabels=['No Tumor', 'Tumor'])

plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

#6. ---Display test images with predicted and actual labels---
plt.figure(figsize=(12, 6))

for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(test_images[i], cmap='gray')
    plt.title(f"Predicted: {categories[y_pred[i]]}\nActual: {categories[y_test[i]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()