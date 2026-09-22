# Name: Amna
# Reg no: 23-NTU-CS-1013
# Task 21
# Fetch the Olivetti faces dataset, inspect it, and display sample images

from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt

faces = fetch_olivetti_faces()
faces.keys()

# Inspect dataset shape and description
n_samples, n_features = faces.data.shape
print((n_samples, n_features))
print(faces.images.shape)
print(faces.data.shape)
print(faces.DESCR)

# Extract features and labels
X, y = faces.data, faces.target

# Show one image (5th index)
plt.imshow(faces.images[5], cmap="gray")
plt.title(f"Face ID: {faces.target[5]}")
plt.axis("off")
plt.show()

# Show first 16 images
plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(faces.images[i], cmap="gray")
    plt.axis("off")
    plt.title(faces.target[i])
plt.show()