# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 19:35:24 2024

@author: Shibo
"""

#1. Retrieve and Load the Olivetti Faces Dataset
from sklearn.datasets import fetch_olivetti_faces
import pandas as pd

# Load the dataset
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
data = faces.data
labels = faces.target

#2. Split the Dataset into Training, Validation, and Test Sets
from sklearn.model_selection import train_test_split

# Stratified split to ensure even distribution of images per person
X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

#3. Apply PCA on the Training Data
from sklearn.decomposition import PCA

pca = PCA(n_components=0.99, whiten=True)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

#4. Determine the Most Suitable Covariance Type for the Dataset
from sklearn.mixture import GaussianMixture
import numpy as np

cov_types = ['full', 'tied', 'diag', 'spherical']
lowest_bic = np.infty
best_gmm = None
best_cov_type = None
bic_scores = []

for cov_type in cov_types:
    gmm = GaussianMixture(n_components=40, covariance_type=cov_type, random_state=42)
    gmm.fit(X_train_pca)
    bic = gmm.bic(X_train_pca)
    bic_scores.append(bic)
    if bic < lowest_bic:
        lowest_bic = bic
        best_gmm = gmm
        best_cov_type = cov_type

#5. Determine the Minimum Number of Clusters Using AIC or BIC
n_components_range = range(1, 281)
lowest_bic = np.infty
best_gmm = None
bic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, covariance_type=best_cov_type, random_state=42)
    gmm.fit(X_train_pca)
    bic = gmm.bic(X_train_pca)
    bic_scores.append(bic)
    if bic < lowest_bic:
        lowest_bic = bic
        best_gmm = gmm

#6. Plot the Results from Steps 3 and 4
import matplotlib.pyplot as plt

# Plot PCA explained variance
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('PCA Explained Variance')
plt.show()

# Plot BIC scores
plt.figure(figsize=(10, 6))
plt.plot(n_components_range, bic_scores, label='BIC Scores')
plt.xlabel('Number of Components')
plt.ylabel('BIC Score')
plt.title('BIC Scores by Number of Components')
plt.legend()
plt.show()

#7. Output Hard Clustering Assignments
hard_assignments = best_gmm.predict(X_test_pca)

# Print the first few hard assignments
print("Hard Assignments (Cluster Labels):")
print(hard_assignments[:10])  

#8. Output Soft Clustering Probabilities
soft_assignments = best_gmm.predict_proba(X_test_pca)

# Print the first few soft assignments
print("Soft Assignments (Cluster Probabilities):")
print(soft_assignments[:10])  

#9. Generate New Faces Using the Model
generated_faces_pca = best_gmm.sample(10)[0]
generated_faces = pca.inverse_transform(generated_faces_pca)

# Visualize generated faces
fig, axes = plt.subplots(2, 5, figsize=(10, 4), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(generated_faces[i].reshape(64, 64), cmap='gray')
plt.show()

#10. Modify Some Images
import cv2
import matplotlib.pyplot as plt

# Original image
original_image = X_test[0].reshape(64, 64)

# Rotate the image
rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)

# Flip the image horizontally
flipped_image = cv2.flip(original_image, 1)

# Darken the image by multiplying pixel values by a factor
darkened_image = cv2.convertScaleAbs(original_image, alpha=0.9, beta=0)

# Plot all images for comparison
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(original_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(rotated_image, cmap='gray')
axes[1].set_title('Rotated Image')
axes[1].axis('off')

axes[2].imshow(flipped_image, cmap='gray')
axes[2].set_title('Flipped Image')
axes[2].axis('off')

axes[3].imshow(darkened_image, cmap='gray')
axes[3].set_title('Darkened Image')
axes[3].axis('off')

plt.show()

#11. Detect Anomalies
normal_score = best_gmm.score_samples(X_test_pca)
anomaly_score = best_gmm.score_samples(pca.transform(rotated_image.flatten().reshape(1, -1)))
anomaly_score2 = best_gmm.score_samples(pca.transform(flipped_image.flatten().reshape(1, -1)))
anomaly_score3 = best_gmm.score_samples(pca.transform(darkened_image.flatten().reshape(1, -1)))

print(f'Normal Image Score: {normal_score[0]}')
print(f'Anomalous Image Score: {anomaly_score[0]}')
print(f'Anomalous Image2 Score: {anomaly_score2[0]}')
print(f'Anomalous Image3 Score: {anomaly_score3[0]}')
