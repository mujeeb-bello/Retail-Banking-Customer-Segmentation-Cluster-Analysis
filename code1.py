import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

#SETTING VISUALIZATION STYLE
sns.set(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'DejaVu Sans'

np.random.seed(42)
raw_data, _ = make_blobs(
    n_samples=1000,
    n_features=5,
    centers=4,
    cluster_std=1.2,
    random_state=42
)

columns = ['age', 'annual_income', 'credit_score', 'total_spending_val', 'tx_frequency']
df = pd.DataFrame(raw_data, columns=columns)

# Rescale features to realistic business ranges
df['age'] = np.interp(df['age'], (df['age'].min(), df['age'].max()), (18, 70)).round(0)
df['annual_income'] = np.interp(df['annual_income'], (df['annual_income'].min(), df['annual_income'].max()), (15000, 150000)).round(2)
df['credit_score'] = np.interp(df['credit_score'], (df['credit_score'].min(), df['credit_score'].max()), (300, 850)).round(0)
df['total_spending_val'] = np.interp(df['total_spending_val'], (df['total_spending_val'].min(), df['total_spending_val'].max()), (500, 25000)).round(2)
df['tx_frequency'] = np.interp(df['tx_frequency'], (df['tx_frequency'].min(), df['tx_frequency'].max()), (2, 120)).round(0)

# Save the synthetic dataset to CSV
df.to_csv(r'C:\Users\USER\Desktop\Machine Learning\Project 1\data\synthetic_customer_data.csv', index=False)

print("📌 Initial Dataset Preview:")
print(df.head())
print("\n" + "="*60 + "\n")

# Perfroming Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# K-DETERMINATION: ELBOW METHOD & SILHOUETTE ANALYSIS

inertia_scores = []
silhouette_scores = []
k_range = range(2, 9)

for k in k_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    
    inertia_scores.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Plot Optimization Diagnostics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Curve
ax1.plot(k_range, inertia_scores, marker='o', color='#1f77b4', linewidth=2)
ax1.set_title('Elbow Method (Inertia Reduction)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia (Sum of Squared Distances)')
ax1.grid(True, linestyle='--')

# Silhouette Score Curve
ax2.plot(k_range, silhouette_scores, marker='s', color='#d62728', linewidth=2)
ax2.set_title('Silhouette Score Evaluation', fontsize=12, fontweight='bold')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.grid(True, linestyle='--')

plt.tight_layout()
plt.show()

# Select Optimal K based on peak silhouette score
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"✅ Optimal K Selected: {optimal_k} (Peak Silhouette Score: {max(silhouette_scores):.4f})")

# =========================================================
# FINAL K-MEANS MODELING
# =========================================================
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = final_kmeans.fit_predict(X_scaled)

# Evaluate Validation Metrics
sil_val = silhouette_score(X_scaled, df['Cluster'])
ch_val = calinski_harabasz_score(X_scaled, df['Cluster'])

print(f"📊 Final Model Silhouette Score: {sil_val:.4f}")
print(f"📊 Final Model Calinski-Harabasz Index: {ch_val:.2f}")

# =========================================================
# PCA 2D CLUSTER VISUALIZATION
# =========================================================
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(pca_coords, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df['Cluster']

# Transform cluster centers to PCA space
centers_pca = pca.transform(final_kmeans.cluster_centers_)

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=pca_df, x='PC1', y='PC2', hue='Cluster',
    palette='tab10', style='Cluster', s=60, alpha=0.85
)

plt.scatter(
    centers_pca[:, 0], centers_pca[:, 1],
    c='black', marker='X', s=200, label='Centroids', edgecolor='white'
)

exp_var = np.sum(pca.explained_variance_ratio_) * 100
plt.title(f'K-Means Customer Segments (2D PCA Projection - {exp_var:.1f}% Variance Explained)', fontweight='bold')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# =========================================================
# BUSINESS PROFILING & PERSONA GENERATION
# =========================================================
cluster_profiles = df.groupby('Cluster').mean().round(2)
cluster_counts = df['Cluster'].value_counts().rename('Customer_Count')
cluster_summary = pd.concat([cluster_counts, cluster_profiles], axis=1)

print("\n" + "="*60)
print("🎯 CUSTOMER SEGMENT PROFILES (FEATURE MEANS)")
print("="*60)
print(cluster_summary.to_string())