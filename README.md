Retail Banking Customer Segmentation & Cluster Analysis
An unsupervised machine learning pipeline designed to segment retail banking customers into distinct behavioral personas using K-Means Clustering and Principal Component Analysis (PCA).
📌 Executive Summary
This project processes multi-dimensional customer financial data—including credit scores, annual income, transaction frequencies, and total spending values—to discover natural customer segments. By leveraging StandardScaler, K-Means, and PCA dimensionality reduction, the pipeline identifies optimal cluster counts and delivers clear business profiles for targeted financial product strategy.
📁 Repository Structure
├── data/                  # Raw and processed datasets
├── outputs/               # Saved visualization plots (Elbow curve, Silhouette, PCA projection)
├── src/
│   └── code1.py           # Core ML execution script
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation

🛠️ Methodology & Pipeline Structure
 * Data Synthesis & Rescaling: Synthetic retail banking features generated via make_blobs and rescaled into realistic domains (e.g., credit scores 300–850, annual income $15k–$150k).
 * Feature Standardization: Zero-mean, unit-variance scaling using StandardScaler to satisfy distance-based algorithm constraints.
 * Cluster Optimization:
   * Elbow Method: Evaluates within-cluster inertia reduction across K \in [2, 8].
   * Silhouette Score: Evaluates inter-cluster separation to auto-select optimal K.
 * Validation Metrics: Computes Silhouette Score and Calinski-Harabasz Index to quantify global cluster compactness and separation.
 * Dimensionality Reduction: 2D projection via Principal Component Analysis (PCA) to visualize high-dimensional clusters and centroid locations.
 * Persona Profiling: Aggregates feature means across identified clusters for data-driven customer profiling.
📊 Model Evaluation & Metrics
| Metric | Score / Output | Description |
|---|---|---|
| Optimal Clusters (K) | Auto-selected | Selected based on peak Silhouette score |
| Silhouette Score | Outputted in console | Measures cluster boundary separation (-1 to +1) |
| Calinski-Harabasz Index | Outputted in console | Ratio of between-cluster to within-cluster dispersion |
| PCA Variance Explained | Computed dynamically | Percentage of original feature variance captured in 2D space |
🚀 Quick Start & Installation
1. Prerequisites
Ensure Python 3.9+ is installed on your system.
2. Setup Environment
Clone this repository and install the required dependencies:
git clone https://github.com/your-username/retail-bank-customer-segmentation.git
cd retail-bank-customer-segmentation
pip install -r requirements.txt

3. Run Pipeline
Execute the full segmentation script:
python src/code1.py

All output visualizations will automatically save to the outputs/ directory.
📈 Visualizations Output
The pipeline generates two core diagnostic plots during execution:
 * Optimization Curves: Side-by-side Inertia reduction (Elbow curve) and Silhouette score profile.
 * 2D PCA Segment Plot: High-dimensional cluster visualization with mapped centroids and total explained variance ratio.
