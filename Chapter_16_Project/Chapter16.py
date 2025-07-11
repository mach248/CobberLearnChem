import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
features_only_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(features_only_df)

# Add cluster labels to DataFrame
features_only_df['cluster'] = cluster_labels

# Plot petal length vs. petal width, colored by cluster
plt.figure(figsize=(8, 6))
plt.scatter(features_only_df['petal length (cm)'],
            features_only_df['petal width (cm)'],
            c=features_only_df['cluster'], cmap='viridis', s=60)

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('K-Means Clustering of Iris Dataset (3 Clusters)')
plt.grid(True)
plt.colorbar(label='Cluster')
plt.savefig("Clustering.png", bbox_inches='tight', dpi=300)
plt.show()
