import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the data
df = pd.read_csv('alkaline_earth_metals.csv')

# Select features for clustering
X = df[['Atomic Radius (pm)', 'First Ionization Energy (kJ/mol)']]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=0)
df['Cluster'] = kmeans.fit_predict(X)

# Plot the clusters
plt.figure(figsize=(8, 6))
colors = ['red', 'green']
for cluster in df['Cluster'].unique():
    subset = df[df['Cluster'] == cluster]
    plt.scatter(subset['Atomic Radius (pm)'], subset['First Ionization Energy (kJ/mol)'],
                label=f'Cluster {cluster}', color=colors[cluster])
    for i in subset.index:
        plt.text(subset['Atomic Radius (pm)'][i] + 1,
                 subset['First Ionization Energy (kJ/mol)'][i],
                 df['Symbol'][i])

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='blue', marker='X', label='Centroids')

# Plot details
plt.title('K-Means Clustering of Alkaline Earth Metals (k=2)')
plt.xlabel('Atomic Radius (pm)')
plt.ylabel('First Ionization Energy (kJ/mol)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('alkaline_earth_kmeans.png')
plt.show()
