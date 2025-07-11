import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y_true = iris.target  # Actual species: 0 = setosa, 1 = versicolor, 2 = virginica

# Fit KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
y_clusters = kmeans.fit_predict(X)

# Create a confusion matrix to compare
conf_matrix = confusion_matrix(y_true, y_clusters)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Cluster 0', 'Cluster 1', 'Cluster 2'],
            yticklabels=iris.target_names)
plt.xlabel('Cluster Label')
plt.ylabel('True Species')
plt.title('Confusion Matrix: KMeans Clusters vs. True Labels')
plt.savefig("Confusion_Matrix.png", bbox_inches='tight', dpi=300)
plt.show()
