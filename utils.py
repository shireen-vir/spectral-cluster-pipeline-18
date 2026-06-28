import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    """
    Main function for the spectral-cluster-pipeline-18 tool.

    This tool takes in a dataset, applies standard scaling, and then performs K-means clustering.
    The number of clusters (K) is currently set to 5, but this can be adjusted as needed.
    """
    # Load the dataset
    data = pd.read_csv('data.csv')

    # Apply standard scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(scaled_data)

    # Get the cluster labels
    labels = kmeans.labels_

    # Print the cluster labels
    print(labels)

    # Save the cluster labels to a file
    np.save('cluster_labels.npy', labels)

if __name__ == '__main__':
    main()