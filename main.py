import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    """
    Main function for spectral-cluster-pipeline-18.

    This function takes in a dataset, applies standard scaling, and performs KMeans clustering.
    """
    # Load the dataset
    dataset = pd.read_csv('dataset.csv')

    # Apply standard scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(dataset)

    # Define the number of clusters
    num_clusters = 5

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(scaled_data)

    # Get the cluster labels
    labels = kmeans.labels_

    # Print the cluster labels
    print(labels)

    # Print the number of clusters
    print("Number of clusters:", num_clusters)

    # Print the cluster centers
    print("Cluster centers:\n", kmeans.cluster_centers_)

if __name__ == "__main__":
    main()