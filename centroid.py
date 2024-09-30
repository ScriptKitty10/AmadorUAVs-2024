import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def load_coordinates(file_path):
    # Reads the txt file (with lots of error handling)
    coordinates = []
    try:
        with open(file_path, "r") as f:
            coords = f.readlines()
            for i, line in enumerate(coords):
                if i == 0:
                    try:
                        N = int(line.split(" ", 1)[0])
                    except ValueError:
                        print("Error: Could not parse the number of points from the first line.")
                        raise
                else:
                    pairs = line.strip().split()
                    try:
                        for j in range(0, len(pairs), 2):
                            x, y = float(pairs[j]), float(pairs[j + 1])
                            coordinates.append((x, y))
                    except (ValueError, IndexError):
                        print(f"Error: Could not parse coordinate pair on line {i + 1}.")
                        raise
        return np.array(coordinates)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        raise
    except PermissionError:
        print(f"Error: Permission denied while trying to read the file '{file_path}'.")
        raise

def main(file_path, distance_threshold_factor=2):
    # Load coordinates from txt file
    coordinates_np = load_coordinates(file_path)

    num_clusters = 5

    # K-Means clustering
    try:
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(coordinates_np)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        filtered_points = []

        # Outlier filtering
        for cluster in range(num_clusters):
            cluster_points = coordinates_np[labels == cluster]
            centroid = centroids[cluster]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)

            mean_distance = np.mean(distances)
            std_distance = np.std(distances)

            #Distance threshold helps with detecting outliers
            distance_threshold = mean_distance + distance_threshold_factor * std_distance
            non_outlier_mask = distances <= distance_threshold
            filtered_cluster_points = cluster_points[non_outlier_mask]
            filtered_points.append(filtered_cluster_points)

        filtered_points_np = np.vstack(filtered_points)

        # 2nd K-Means clustering for filtered points
        kmeans_filtered = KMeans(n_clusters=num_clusters, random_state=0).fit(filtered_points_np)
        labels_filtered = kmeans_filtered.labels_
        centroids_filtered = kmeans_filtered.cluster_centers_
        centroid_list = []

        # Plot clusters (optional)
        plt.figure(figsize=(8, 6))
        for cluster in range(num_clusters):
            cluster_points = filtered_points_np[labels_filtered == cluster]
            plt.plot(cluster_points[:, 0], cluster_points[:, 1], 'o', label=f'Cluster {cluster + 1}')

        # Plot the centroids of the clusters (also optional)
        for cluster in range(num_clusters):
            centroid = np.round(centroids_filtered[cluster], 5)
            plt.plot(centroid[0], centroid[1], 'x', markersize=10, color='black', label=f'Centroid {cluster + 1}')
            centroid_list.append(centroid)

        plt.legend()
        plt.title(f"Clusters of Points with Centroids (Filtered Outliers for {num_clusters} Clusters)")
        plt.show()

        # Write output to .out file
        output_file = "centers.out"
        try:
            with open(output_file, "w") as output:
                for point in centroid_list:
                    output.write(f"{point[0]} {point[1]}\n")
            print(f"Output successfully written to {output_file}!")
        except PermissionError:
            print(f"Error: Permission denied while trying to write to '{output_file}'.")
            raise

    except Exception as e:
        print(f"An error occurred during clustering or plotting: {e}")
        raise

if __name__ == "__main__":

    #Command line parsing
    parser = argparse.ArgumentParser(description="Centroid calculator, for plots with outlier or no outlier.")
    parser.add_argument("file", help="Path to the input txt file containing coordinates.")
    
    args = parser.parse_args()

    try:
        main(args.file)
    except Exception as e:
        print(f"An error occurred: {e}")
