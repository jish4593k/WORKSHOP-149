import argparse
import random
import math
import copy
from scipy.spatial.distance import cdist
import numpy as np

class KMeans:
    def __init__(self, database_file, k_clusters, max_iterations, min_distance, output_file):
        self.database_file = database_file
        self.k_clusters = k_clusters
        self.max_iterations = max_iterations
        self.min_distance = min_distance
        self.output_file = output_file

    def read_data(self):
        with open(self.database_file, 'r') as fp:
            return np.array([list(map(float, line.split())) for line in fp])

    def get_centroids(self, database):
        row_candidates = []
        centroids = []
        for _ in range(self.k_clusters):
            while True:
                row = random.randint(0, len(database) - 2)
                if row not in row_candidates:
                    row_candidates.append(row)
                    centroids.append(copy.deepcopy(database[row]))
                    break
        return centroids

    def separate(self, database, centroids):
        distances = cdist(database, centroids)
        return np.argmin(distances, axis=1)

    def update_centroids(self, database, clusters, centroids):
        for i in range(self.k_clusters):
            cluster_points = database[clusters == i]
            if len(cluster_points) > 0:
                centroids[i] = np.mean(cluster_points, axis=0)
        return centroids

    def check_e_distance(self, old_centroids, new_centroids):
        return np.max(np.sqrt(np.sum((old_centroids - new_centroids) ** 2, axis=1)))

    def print_output(self, clusters):
        with open(self.output_file, "w") as f:
            for i in range(self.k_clusters):
                f.write(f"{i}: {' '.join(map(str, np.where(clusters == i)[0]))}\n")

    def gen_kmeans(self, database):
        centroids = self.get_centroids(database)
        for _ in range(self.max_iterations):
            clusters = self.separate(database, centroids)
            old_centroids = copy.deepcopy(centroids)
            self.update_centroids(database, clusters, centroids)
            distance = self.check_e_distance(old_centroids, centroids)
            if distance < self.min_distance:
                break
        self.print_output(clusters)

    def main(self):
        database = self.read_data()
        self.gen_kmeans(database)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-database_file')
    parser.add_argument('-k')
    parser.add_argument('-max_iters')
    parser.add_argument('-eps')
    parser.add_argument('-output_file')

    args = parser.parse_args()
    kmeans = KMeans(args.database_file, int(args.k), int(args.max_iters), float(args.eps), args.output_file)
    kmeans.main()
