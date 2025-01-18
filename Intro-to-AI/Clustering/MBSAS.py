from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

def MBSAS(data, threshold, max_clusters):
    Clusters = []
    
    m = 1
    C_start = [data[0]]
    Clusters.append(C_start)

    #first pass
    for i in range(1, len(data)):
        xi = data[i]
        distance, Ck = best_cluster(xi, Clusters)

        if distance > threshold and m < max_clusters:
            m += 1
            C = [data[i]]
            Clusters.append(C)

    #second pass
    for i in range(len(data)):
        xi = data[i]
        if not already_in_cluster(xi, Clusters):
            distance, Ck = best_cluster(xi, Clusters)
            Ck.append(xi)

    return Clusters

# min proximity function between vector and set: d(x,C) = min [d(x,y) for all y in C]
def best_cluster(vector, clusters):
    best_cluster = None

    min_distance = float('inf')

    for cluster in clusters:
        for vec in cluster:

            distance = euclidean_distance(vector, vec)

            if distance < min_distance:
                best_cluster = cluster
                min_distance = distance
            else:
                continue
    
    return min_distance, best_cluster

def euclidean_distance(vector1, vector2):
    distance = 0

    for feature1, feature2 in zip(vector1, vector2):
        distance += (feature1 - feature2)**2

    return sqrt(distance)

def already_in_cluster(vector, clusters):
    for cluster in clusters:
        if vector in cluster:
            return True

    return False

def plot_theta_estimator(theta_range):
    num_of_clusters = []

    for theta in theta_range:
        random.shuffle(data)
        Clusters = MBSAS(data, theta, 70)
        num_of_clusters.append(len(Clusters))

    plt.plot(theta_range, num_of_clusters)
    plt.title('Cluster Estimator Plot')
    plt.xlabel('Theta')
    plt.ylabel('Number of Clusters')
    plt.show()


file = '/Users/daviddechantsreiter/Desktop/WPI/Courses/Intro to AI/HW3/cluster_data.txt'
data = pd.read_csv(file, delim_whitespace=True, header=None, usecols=[1,2])

data.columns = ['Width', 'Length']

cluster_data = data[['Width', 'Length']].values


estimator_range = np.arange(0.5, 30, 0.5)
plot_theta_estimator(estimator_range)


'''
colors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'w', 'orange', 'purple', 'pink', 'brown', 'teal', 'navy', 'lime']

for i, cluster in enumerate(Clusters):
    x, y = zip(*cluster)
    plt.scatter(x, y, c=colors[i])

plt.legend()
plt.show()
'''


