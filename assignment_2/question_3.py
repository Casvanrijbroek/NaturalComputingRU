import numpy as np
from random import sample
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd

# Settings
iris_data_path = R"C:\Users\anton\Documents\Radboud\MEGAsync\Natural Computing\iris.data"

### Some helper methods
def get_distance(z, centroid):
    return np.linalg.norm(np.array(z) - np.array(centroid))

def get_QE(clusters):
    cluster_fitness_list = []
    for c in clusters:
        c_distances = [get_distance(c[0], c[1][i]) for i in range(len(c[1]))]
        try:
            c_fitness = sum(c_distances) / len(c[1])
        except:
            c_fitness = 0
        cluster_fitness_list.append(c_fitness)
    return sum(cluster_fitness_list) / len(cluster_fitness_list)

### Particle swarm optimization
def PSO(n_centroids, n_particles, n_iterations, data, w=0.72, a1=1.49, a2=1.49, r1=0.5, r2=0.5):
    
    # Initialization
    centroids = sample(range(len(data)), n_centroids*n_particles)
    particles = np.array([centroids[i*n_centroids:i*n_centroids+n_centroids] for i in range(n_particles)])

    v_list = [0 for i in range(n_particles)]
    local_best = [[10000, []] for i in range(n_particles)]
    global_best = [10000, []]

    clusters = []
    for x in particles:
        oi = []
        for c in x:
            oi.append([data[c], []])
        clusters.append(oi)
    particles = clusters

    # Perform iterations
    for iteration in range(n_iterations):
        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                clusters[i][j][1] = []
            for z in data:
                distances = [get_distance(z, clusters[i][j][0]) for j in range(len(clusters[i]))]
                min_dist = min(distances)
                closest_centroid = distances.index(min_dist)
                clusters[i][closest_centroid][1].append(z)

            fitness = get_QE(clusters[i])
            if fitness < local_best[i][0]:
                local_best[i] = [fitness, [clusters[i][j][0] for j in range(len(clusters[i]))]]
            if fitness < global_best[0]:
                global_best = [fitness, [clusters[i][j][0] for j in range(len(clusters[i]))]]
        
        for i in range(n_particles):
            v_i = w*v_list[i] + a1*r1*(local_best[i][1] - np.array([x[0] for x in clusters[i]])) + a2*r2*(global_best[1] - np.array([x[0] for x in clusters[i]]))
            v_list[i] = v_i
        
            for j in range(len(clusters[i])):
                clusters[i][j][0] = clusters[i][j][0] + v_i[j]

    clusters = [[centroid, []] for centroid in global_best[1]]
    for z in data:
        distances = [get_distance(z, centroid) for centroid in global_best[1]]
        min_dist = min(distances)
        closest_centroid_index = distances.index(min_dist)
        clusters[closest_centroid_index][1].append(z)

    return clusters

### K-means clustering
def Kmeans(k, n_iterations, data):
    # Initialization
    init_centroids = sample(range(len(data)), k)
    clusters = [[data[init_centroids[i]], [data[init_centroids[i]]], i] for i in range(len(init_centroids))]
    
    # Perform iterations
    for iteration in range(n_iterations):
        clusters = [[c[0], [], c[2]] for c in clusters]
        for z in data:
            centroid_distances = [get_distance(z, c[0]) for c in clusters]
            closest_centroid_index = centroid_distances.index(min(centroid_distances))
            clusters[closest_centroid_index][1].append(z)

        for i in range(k): # update centroids
            clusters[i][0] = np.mean(np.array(clusters[i][1]), axis=0)
    
    return clusters

# Generate artificial dataset 1
AD1 = (np.random.random((400, 2)) * 2 - 1)
AD1_labels = np.logical_or(AD1[:,0]>=0.7, np.logical_and(AD1[:,0]<=0.3, AD1[:,1]>=-0.2-AD1[:,0]))

# Read iris data
iris_data, iris_labels = [], []
with open(iris_data_path, 'r') as f:
    for line in f.readlines():
        if line != '\n':
            iris_data.append(np.array([float(i) for i in line.split(',')[:-1]]))
            iris_labels.append(line.split(',')[-1].strip('\n'))

# Run algorithms 30 times and take average quantization error
iris_pso_trials, iris_kmeans_trials, AD1_pso_trials, AD1_kmeans_trials = [], [], [], []
for i in range(30):
    iris_pso_clusters = PSO(n_centroids=3, n_particles=10, n_iterations=100, data=iris_data)
    iris_kmeans_clusters = Kmeans(k=3, n_iterations=100, data=iris_data)
    iris_pso_trials.append(get_QE(iris_pso_clusters))
    iris_kmeans_trials.append(get_QE(iris_kmeans_clusters))

    AD1_pso_clusters = PSO(n_centroids=2, n_particles=10, n_iterations=100, data=AD1)
    AD1_kmeans_clusters = Kmeans(k=2, n_iterations=100, data=AD1)
    AD1_pso_trials.append(get_QE(AD1_pso_clusters))
    AD1_kmeans_trials.append(get_QE(AD1_kmeans_clusters))

iris_pso_QE = sum(iris_pso_trials)/len(iris_pso_trials)
iris_kmeans_QE = sum(iris_kmeans_trials)/len(iris_kmeans_trials)
AD1_pso_QE = sum(AD1_pso_trials)/len(AD1_pso_trials)
AD1_kmeans_QE = sum(AD1_kmeans_trials)/len(AD1_kmeans_trials)


### Plotting
n_groups = 2
means_pso = (iris_pso_QE, AD1_pso_QE)
means_kmeans = (iris_kmeans_QE, AD1_kmeans_QE)

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.15
opacity = 1

rects1 = plt.bar(index, means_pso, bar_width, alpha=opacity, color='b', label='Particle swarm optimization')
rects2 = plt.bar(index + bar_width, means_kmeans, bar_width, alpha=opacity, color='g', label='K-means')

plt.ylabel('Scores')
plt.xticks(index + 0.5*bar_width, ('Iris dataset', 'Artificial dataset 1'))
plt.legend()

plt.tight_layout()
plt.show()

# plot clusters for AD1 trial #30
some_colors = ['black', 'red', 'green', 'yellow', 'blue']

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10,15))

for i in range(len(AD1)):
    axs[0, 0].scatter(AD1[i,0], AD1[i,1], color=some_colors[int(AD1_labels[i]==True)])
    axs[0, 0].set_title('Original artifical dataset 1')

for i in range(len(AD1_pso_clusters)):
    axs[1, 0].scatter(np.array(AD1_pso_clusters[i][1])[:,0], np.array(AD1_pso_clusters[i][1])[:,1], color=some_colors[i])
    axs[1, 0].set_title('PSO clustering of artificial dataset 1')

for i in range(len(AD1_kmeans_clusters)):
    axs[2, 0].scatter(np.array(AD1_kmeans_clusters[i][1])[:,0], np.array(AD1_kmeans_clusters[i][1])[:,1], color=some_colors[i])
    axs[2, 0].set_title('K-means clustering of artificial dataset 1')


def plot_iris(transformed_iris_data, labels, ax, title, classes):
    transformed_iris_data['label'] = labels
    for c in classes:
        ax.scatter(transformed_iris_data[transformed_iris_data['label']==c][0], transformed_iris_data[transformed_iris_data['label']==c][1], label=str(c), c=some_colors[classes.index(c)])
        ax.set_title(title)
    ax.legend()

pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed_iris_data = pd.DataFrame(pca.fit_transform(iris_data))

iris_kmeans_labels, iris_pso_labels = [], []
for z in iris_data:

    for i in range(len(iris_pso_clusters)):
        twodlist_b = [list(x) for x in iris_pso_clusters[i][1]]
        if list(z) in twodlist_b:
            iris_pso_labels.append(i)

    for i in range(len(iris_kmeans_clusters)):
        twodlist_a = [list(x) for x in iris_kmeans_clusters[i][1]]
        if list(z) in twodlist_a:
            iris_kmeans_labels.append(i)

plot_iris(transformed_iris_data, iris_labels, axs[0, 1], 'Original iris dataset', ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
plot_iris(transformed_iris_data, iris_pso_labels, axs[1, 1], 'PSO clustering of the iris dataset', [0, 1, 2])
plot_iris(transformed_iris_data, iris_kmeans_labels, axs[2, 1], 'K-means clustering of the iris dataset', [0, 1, 2])

plt.show()
