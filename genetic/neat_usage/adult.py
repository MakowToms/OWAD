from genetic.data_manipulation import load_adult
import numpy as np
from genetic.neat import NEAT
import copy
from neural_network.plots import plot_data_2d, plot_measure_results_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from neural_network.activations import softmax

train, classes = load_adult()
ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(classes.reshape([-1, 1]))


# create score function for adult
def score_function(res, type="score"):
    results = np.vstack([res[108], res[109]])
    results = softmax(results).transpose()
    if type == "score":
        return - np.sum((results - y_ohe)**2) / res[108].shape[0] / 2
    else:
        results = results.transpose()
        return np.sum(classes == np.argmax(results, axis=0)) / res[108].shape[0]


# res = {2: np.array([1,2,3]), 3: np.array([3,1,2]), 4: np.array([0,4,0])}
data = {}
for i in range(train.shape[1]):
    data[i] = train[:, i]


# plot data classes
plt.figure(figsize=(12.8, 9.6))
plt.subplot(221)
pca = PCA(n_components=2)
pca.fit(train)
x_transformed = pca.transform(train)
plot_data_2d(x_transformed[:, 0], x_transformed[:, 1], classes, title='True classes of points on the plane', show=False)

n_input = 108
n_output = 2
n = 200
scores, accuracies = [], []
labels = ["Without speciation, maximize accuracy", "With speciation, maximize accuracy", "With speciation, maximize score"]
for index, score_type, speciation in zip([2, 3, 4], ["accuracy", "accuracy", "score"], [False, True, True]):
    neat = NEAT(n_input, n_output, n, score_function, data, 1.3, 15, score_type, use_speciation=speciation)

    for i in range(20):
        print(f'Start of epoch: {i+1}')
        neat.epoch(i)

    scores.append(neat.best_scores)
    accuracies.append(neat.best_accuracies)

    best = neat.get_best_population("accuracy")[0]
    res = neat.population[best].evaluate(copy.deepcopy(data))
    results = np.vstack([res[108], res[109]])
    y_predicted = np.argmax(results, axis=0)
    plt.subplot(2, 2, index)
    plot_data_2d(x_transformed[:, 0], x_transformed[:, 1], y_predicted, title=labels[index-2], show=False)

plt.show()

plot_measure_results_data(accuracies, title_base="Accuracy ", ylabel="Accuracy ", labels=labels, from_error=0)
