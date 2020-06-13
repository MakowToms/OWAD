from genetic.data_manipulation import load_iris
import numpy as np
from genetic.neat import NEAT
import copy
from neural_network.plots import plot_data_2d, plot_measure_results_data
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from neural_network.activations import softmax

train, classes = load_iris()
ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(classes.reshape([-1, 1]))


# create score function for iris
def score_function(res, type="score"):
    results = np.vstack([res[4], res[5], res[6]])
    results = softmax(results).transpose()
    if type == "score":
        return - np.sum((results - y_ohe)**2) / res[4].shape[0] / 3
    else:
        results = results.transpose()
        return np.sum(classes == np.argmax(results, axis=0)) / res[4].shape[0]


data = {0: train[:, 0], 1: train[:, 1], 2: train[:, 2], 3: train[:, 3]}

# plot data classes
plt.figure(figsize=(12.8, 9.6))
plt.subplot(221)
pca = PCA(n_components=2)
pca.fit(train)
x_transformed = pca.transform(train)
plot_data_2d(x_transformed[:, 0], x_transformed[:, 1], classes, title='True classes of points on the plane', show=False)

n_input = 4
n_output = 3
n = 200
scores, accuracies = [], []
labels = ["Without speciation, maximize accuracy", "With speciation, maximize accuracy", "With speciation, maximize score"]
for index, score_type, speciation in zip([2, 3, 4], ["accuracy", "accuracy", "score"], [False, True, True]):
    neat = NEAT(n_input, n_output, n, score_function, data, 1.3, 15, score_type, use_speciation=speciation)

    for i in range(50):
        print(f'Start of epoch: {i+1}')
        neat.epoch(i)

    scores.append(neat.best_scores)
    accuracies.append(neat.best_accuracies)

    best = neat.get_best_population("accuracy")[0]
    res = neat.population[best].evaluate(copy.deepcopy(data))
    results = np.vstack([res[4], res[5], res[6]])
    y_predicted = np.argmax(results, axis=0)
    plt.subplot(2, 2, index)
    plot_data_2d(x_transformed[:, 0], x_transformed[:, 1], y_predicted, title=labels[index-2], show=False)

plt.show()

plot_measure_results_data(accuracies, title_base="Accuracy ", ylabel="Accuracy ", labels=labels, from_error=0)


n_input = 4
n_output = 3
n = 200
scores, accuracies = [], []
labels = ["Speciation threshold = 1", "Speciation threshold = 1.3", "Speciation threshold = 1.5"]
for speciation_threshold in [1, 1.3, 1.5]:
    neat = NEAT(n_input, n_output, n, score_function, data, speciation_threshold, 15, "accuracy", use_speciation=True)

    for i in range(50):
        print(f'Start of epoch: {i+1}')
        neat.epoch(i)

    scores.append(neat.best_scores)
    accuracies.append(neat.best_accuracies)

plot_measure_results_data(accuracies, title_base="Accuracy ", ylabel="Accuracy ", labels=labels, from_error=0)
