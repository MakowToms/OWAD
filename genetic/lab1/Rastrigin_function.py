from genetic.Genetic import Genetic
import numpy as np
from neural_network.plots import plot_measure_results_data


# function to evaluate how good is population (each person)
def eval_function(population, n=5, A=10):
    result = np.ones([population.shape[0]]) * A * n
    for i in range(n):
        result += (population[:, i] ** 2) - A * np.cos(2*np.pi*population[:, i])
    return result


# create genetic class
mses0 = []
population_size = [100, 200, 500, 1000]
for size in population_size:
    best_results = []
    gen = Genetic(size, 5, eval_function, mutation_coef=1, mutation_percentage=0.2)
    for i in range(100):
        gen.learn_population(epochs=1)
        best_results.append(gen.get_best()[0])
    mses0.append(best_results)
    print(f'Ended {size}')

mses1 = []
mutation_percentages = [0.05, 0.1, 0.2, 0.3]
for mutation_percentage in mutation_percentages:
    best_results = []
    gen = Genetic(500, 5, eval_function, mutation_coef=1, mutation_percentage=mutation_percentage)
    for i in range(100):
        gen.learn_population(epochs=1)
        best_results.append(gen.get_best()[0])
    mses1.append(best_results)
    print(f'Ended {mutation_percentage}')

mses2 = []
mutation_coefs = [0.1, 0.2, 0.5, 1]
for mutation_coef in mutation_coefs:
    best_results = []
    gen = Genetic(500, 5, eval_function, mutation_coef=mutation_coef, mutation_percentage=0.3)
    for i in range(100):
        gen.learn_population(epochs=1)
        best_results.append(gen.get_best()[0])
    mses2.append(best_results)
    print(f'Ended {mutation_coef}')

# plot results
title = "Function value "
y_label = "Logarithm of function value "
labels = ["Population size = " + str(val) for val in population_size]
plot_measure_results_data(mses0, title_base=title, ylabel=y_label, labels=labels, from_error=0, y_log=True)

labels = ["Mutation percentage = " + str(val) for val in mutation_percentages]
plot_measure_results_data(mses1, title_base=title, ylabel=y_label, labels=labels, from_error=0, y_log=True)

labels = ["Mutation coefficient = " + str(val) for val in mutation_coefs]
plot_measure_results_data(mses2, title_base=title, ylabel=y_label, labels=labels, from_error=0, y_log=True)
