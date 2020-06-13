import pandas as pd
import math
import numpy as np
from neural_network.plots import plot_measure_results_data
from genetic.lab2.CuttingStock import CuttingStock
import matplotlib.pyplot as plt


AAscores1, labels1, r_labels = [], [], []
for r in [800, 850, 1000, 1100, 1200]:
    r_labels.append(str(r))
    Ascores1 = []
    for size in [500, 1000, 1500, 2000]:
        labels1.append(f'Population size = {size}')
        cut = CuttingStock(r, "r" + str(r) + ".csv", n=size)
        circle = r*r*math.pi
        indexes = np.argsort(cut.evaluate())
        scores = []
        for i in range(50):
            cut.learn_population(1)
            indexes = np.argsort(cut.evaluate())
            res = cut.__evaluate_one__(indexes[0], return_area=False)
            scores.append(res / cut.best_rectangle_score_per_unit / circle)
        Ascores1.append(scores)
    AAscores1.append(Ascores1)

plt.figure(figsize=(12.8, 14.4))
for i, Ascores1 in enumerate(AAscores1):
    plt.subplot(3, 2, i+1)
    plot_measure_results_data(Ascores1, title_base=f"Score for R = {r_labels[i]}", labels=labels1, ylabel="Score normalized", show=False)
plt.show()


AAscores2, labels2, r_labels = [], [], []
for r in [800, 850, 1000, 1100, 1200]:
    r_labels.append(str(r))
    Ascores1 = []
    for mutation_percentage in [0.1, 0.2, 0.3, 0.5]:
        labels2.append(f'Mutation percentage = {mutation_percentage}')
        cut = CuttingStock(r, "r" + str(r) + ".csv", n=1500)
        circle = r*r*math.pi
        indexes = np.argsort(cut.evaluate())
        scores = []
        for i in range(50):
            cut.learn_population(1)
            indexes = np.argsort(cut.evaluate())
            res = cut.__evaluate_one__(indexes[0], return_area=False)
            scores.append(res / cut.best_rectangle_score_per_unit / circle)
        Ascores1.append(scores)
    AAscores2.append(Ascores1)

plt.figure(figsize=(12.8, 14.4))
for i, Ascores1 in enumerate(AAscores2):
    plt.subplot(3, 2, i+1)
    plot_measure_results_data(Ascores1, title_base=f"Score for R = {r_labels[i]}", labels=labels2, ylabel="Score normalized", show=False)
plt.show()

# all area means area of the strips - can be used by rectangles
# area means the area used by rectangles but it is not the area of rectangles:
# why? example:
# if on the strip of height 100 and width 1000 there is a rectangle of height 80 and width 900
# all area = 100 * 1000
# area = 100 * 900 (used)
# area of rectangle = 80 * 900
Ascores, Aall_areas, Aareas, labels = [], [], [], []
for r in [800, 850, 1000, 1100, 1200]:
    labels.append(f'R = {r}')
    cut = CuttingStock(r, "r" + str(r) + ".csv", n=1500, mutation_percentage=0.3)
    circle = r*r*math.pi
    indexes = np.argsort(cut.evaluate())
    scores, all_areas, areas = [], [], []
    for i in range(20):
        cut.learn_population(1)
        indexes = np.argsort(cut.evaluate())
        res = cut.__evaluate_one__(indexes[0], return_area=True)
        scores.append(res[0] / cut.best_rectangle_score_per_unit / circle)
        all_areas.append(res[1]/circle)
        areas.append(res[2]/circle)
        # print(f'Result {res[0]}, found_area {res[1]/circle}, used_area {res[2]/circle}')
    Ascores.append(scores)
    Aall_areas.append(all_areas)
    Aareas.append(areas)

plt.figure(figsize=(12.8, 9.6))
plt.subplot(221)
plot_measure_results_data(Ascores, title_base="Score ", labels=labels, ylabel="Score normalized", show=False)
plt.subplot(222)
plot_measure_results_data(Aall_areas, title_base="Possible area percent ", labels=labels, ylabel="Percent", show=False)
plt.subplot(223)
plot_measure_results_data(Aareas, title_base="Used area percent ", labels=labels, ylabel="Percent", show=False)
plt.show()
