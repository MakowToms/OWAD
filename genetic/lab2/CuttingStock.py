import pandas as pd
import math
import numpy as np
from neural_network.plots import plot_measure_results_data


"""
Polish description:

 Rozwiazanie opiera sie na podziale kola na poziome paski. 
 Nastepnie paski zapelniane sa zachlannie zaczynajac od prostokata, 
 ktory da najwiekszy wzrost na jedna jednostke szerokosci paska.

Gen ciagly koduje jak wysoko zaczyna sie "pierwszy" gorny pasek - 
jest to liczba od 0 do najwiekszego wymiaru sposrod dostepnych prostokatow (
dla r=800 jest to 400). Geny dyskretne koduja, jaka wysokosc maja 
kolejne paski (paski stykaja sie ze soba). 
Dla r=800 dostepne wartosci to 30, ..., 400, zatem 1 koduje wysokosc 30, 
a 7 koduje wysokosc 400. Pierwsze floor(r/30) koduje wielkosci paskow
na gornej czesci, a nastepne floor((r+400)/30) koduje wielkosci paskow 
w dolnej czesci. Oczywiscie bardzo czesto ostatnie geny z obu "polowek" 
nie sa uzywane poniewaz wymagaja konkretnych wartosci w innych genach.

Mutacja to dodanie liczb do genow z rozkladem normalnym. 
Dla genow dyskretnych jest to wartosc calkowita zatem bardzo wiele genow 
nie mutuje - co jest raczej dobre (bo zmiany we wczesniejszych genach 
maja duzy wplyw na ewaluacje nastepnych genow).

Krzyzowanie jest to krzyzowanie jednopunktowe, gdzie gen ciagly jest 
brany z pierwszego wybranego osobnika."""
class CuttingStock:
    def __init__(self, r, file_name, n=1000, mutation_percentage=0.2, file_directory="genetic/data/cutting/"):
        self.rectangles, unique_sizes, self.best_rectangle_score_per_unit = CuttingStock.read_data(file_directory, file_name)
        self.unique_sizes = unique_sizes
        self.minimum = unique_sizes[0]
        self.maximum = unique_sizes[-1]
        self.max_gen_number = len(unique_sizes)
        self.gens_up = math.floor(r/self.minimum)
        self.n_gens = math.floor(r/self.minimum) + math.floor((r+self.maximum)/self.minimum)
        self.n = n
        self.r = r
        self.mutation_percentage = mutation_percentage
        self.discrete, self.continous = self.create_population()
        self.normalize_population()
        self.rectangle_container = self.create_rectangle_container()

    # contain rectangle ordered from the best according to score/used_width ratio
    def create_rectangle_container(self):
        containers = {}
        for size in self.unique_sizes:
            this_rectangles = []
            for rectangle in self.rectangles:
                min_val = min(rectangle[0], rectangle[1])
                max_val = max(rectangle[0], rectangle[1])
                if max_val <= size:
                    this_rectangles.append([rectangle[2] / min_val, min_val])
                elif min_val <= size:
                    this_rectangles.append([rectangle[2] / max_val, max_val])
            this_rectangles.sort(key=lambda val: -val[0])
            containers[size] = this_rectangles
        return containers

    # the population has some discrete gens (means which height to use)
    # and one continous gene means how many units from the middle of circle is the bottom of first upper strip
    def create_population(self):
        discrete = np.random.randint(0, self.max_gen_number, [self.n, self.n_gens]) + 1
        continous = np.random.normal(self.maximum/2, self.minimum/2, [self.n, 1])
        return discrete, continous

    def mutate_population(self):
        n = self.n
        n_mutated = int(n*self.mutation_percentage)
        observations_mutated_discrete = np.random.choice(np.linspace(0, n-1, n, dtype=np.int), n_mutated)
        self.discrete[observations_mutated_discrete, :] += np.random.standard_normal([n_mutated, self.n_gens]).astype(int)
        observations_mutated_continous = np.random.choice(np.linspace(0, n-1, n, dtype=np.int), n_mutated)
        self.continous[observations_mutated_continous, :] += np.random.normal(0, self.minimum/2, [n_mutated, 1])
        self.normalize_population()

    # do not allow to left value outside bounds
    def normalize_population(self):
        self.discrete[self.discrete > self.max_gen_number] = self.max_gen_number
        self.discrete[self.discrete < 1] = 1
        self.continous[self.continous < 0] = 0
        self.continous[self.continous > self.maximum] = self.maximum
        self.discrete = self.discrete.astype(int)

    def cross_population(self, best_discrete, best_continous):
        n_best = best_discrete.shape[0]
        discrete = np.zeros([self.n, self.n_gens])
        discrete[:n_best, :] = best_discrete
        continous = np.zeros([self.n, 1])
        continous[:n_best, :] = best_continous
        for i in range(n_best, self.n):
            d, c = self.create_new_person(best_discrete, best_continous)
            discrete[i, :] = d
            continous[i, :] = c
        self.discrete = discrete
        self.continous = continous

    # one point cross of two objects from best population
    def create_new_person(self, best_discrete, best_continous):
        n_best = best_discrete.shape[0]
        to_cross = np.random.choice(np.linspace(0, n_best-1, n_best, dtype=np.int), 2, replace=False)
        cross_bit = np.random.choice(np.linspace(1, self.n_gens-1, self.n_gens-1, dtype=np.int), 1)[0]
        new = np.zeros([1, self.n_gens])
        new[0, :cross_bit] = best_discrete[to_cross[0], :cross_bit]
        new[0, cross_bit:] = best_discrete[to_cross[1], cross_bit:]
        return new, best_continous[to_cross[0]]

    def evaluate(self):
        scores = np.zeros([self.n])
        for i in range(self.n):
            scores[i] = -self.__evaluate_one__(i)
        return scores

    # compute score of one person
    def __evaluate_one__(self, person_number, return_area=False):
        score = 0
        area = 0
        all_area = 0
        r = self.r
        current_up = self.continous[person_number, 0]
        gen_number = 0
        while current_up < r:
            this_height = self.unique_sizes[self.discrete[person_number, gen_number]-1]
            current_up += this_height
            if current_up >= r:
                break
            if return_area:
                new_score, new_all_area, new_area = self.compute_score_for_strip(current_up, this_height, True)
                score += new_score
                all_area += new_all_area
                area += new_area
            else:
                score += self.compute_score_for_strip(current_up, this_height)
            gen_number += 1
        current_down = -self.continous[person_number, 0]
        gen_number = self.gens_up
        while current_down < r:
            this_height = self.unique_sizes[self.discrete[person_number, gen_number]-1]
            current_down_new = current_down + this_height
            if current_down_new >= r:
                break
            if return_area:
                new_score, new_all_area, new_area = self.compute_score_for_strip(max(abs(current_down), abs(current_down_new)), this_height, True)
                score += new_score
                all_area += new_all_area
                area += new_area
            else:
                score += self.compute_score_for_strip(max(abs(current_down), abs(current_down_new)), this_height)
            current_down = current_down_new
            gen_number += 1
        if return_area:
            return score, all_area, area
        return score

    def compute_score_for_strip(self, bigger_abs_distance, height, return_area=False):
        smaller_width = 2*math.sqrt(self.r**2 - bigger_abs_distance**2)
        if return_area:
            area = smaller_width
        score = 0
        for rectangle in self.rectangle_container[height]:
            score += rectangle[0] * rectangle[1] * math.floor(smaller_width/rectangle[1])
            smaller_width -= math.floor(smaller_width/rectangle[1]) * rectangle[1]
        if return_area:
            return score, area * height, (area-smaller_width) * height
        return score

    def __learn_population_epoch__(self):
        indexes = np.argsort(self.evaluate())
        self.cross_population(self.discrete[indexes[:int(self.n/2)], :], self.continous[indexes[:int(self.n/2)], :])
        self.mutate_population()

    def learn_population(self, epochs=10):
        for i in range(epochs):
            self.__learn_population_epoch__()

    @staticmethod
    def read_data(file_directory, file_name):
        data = pd.read_csv(file_directory + file_name, header=None)
        rectangles = []
        unique_sizes = set()
        # the normalization factor of scores from different datasets
        best_rectangle_score_per_unit = 0
        for i in range(data.shape[0]):
            rectangles.append([data.iloc[i, 0], data.iloc[i, 1], data.iloc[i, 2]])
            new_best = data.iloc[i, 2] / (data.iloc[i, 0] * data.iloc[i, 1])
            if new_best > best_rectangle_score_per_unit:
                best_rectangle_score_per_unit = new_best
            unique_sizes.add(data.iloc[i, 0])
            unique_sizes.add(data.iloc[i, 1])
        unique_sizes = sorted(list(unique_sizes))
        return rectangles, unique_sizes, best_rectangle_score_per_unit
