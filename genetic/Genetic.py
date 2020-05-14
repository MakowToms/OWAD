import numpy as np
import math

'''
The base genetic algorithm
With crossing population in one point
With gauss mutation
Evaluation by eval_function
And choosing best 50% of population to next population
Old population is left
New population is made iteratively by choosing to random parents and random cross point
Possibility to change any of above points or add more methods
'''
class Genetic:
    def __init__(self, n, n_gens, eval_function, mutation_coef=0.1):
        self.n = n
        self.n_gens = n_gens
        self.population = self.generate_base_population()
        self.eval_function = eval_function
        self.mutation_coef = mutation_coef

    def generate_base_population(self):
        return np.random.normal(0, 1, [self.n, self.n_gens])

    def mutate_population(self, mutation_type="gauss"):
        if mutation_type=="gauss":
            self.population += self.mutation_coef * self.gauss_mutation()

    def gauss_mutation(self):
        return np.random.normal(0, 1, [self.n, self.n_gens])

    def cross_population(self, best_population):
        n_best = best_population.shape[0]
        population = np.zeros([self.n, self.n_gens])
        population[:n_best, :] = best_population
        for i in range(n_best, self.n):
            population[i, :] = self.create_new_person(best_population)
        self.population = population

    def create_new_person(self, best_population):
        n_best = best_population.shape[0]
        to_cross = np.random.choice(np.linspace(0, n_best-1, n_best, dtype=np.int), 2, replace=False)
        cross_bit = np.random.choice(np.linspace(1, self.n_gens-1, self.n_gens-1, dtype=np.int), 1)[0]
        new = np.zeros([1, self.n_gens])
        new[0, :cross_bit] = best_population[to_cross[0], :cross_bit]
        new[0, cross_bit:] = best_population[to_cross[1], cross_bit:]
        return new

    def evaluate(self):
        return self.eval_function(self.population)

    def __learn_population_epoch__(self):
        indexes = np.argsort(self.evaluate())
        self.cross_population(self.population[indexes[:int(self.n/2)], :])
        self.mutate_population()

    def learn_population(self, epochs=10):
        for i in range(epochs):
            self.__learn_population_epoch__()

