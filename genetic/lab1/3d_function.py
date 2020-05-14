from genetic.Genetic import Genetic
import numpy as np


# function to evaluate how good is population (each person)
def eval_function(population):
    return (population[:, 0] ** 2) + (population[:, 1] ** 2) + 2 * (population[:, 2] ** 2)


# create genetic class
gen = Genetic(1000, 3, eval_function, mutation_coef=1)
# set base population to quite big values
gen.population = 10 * np.ones([1000, 3])

# learn 100 epochs and then see results
# the population has weights definitely less than 1 in each genom so much less
gen.learn_population(epochs=50)
print("After 50 epochs")
print(f'some values in population: {gen.population}')
print(f'Mean of absolute value in each genom {np.mean(np.abs(gen.population), axis=0)} \n')

# learn 10 epochs more and see results -- nothing change
gen.learn_population(epochs=10)
print("After 10 epochs")
print(f'some values in population: {gen.population}')
print(f'Mean of absolute value in each genom {np.mean(np.abs(gen.population), axis=0)} \n')

# set mutation coefficient to less value (the gauss mutation will be less aggressive)
gen.mutation_coef = 0.05
gen.learn_population(epochs=10)
print("After setting mutation_coef to less (0.5) and 10 epochs")
print(f'some values in population: {gen.population}')
print(f'Mean of absolute value in each genom {np.mean(np.abs(gen.population), axis=0)} \n')

# so the algorithm works well without any big problems
