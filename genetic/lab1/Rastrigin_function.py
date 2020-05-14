from genetic.Genetic import Genetic
import numpy as np


# function to evaluate how good is population (each person)
def eval_function(population, n=5, A=10):
    result = np.ones([population.shape[0]]) * A * n
    for i in range(n):
        result += (population[:, i] ** 2) - A * np.cos(2*np.pi*population[:, i])
    return result


# create genetic class
gen = Genetic(1000, 5, eval_function)
# set base population to values with bigger variance around 0
gen.population = 2 * np.random.standard_normal([1000, 5])

# learn 100 epochs and then see results
gen.learn_population(epochs=50)
print("After 100 epochs")
print(f'some values in population: {gen.population}')
print(f'Mean of absolute value in each genom {np.mean(np.abs(gen.population), axis=0)} \n')

# learn 10 epochs more and see results
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
