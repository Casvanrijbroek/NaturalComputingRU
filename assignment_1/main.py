from random import random
import matplotlib.pyplot as plt
import numpy as np



def run_evolution(population_size, mutation_rate, amount_iterations):
    fitnesses = []
    population = np.random.randint(2, size=population_size)
    for single_iteration in range(amount_iterations):
        newpop = np.copy(population)
        for num, induvidual in enumerate(population):
            if random() <= mutation_rate:
                newpop[num] = 1 - induvidual

        # if sum(newpop) > sum(population):
        #     population = newpop

        population = newpop


        fitnesses.append(sum(population))
    return population, fitnesses


population_size = 100
mutation_rate = 1/population_size
amount_iterations = 1500


axis = [i for i in range(amount_iterations)]
amount_max = 0

for i in range(10):
    population, fitnesses = run_evolution(population_size, mutation_rate, amount_iterations)
    best_fitnesses = []
    best_fitness = 0
    for fitness in fitnesses:
        if fitness > best_fitness:
            best_fitness = fitness
            best_fitnesses.append(fitness)
        else:
            best_fitnesses.append(best_fitness)
    plt.plot(axis, best_fitnesses, label = "run number "+str(i))
    if fitnesses[-1] == 100:
        amount_max += 1
print(amount_max)
plt.title('fitness over time')
plt.xlabel('Iterations')
plt.ylabel('fitness')
plt.show()
