import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


def crossover(parent1, parent2):
    start, end = sorted(np.random.choice(range(1, len(parent1)), size=2, replace=False))
    window1 = parent1[start:end]
    window2 = parent2[start:end]
    offspring1 = []
    offspring2 = []

    for i in range(len(parent1)):
        if parent2[i] not in window1:
            offspring1.append(parent2[i])
        if parent1[i] not in window2:
            offspring2.append(parent1[i])

    for i, index in enumerate(range(start, end)):
        offspring1.insert(index, window1[i])
        offspring2.insert(index, window2[i])

    return np.array(offspring1), np.array(offspring2)


def mutate(sequence):
    pos1, pos2 = np.random.choice(range(len(sequence)), size=2, replace=False)

    sequence[pos1], sequence[pos2] = sequence[pos2], sequence[pos1]


def local_search(locations, route):
    best_fitness = get_fitness(locations, route)

    for i in range(len(route)):
        for k in range(i + 1, len(route)):
            new_route = two_opt(route, i, k)
            new_fitness = get_fitness(locations, new_route)

            if new_fitness > best_fitness:
                route = new_route
                best_fitness = new_fitness

    return route


def two_opt(route, i, k):
    new_route = []

    new_route.extend(route[:i])
    new_route.extend(np.flip(route[i:k]))
    new_route.extend(route[k:])

    return np.array(new_route)


def get_fitness(locations, order):
    total_distance = 0
    first_try = False
    for loc in order:
        if first_try:
            total_distance += ((locations[0][loc] - locations[0][prev]) ** 2 + (
                        locations[1][loc] - locations[1][prev]) ** 2) ** 0.5
        prev = loc
        first_try = True
    return 1 / total_distance


def plot_fitness(locations, order):
    plt.scatter(locations[0], locations[1])
    total_distance = 0
    first_try = False
    for loc in order:
        if first_try:
            total_distance += ((locations[0][loc] - locations[0][prev])**2 + (locations[1][loc] - locations[1][prev])**2)**0.5
            plt.plot([locations[0][prev], locations[0][loc]], [locations[1][prev], locations[1][loc]])
        prev = loc
        first_try = True
    return 1 / total_distance


def read_coordinates(path):
    coordinates = []

    with open(path, 'r') as input_handle:
        for line in input_handle.readlines():
            line = line.rstrip().split()
            line[0] = float(line[0])
            line[1] = float(line[1])
            coordinates.append(line)

    return np.swapaxes(np.array(coordinates), 0, 1)


def evolutionary_algorithm(iterations=1500, localsearch=True, filepath='file-tsp.txt'):
    locations = read_coordinates(filepath)
    amount_cities = locations.shape[1]
    order = np.random.permutation(amount_cities)
    order2 = np.random.permutation(amount_cities)

    fitness1 = get_fitness(locations, order)
    fitness2 = get_fitness(locations, order2)
    best_fitness = [max(fitness1, fitness2)]
    average_fitness = [np.average([fitness1, fitness2])]

    for i in range(iterations):
        new_order, new_order2 = crossover(order, order2)
        mutate(new_order)
        mutate(new_order2)
        if localsearch:
            new_order = local_search(locations, new_order)
            new_order2 = local_search(locations, new_order2)
        new_fitness1 = get_fitness(locations, new_order)
        new_fitness2 = get_fitness(locations, new_order2)

        if new_fitness1 > fitness1:
            fitness1 = new_fitness1
            order = new_order
        if new_fitness2 > fitness2:
            fitness2 = new_fitness2
            order2 = new_order

        best_fitness.append(max(fitness1, fitness2))
        average_fitness.append(np.average([fitness1, fitness2]))

    return best_fitness, average_fitness


def main(repeats=10, iterations=10):
    best_fitness_memetic_txt = []
    average_fitness_memetic_txt = []
    best_fitness_simple_txt = []
    average_fitness_simple_txt = []
    best_fitness_memetic_berlin = []
    average_fitness_memetic_berlin = []
    best_fitness_simple_berlin = []
    average_fitness_simple_berlin = []

    with Pool() as pool:
        args = []
        for i in range(repeats):
            args.append((iterations, True, 'file-tsp.txt'))
        for i in range(repeats):
            args.append((iterations, False, 'file-tsp.txt'))
        for i in range(repeats):
            args.append((iterations, True, 'berlin.txt'))
        for i in range(repeats):
            args.append((iterations, False, 'berlin.txt'))
        result = pool.starmap(evolutionary_algorithm, args)

    """
    for i in range(repeats):
        result = evolutionary_algorithm(iterations)
        best_fitness_memetic_txt.append(result[0])
        average_fitness_memetic_txt.append(result[1])

        result = evolutionary_algorithm(iterations, localsearch=False)
        best_fitness_simple_txt.append(result[0])
        average_fitness_simple_txt.append(result[1])

        result = evolutionary_algorithm(iterations, filepath='berlin.txt')
        best_fitness_memetic_berlin.append(result[0])
        average_fitness_memetic_berlin.append(result[1])

        result = evolutionary_algorithm(iterations, localsearch=False, filepath='berlin.txt')
        best_fitness_simple_berlin.append(result[0])
        average_fitness_simple_berlin.append(result[1])
    """
    for i in range(0, 10):
        best_fitness_memetic_txt.append(result[i][0])
        average_fitness_memetic_txt.append(result[i][1])
    for i in range(10, 20):
        best_fitness_simple_txt.append(result[i][0])
        average_fitness_simple_txt.append(result[i][1])
    for i in range(20, 30):
        best_fitness_memetic_berlin.append(result[i][0])
        average_fitness_memetic_berlin.append(result[i][1])
    for i in range(30, 40):
        best_fitness_simple_berlin.append(result[i][0])
        average_fitness_simple_berlin.append(result[i][1])

    plot_results('memetic',
                 'file-tsp.txt',
                 best_fitness_memetic_txt,
                 average_fitness_memetic_txt,
                 iterations=iterations)
    plot_results('simple ea',
                 'file-tsp.txt',
                 best_fitness_simple_txt,
                 average_fitness_simple_txt,
                 iterations=iterations)
    plot_results('memetic',
                 'berlin',
                 best_fitness_memetic_berlin,
                 average_fitness_memetic_berlin,
                 iterations=iterations)
    plot_results('simple ea',
                 'berlin',
                 best_fitness_simple_berlin,
                 average_fitness_simple_berlin,
                 iterations=iterations)


def plot_results(algorithm, problem, best_fitness, average_fitness, iterations=1500):
    axis = [i for i in range(iterations + 1)]

    for fitness in best_fitness:
        plt.plot(axis, fitness)
    plt.title(f'Best fitness for the {algorithm} algorithm on {problem}')
    plt.xlabel('Iteration #')
    plt.ylabel('Best fitness')
    plt.tight_layout()
    plt.show()

    for fitness in average_fitness:
        plt.plot(axis, fitness)
    plt.title(f'Avg fitness for the {algorithm} algorithm on {problem}')
    plt.xlabel('Iteration #')
    plt.ylabel('Average fitness')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main(iterations=1500)
