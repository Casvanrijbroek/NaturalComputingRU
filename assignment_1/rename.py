from random import randint, choice, shuffle, seed
from numpy.random import rand
import numpy as np
from math import isnan
from copy import deepcopy
from matplotlib import pyplot as plt
from json import dump

starting_population_size = 1000
generations = 50
crossover_prop = 0.7
branch_prob = 0.6
mutation_prop = 0
terminal_set = ["x"]
function_set = ["+", "-", "*", "/", "log", "exp", "sin", "cos"]

np.random.seed(420)
seed(420)

new_list = []
with open("numbers.txt", "r", encoding="utf-8") as new_file:
    test = (new_file.readlines())
    for i in test:
        new_list.append([(i.split(" ")[0]), i.split(" ")[1][:len(i.split(" ")[1]) - 1]])


def get_num_lists(nest):
    num_lists = 0
    for i in nest[1:]:
        num_lists += 1
        if type(i) == list:
            num_lists += get_num_lists(i)
    return num_lists


def get_num_nodes(nest):
    num_lists = 0
    for i in nest[1:]:
        if type(i) == list:
            num_lists += 1
            num_lists += get_num_lists(i)

    return num_lists


def get_random_kids(nest):
    global chosen_kid
    return_list = False
    for i in nest[1:]:
        if chosen_kid == 0:
            return i
        else:
            chosen_kid -= 1
            if type(i) == list:
                return_list = get_random_kids(i)
                if return_list:
                    return return_list
    return return_list


def create_individual(nesting_chance, small_terminal_set, small_function_set):
    in_function = choice(small_function_set)
    if in_function in ["log", "exp", "sin", "cos"]:
        return [in_function, (create_individual(nesting_chance, small_terminal_set,
                                                small_function_set) if rand() < nesting_chance else choice(
            small_terminal_set))]
    else:
        return [in_function,
                (create_individual(nesting_chance, small_terminal_set,
                                   small_function_set) if rand() < nesting_chance else choice(small_terminal_set)),
                (create_individual(nesting_chance, small_terminal_set,
                                   small_function_set) if rand() < nesting_chance else choice(small_terminal_set))
                ]


def get_expression(new_function):
    if new_function[0] in ["log", "exp", "sin", "cos"]:
        return f"np.{new_function[0]}({('X' if new_function[1] == 'x' else get_expression(new_function[1]))})"
    else:
        return f"({('X' if new_function[1] == 'x' else get_expression(new_function[1]))}{new_function[0]}{('X' if new_function[2] == 'x' else get_expression(new_function[2]))})"


def get_fitness(fit_function):
    expression = get_expression(fit_function)
    nega_fit = 0
    for i in new_list:
        tmp_str = f"{expression}"
        tmp_str = tmp_str.replace("X", str(i[0]))
        try:
            test = (eval(tmp_str))
        except ZeroDivisionError as er:
            return -1000000
        if isnan(test):
            return -1000000
        nega_fit -= abs(test - float(i[1]))
    return nega_fit


new_pop = []
for i in range(1000):
    new_pop.append(create_individual(branch_prob, terminal_set, function_set))


def place_branch(full_tree, replace_location, replace_branch):
    for i in full_tree[1:]:
        if replace_location < 1:
            full_tree.remove(i)
            full_tree.append(replace_branch)
            return full_tree, -1
        else:
            replace_location -= 1
            if type(i) == list:
                tmp_list, replace_location = place_branch(full_tree, replace_location, replace_branch)
                if replace_location < 0:
                    return tmp_list, -1


def mutate(population):
    after_mut = []
    shuffle(population)
    while len(population) > 1:
        induvidual1 = population.pop()
        induvidual2 = choice(population)

        if rand() < crossover_prop:
            num_take1 = randint(0, -1 + (get_num_lists(deepcopy(induvidual1))))
            num_take2 = randint(0, -1 + (get_num_lists(deepcopy(induvidual2))))
            global chosen_kid
            chosen_kid = deepcopy(num_take1)
            branch_take1 = get_random_kids(deepcopy(induvidual1))
            chosen_kid = deepcopy(num_take2)
            branch_take2 = get_random_kids(deepcopy(induvidual2))

            kid1, __ = place_branch(deepcopy(induvidual1), num_take1, branch_take2)
            kid2, __ = place_branch(deepcopy(induvidual2), num_take2, branch_take1)

            fitnesses = [get_fitness(induvidual1), get_fitness(kid1), get_fitness(kid2)]
            highest = fitnesses.index(max(fitnesses))
            if highest == 0:
                after_mut.append(induvidual1)
            if highest == 1:
                after_mut.append(kid1)
            if highest == 2:
                after_mut.append(kid2)
        else:
            after_mut.append(induvidual1)
    return after_mut


def get_best_fitness(population):
    all_scores = []
    for i in population:
        all_scores.append(get_fitness(i))
    return max(all_scores)


def get_best_function(population):
    all_scores = []
    for ind in population:
        all_scores.append(get_fitness(ind))
    return population[all_scores.index(max(all_scores))]


scatter_x = []
scatter_y = []
for i in new_list:
    scatter_x.append(float(i[0]))
    scatter_y.append(float(i[1]))

write_this_dict = []

best_of_generations = []
amount_nodes_best = []
amount_nodes_average = []
amount_nodes_most = []
amount_nodes_least = []

for i in range(generations):
    mutated_population = mutate(new_pop)

    tmp_fit_list = []
    for induv in mutated_population:
        tmp_fit_list.append(get_fitness(induv))

    sorted_list = [x for _, x in sorted(zip(tmp_fit_list, mutated_population), key=lambda pair: pair[0])]

    new_sorted_list = sorted_list[int(50 / (i + 1)):]
    best_scores = sorted_list[-int(50 / (i + 1)):]
    mutated_population = new_sorted_list + best_scores

    tmp_fit_list = []
    tmp_num_list = []
    amount_nodes_this_gen = []
    for induv in mutated_population:
        tmp_fit_list.append(get_fitness(induv))
        tmp_num_list.append(get_num_lists(induv))
        amount_nodes_this_gen.append(get_num_lists(induv))

    write_this_dict.append({f"generation{i + 1}": {"fitnesses": tmp_fit_list,
                                                   "num_nodes": tmp_num_list}})

    print(get_best_function(mutated_population))
    print(get_best_fitness(mutated_population))

    amount_nodes_best.append(get_num_lists(get_best_function(mutated_population)))
    amount_nodes_least.append(min(amount_nodes_this_gen))
    amount_nodes_most.append(max(amount_nodes_this_gen))
    amount_nodes_average.append(np.mean(amount_nodes_this_gen))

    best_of_generations.append(get_best_fitness(mutated_population))

    best_expression = get_expression(get_best_function(mutated_population))
    cur_best = []
    for i in new_list:
        tmp_deepcopy = deepcopy(best_expression)
        tmp_str = tmp_deepcopy.replace("X", str(i[0]))
        cur_best.append(eval(tmp_str))
    plt.plot(scatter_x, cur_best)
    plt.scatter(np.asarray(scatter_x), np.asarray(scatter_y))
    plt.show()
    new_pop = deepcopy(mutated_population)

with open("output_file.json", "w") as write_file:
    dump(write_this_dict, write_file)

plt.plot(best_of_generations)
plt.xlabel("Generation")
plt.title("Best fitness of each generation")
# set label name of x-axis title
plt.ylabel("Fitness")
plt.show()

plt.plot(amount_nodes_best, label="Number of nodes for best function")
plt.plot(amount_nodes_least, label="Number of nodes for function with least branches")
plt.plot(amount_nodes_most, label="Number of nodes for function with most branches")
plt.plot(amount_nodes_average, label="Mean amount of branches for each function")
plt.xlabel("Generation")
plt.legend()

plt.title("Amount of nodes")
# set label name of x-axis title
plt.ylabel("Number of branches")
plt.show()
