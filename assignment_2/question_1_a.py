from matplotlib import pyplot as plt

def calculate_fitness(x):
    return x**2

def update_velocity(velocity, particle, weight, local_best, global_best, a, r):
    return weight*velocity + a*r*(local_best-particle) + a*r*(global_best-particle)

def update_particle(x, weight, velocity, local_best, global_best, a, r):
    new_velocity = update_velocity(velocity, x, weight, local_best, global_best, a, r)
    new_location = new_velocity + x
    if calculate_fitness(new_location) < calculate_fitness(local_best):
        local_best = new_location
        global_best = new_location
    return new_location, new_velocity, local_best, global_best

def run_experiement(starting_location, starting_velocity, weight, a, r, maximum_operation = 100):
    all_locations = [starting_location]
    global_best = starting_location
    local_best = starting_location
    current_location = starting_location
    current_velocity = starting_velocity
    while current_velocity != 0 and maximum_operation > 0:
        current_location, current_velocity, local_best, global_best = update_particle(current_location, weight, current_velocity, local_best, global_best, a, r)
        all_locations.append(current_location)
        print(str(current_location) + "\t" + str(current_velocity))
        maximum_operation -= 1
    plt.plot(all_locations)
    plt.xlabel("Time by generation")
    plt.ylabel("Position x (global optimum = 0)")
    plt.show()

def main():
    run_experiement(20, 10, 0.5, 1.5, 0.5)

    run_experiement(20, 10, 0.7, 1.5, 1)


main()