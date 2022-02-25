import numpy as np


def calculate_fitness(particles):
    return sum([(-x)*np.sin(abs(x)**0.5)for x in particles])


def update_particle(particle_positions: list, social_best: list, personal_best: list, weight: float, velocities: list, r: float = 0.5, a: float = 1):
    update_locations = [particle + (weight * velocities[num] + a*r*(personal_best[num] - particle) + a*r*(social_best[num] - particle)) for num, particle in enumerate(particle_positions)]
    for num, location in enumerate(update_locations):
        if location > 500:
            update_locations[num] = 500
        elif location < -500:
            update_locations[num] = -500
    return update_locations

particles = [[-400, -400], [-410, -410], [-415, -415]]

print("Initial positions and fitnesses are:")
print(particles)
print([calculate_fitness(i) for i in particles])


tmp_weight = 2
print(f"\n\nPositions and fitnisses for {tmp_weight}")
new_positions = [update_particle(particle_positions=sin_part, social_best=[-415, -415], personal_best=sin_part, weight=tmp_weight, velocities=[-50, -50]) for sin_part in particles]
print(new_positions)
print([calculate_fitness(i) for i in new_positions])

tmp_weight = 0.5
print(f"\n\nPositions and fitnisses for {tmp_weight}")
new_positions = [update_particle(particle_positions=sin_part, social_best=[-415, -415], personal_best=sin_part, weight=tmp_weight, velocities=[-50, -50]) for sin_part in particles]
print(new_positions)
print([calculate_fitness(i) for i in new_positions])

tmp_weight = 0.1
print(f"\n\nPositions and fitnisses for {tmp_weight}")
new_positions = [update_particle(particle_positions=sin_part, social_best=[-415, -415], personal_best=sin_part, weight=tmp_weight, velocities=[-50, -50]) for sin_part in particles]
print(new_positions)
print([calculate_fitness(i) for i in new_positions])
