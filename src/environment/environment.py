import numpy as np
import random

def plume(scenario, position):
    # Placeholder for the actual plume function
    diffusion_coefficient = 2.0
    concentration = 0.0
    for source in scenario:
        source_position, source_intensity = source
        distance = ((position[0] - source_position[0]) ** 2 + (position[1] - source_position[1]) ** 2) ** 0.5
        concentration += source_intensity * np.exp(-distance / diffusion_coefficient) 

    return concentration


def generate_random_scenario(source_range, domain_size, intensity_range):
    scenario = []
    num_sources = random.randint(source_range[0], source_range[1])
    for _ in range(num_sources):
        source_position = (random.uniform(0, domain_size[0]), random.uniform(0, domain_size[1]))
        source_intensity = random.uniform(intensity_range[0], intensity_range[1])
        scenario.append((source_position, source_intensity))
    return scenario