import numpy as np
import random

def plume(scenario, position):
    # Placeholder for the actual plume function
    diffusionCoefficient = 2.0
    concentration = 0.0
    for source in scenario:
        source_position, source_intensity = source
        distance = ((position[0] - source_position[0]) ** 2 + (position[1] - source_position[1]) ** 2) ** 0.5
        concentration += source_intensity * np.exp(-distance / diffusionCoefficient ) 

    return concentration


def generateRandomScenario(sourceRange, domainSize, intensityRange):
    scenario = []
    numSources = random.randint(sourceRange[0], sourceRange[1])
    for _ in range(numSources):
        source_position = (random.uniform(0, domainSize[0]), random.uniform(0, domainSize[1]))
        source_intensity = random.uniform(intensityRange[0], intensityRange[1])
        scenario.append((source_position, source_intensity))
    return scenario