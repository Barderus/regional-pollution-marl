# Simple policies with the same local information as IQL
import random

def random_action(observation, rng):
    return rng.randrange(3)

def fixed_medium_action(observation):
    return 1

def threshold_action(observation):
    # Bin 1 includes 15 <= x < 25; bin 2 includes 25 <= x < 35.
    return (0, 0, 1, 2)[observation[1]]
