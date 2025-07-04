import random

# Constants
TARGET = "METHINKS IT IS LIKE A WEASEL"
POPULATION_SIZE = 100
MUTATION_RATE = 0.05
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "

# Scoring function: how many characters match exactly
def score_match(candidate, target):
    return sum(c == t for c, t in zip(candidate, target))

# Random character from A-Z and space
def random_char():
    return random.choice(ALPHABET)

# Generate a random string of the same length as TARGET
def random_string(length):
    return ''.join(random_char() for _ in range(length))

# Mutation function
def mutate_string(parent, mutation_rate=MUTATION_RATE):
    return ''.join(
        random_char() if random.random() < mutation_rate else c
        for c in parent
    )

# Evolution loop
def evolve():
    generation = 0
    best_parent = random_string(len(TARGET))
    best_score = score_match(best_parent, TARGET)

    while best_score < len(TARGET):
        generation += 1
        offspring = [mutate_string(best_parent) for _ in range(POPULATION_SIZE)]
        scored_offspring = [(child, score_match(child, TARGET)) for child in offspring]
        child, child_score = max(scored_offspring, key=lambda x: x[1])

        if child_score > best_score:
            best_parent, best_score = child, child_score

        print(f"Generation {generation}: Score {best_score} | {best_parent}")

# Run it
evolve()
