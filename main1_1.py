from deap import gp, base, creator, tools, algorithms
import random
import pickle
import numpy as np
import os
import pandas as pd
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend

# Cache for compiled subtrees
subtree_cache = {}

def main():
    # Load datasets
    organ = "Lung"
    train_features = extract_organ_features("train_lungandkidney.xlsx", "AllKimiaPatches", organ)
    val_features = extract_organ_features("validate_lungandkidney.xlsx", "AllKimiaPatches", organ)
    test_features = extract_organ_features("test_lungandkidney.xlsx", "AllKimiaPatches", organ)

    # Calculate initial accuracy on raw features
    accuracy = current_accuracy(train_features, test_features)
    print(f'Initial Accuracy on Raw Features: {accuracy}')

    # Initialize population and toolbox
    population, toolbox, pset = initialization(train_features, val_features, test_features)
    print("Done initializing population.")
    """
    # Statistics and logbook
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    logbook = tools.Logbook()
    best_fitness_per_gen = []
    
    # Main evolutionary loop
    print("Starting evolution...")
    for gen in range(15):  # Run for 25 generations
        # Apply crossover and mutation
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.8, mutpb=0.1)

        # Evaluate the offspring
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Select the next generation
        population = toolbox.select(offspring, len(population))

        # Record statistics
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        best_fitness_per_gen.append(record["max"])

        print(f"Generation {gen}: Max Accuracy = {record['max']}")

    # Plot the best accuracy over generations
    plt.plot(range(len(best_fitness_per_gen)), best_fitness_per_gen, marker='o', linestyle='-', color='b', label="Best Accuracy")
    plt.xlabel("Generation")
    plt.ylabel("Best Accuracy")
    plt.title("Evolution of Best Accuracy Over Generations")
    plt.legend()
    plt.grid()
    plt.show()
    """
    indexp = "mul(mul(add(mul(mul(ARG520, ARG182), mul(ARG183, ARG31)), sub(sub(add(ARG967, ARG875), add(ARG25, ARG110)), sub(ARG70, ARG242))), sub(mul(add(ARG714, ARG199), sub(ARG632, ARG577)), sub(sub(ARG796, add(mul(mul(ARG289, ARG678), sub(ARG344, ARG559)), sub(sub(ARG107, add(ARG772, ARG77)), ARG346))), mul(ARG522, ARG556)))), sub(mul(mul(add(mul(ARG235, ARG974), mul(ARG804, ARG426)), add(mul(ARG8, ARG908), mul(ARG687, ARG286))), mul(sub(sub(ARG607, ARG180), sub(ARG945, ARG614)), add(mul(ARG879, ARG979), mul(ARG527, ARG49)))), mul(sub(mul(ARG421, ARG940), add(sub(ARG677, ARG785), ARG743)), mul(add(ARG77, ARG904), add(ARG44, add(ARG970, sub(ARG677, ARG785)))))))"
    ind = gp.PrimitiveTree.from_string(indexp, pset)
    accuracy_results = evaluate(ind, toolbox, pset, data_type="test")
    print(f'{accuracy_results}, {ind}')
    
    # Evaluate the best individual in the final population
    """for i in range(len(population)):
        accuracy_results = evaluate(population[i], toolbox, pset, data_type="test")
        print(f'{accuracy_results}, {population[i]}')
    """
def extract_subtrees(individual, pset):
    """Extract and compile subtrees from an individual."""
    subtrees = []
    for i in range(len(individual)):
        subtree = individual.searchSubtree(i)
        subtree_str = str(subtree)
        if subtree_str not in subtree_cache:
            subtree_cache[subtree_str] = gp.compile(gp.PrimitiveTree(individual[subtree]), pset)
        subtrees.append(subtree_cache[subtree_str])
    return subtrees

def evaluate(individual, toolbox, pset, data_type):
    """
    Evaluate an individual on the specified dataset (validation or test).
    """
    if data_type == "validation":
        train_data = toolbox.train_data
        eval_data = toolbox.validate_data
    elif data_type == "test":
        train_data = toolbox.train_data
        eval_data = toolbox.test_data
    else:
        raise ValueError("data_type must be 'validation' or 'test'")

    # Transform data using the individual's feature transformation
    train_features, train_labels, eval_features, eval_labels = transform_data(individual, train_data, eval_data, pset)

    if len(train_features) > 0 and len(eval_features) > 0:
        # Train a new KNN model on the transformed training data
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(train_features, train_labels)
        # Evaluate on the transformed evaluation data
        accuracy = knn.score(eval_features, eval_labels)
        return (accuracy,)  # Return a tuple containing the accuracy
    else:
        print("Error: No valid features to evaluate the individual.")
        return (0.0,)  # Return a tuple containing 0.0

def transform_data(individual, train_data, testVal_data, pset):
    """Transform data using the individual's subtrees (parallelized with joblib)."""
    subtrees = extract_subtrees(individual, pset)

    def process_instance(instance, subtrees):
        """Process a single instance (datapoint) using the subtrees."""
        datapoint = []
        features = instance[:-1]
        for subtree_func in subtrees:
            try:
                feature_value = subtree_func(*features)
                if np.isfinite(feature_value):
                    datapoint.append(feature_value)
            except Exception as e:
                print(f"Error applying subtree: {e}")
        return datapoint, instance[-1]

    # Process training and validation data in parallel
    with parallel_backend('threading', n_jobs=4):
        train_results = Parallel()(delayed(process_instance)(instance, subtrees) for instance in train_data)
        testVal_results = Parallel()(delayed(process_instance)(instance, subtrees) for instance in testVal_data)

    train_features, train_labels = zip(*train_results)
    testVal_features, testVal_labels = zip(*testVal_results)

    return list(train_features), list(train_labels), list(testVal_features), list(testVal_labels)

def initialization(train_data, validate_data, test_data):
    """Initialize the GP population and toolbox."""
    pset = gp.PrimitiveSet("individual", 1024)  # Primitive set with 1024 arguments
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.train_data = train_data
    toolbox.validate_data = validate_data
    toolbox.test_data = test_data
    toolbox.register("expr", gp.genFull, pset=pset, min_=5, max_=7)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate, toolbox=toolbox, pset=pset, data_type="validation")
    toolbox.register("select", tools.selTournament, tournsize=4)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutInsert, pset=pset)

    population = toolbox.population(n=80)  # Initial population size
    return population, toolbox, pset

def current_accuracy(train_data, test_data):
    """Calculate accuracy on raw features using KNN."""
    train_features = [instance[:-1] for instance in train_data]
    train_labels = [instance[-1] for instance in train_data]
    test_features = [instance[:-1] for instance in test_data]
    test_labels = [instance[-1] for instance in test_data]

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_features, train_labels)
    return knn.score(test_features, test_labels)

def extract_organ_features(metadata_file, directory, organ):
    """Extract features and labels for a specific organ."""
    imagesfeatures = []
    data = pd.read_excel(metadata_file)
    organ_data = data[data['primary_site'] == organ]
    labels_dict = pd.Series(organ_data['disease_type'].values, index=organ_data['slide_name']).to_dict()

    for filename in os.listdir(directory):
        slide_id = filename.split('_', 1)[0]
        if slide_id in labels_dict:
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as filecontents:
                data = pickle.load(filecontents)
                arr = [patch for _, patch in data.items() if not np.any(np.isnan(patch))]
                if arr:
                    average_vector = np.mean(arr, axis=0).tolist()
                    average_vector.append(labels_dict[slide_id])
                    imagesfeatures.append(average_vector)
    return imagesfeatures

if __name__ == "__main__":
    main()