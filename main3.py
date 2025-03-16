#imports
from deap import gp, base, creator, tools, algorithms
import random
import pickle
import numpy as np
import os
import pandas as pd
import math
import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import cProfile
import pstats
from joblib import Parallel, delayed, parallel_backend
from sklearn.preprocessing import LabelEncoder
import re


subtree_cache = {}

def main():

    organ = "Lung" #get lung dataset loaded
    train_features = extract_organ_features("train_lungandkidney.xlsx", "AllKimiaPatches", organ)
    val_features = extract_organ_features("validate_lungandkidney.xlsx", "AllKimiaPatches", organ)
    test_features = extract_organ_features("test_lungandkidney.xlsx", "AllKimiaPatches", organ)

    accuracy = current_accuracy(train_features, test_features) # calculate the accuracy on raw features
    print(f'Initial Accuracy on Raw Features: {accuracy}')

    population, toolbox, pset = initialization(train_features, val_features, test_features) # Initialize a population
    print("done initalizaing")

    best_fitness_per_gen = []


    stats = tools.Statistics(lambda ind: ind.fitness.values[0]) # Define statistics object to track the highest fitness (accuracy)
    stats.register("max", np.max)  # Track best accuracy each generation

    logbook = tools.Logbook() # Logbook to store the statistics

    print("starting loop")

    for gen in range(25):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.8, mutpb=0.1)
        elites = tools.selBest(population, k=8)  
        offspring = tools.selWorst(offspring, k=len(elites))  
        offspring.extend(elites)  
        population = offspring

        # Evaluate the population
        population = unique_population(population, 40, toolbox, pset)
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        population = inject_new_individuals(population, gen, toolbox, pset)
        # Record statistics
        record = stats.compile(population)
        logbook.record(gen=gen, **record)
        best_fitness_per_gen.append(record["max"])

        print(f"Generation {gen}: Max Accuracy = {record['max']}")
        """if gen == 0:
            first_gen_unique_args = count_unique_args(population)
            first_gen_avg_acc = average_accuracy(population, toolbox, pset)
            print(f"First Generation: Unique Args = {first_gen_unique_args}, Avg Accuracy = {first_gen_avg_acc:.4f}")

    # Last generation unique args and average accuracy
    last_gen_unique_args = count_unique_args(population)
    last_gen_avg_acc = average_accuracy(population, toolbox, pset)
    print(f"Last Generation: Unique Args = {last_gen_unique_args}, Avg Accuracy = {last_gen_avg_acc:.4f}")
    """
        
    plt.plot(range(len(best_fitness_per_gen)), best_fitness_per_gen, marker='o', linestyle='-', color='b', label="Best Accuracy")
    plt.xlabel("Generation")
    plt.ylabel("Best Accuracy")
    plt.title("Evolution of Best Accuracy Over Generations")
    plt.legend()
    plt.grid()
    plt.show()
    """
    best_individual = 'sub(add(sub(add(sub(ARG1017, ARG695), mul(ARG636, ARG705)), add(add(ARG140, ARG550), ARG189)), add(sub(add(ARG728, mul(ARG485, ARG485)), sub(ARG599, sub(ARG801, ARG712))), ARG728)), add(ARG591, sub(add(sub(mul(sub(ARG130, add(sub(add(ARG676, ARG816), ARG710), sub(ARG649, ARG797))), ARG988), sub(ARG550, ARG145)), sub(add(ARG769, ARG550), mul(ARG1020, add(ARG539, ARG728)))), mul(mul(ARG705, ARG198), mul(sub(ARG202, ARG303), ARG269)))))'
    individual = gp.PrimitiveTree.from_string(best_individual, pset)
    print(individual)
    best_individual.fitness.values = toolbox.evaluate(best_individual)
    accuracy_results = evaluate_individual(individual, toolbox, pset)
    print(accuracy_results, individual)
    # After all generations, print the last population and their accuracy"""
    #extract_subtrees(individual, pset)
    for i in range(39):
        individual = population[i]
        # Evaluate the individual on 3 models
        accuracy_results = evaluate_individual(individual, toolbox, pset)
        print(accuracy_results, individual)
    return

def extract_subtrees(individual, pset):
    subtrees = []
    for i in range(len(individual)):
        subtree = individual.searchSubtree(i)
        subtree_str = str(subtree)

        # Check if the subtree has been cached already
        if subtree_str not in subtree_cache:
            subtree_cache[subtree_str] = gp.compile(gp.PrimitiveTree(individual[subtree]), pset)
        
        # Add the compiled subtree only if it's not already in subtrees
        if subtree_cache[subtree_str] not in subtrees:
            subtrees.append(subtree_cache[subtree_str])
        
    return subtrees

def evaluate(individual, toolbox, pset):
    # Use the parallelized transform_data function
    train_features, train_labels, validate_features, validate_labels = transform_data(individual, toolbox.train_data, toolbox.validate_data,  pset)

    if len(train_features) > 0 and len(validate_features) > 0:
        try:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(train_features, train_labels)
            accuracy = knn.score(validate_features, validate_labels)
            return accuracy,  # return the accuracy as a tuple (required by DEAP)
        except ValueError as e:
            print(f"Error during KNN fitting: {e}")
            return 0.0,  # in case of error, return 0 accuracy
    else:
        print("Error: No valid features to train the KNN classifier.")
        return 0.0,  # return 0 accuracy if there are no valid features
def transform_data(individual, train_data, testVal_data, pset):

    subtrees = extract_subtrees(individual, pset)  # Extract subtree functions

    def process_instance(instance, subtrees):
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

    # Process training data in parallel
    with parallel_backend('threading'):  # Use threading backend
        train_results = Parallel(n_jobs=4)(delayed(process_instance)(instance, subtrees) for instance in train_data)
    train_features, train_labels = zip(*train_results)

    # Process validation data in parallel
    with parallel_backend('threading'):  # Use threading backend
        testVal_results = Parallel(n_jobs=4)(delayed(process_instance)(instance, subtrees) for instance in testVal_data)
    testVal_features, testVal_labels = zip(*testVal_results)

    return list(train_features), list(train_labels), list(testVal_features), list(testVal_labels)

def inject_new_individuals(population, gen, toolbox, pset):
    if gen % 3 == 0 and gen != 0:  # Every 3 generations (excluding generation 0)
        num_new = 4  # Number of new individuals to inject
        new_individuals = toolbox.population(n=num_new)  # Generate new individuals
        # Replace the worst performing individuals in the population with new unique ones
        sorted_population = sorted(population, key=lambda ind: ind.fitness.values[0])
        for i in range(num_new):
            ind = new_individuals[i]
            population.remove(sorted_population[i])  # Remove worst-performing individuals
            population.append(ind)  # Add new individuals
        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
    return population

def unique_population(population, n, toolbox, pset):
    #Ensure the population has `n` unique individuals.
    #Preserves the fitness values of existing individuals.
    unique_set = set()
    new_population = []
    # Add existing individuals to the new population if they are unique
    for ind in population:
        ind_str = str(ind)
        if ind_str not in unique_set:
            unique_set.add(ind_str)
            new_population.append(ind)

    # If the population is still smaller than `n`, generate new unique individuals
    while len(new_population) < n:
        ind = toolbox.individual()
        ind_str = str(ind)
        if ind_str not in unique_set:
            unique_set.add(ind_str)
            new_population.append(ind)
    return new_population

def initialization(train_data, validate_data, test_data):
    skeleton = gp.PrimitiveSet("individual", 1024)  # Set the skeleton, so max number is 1024

    # Adding functions / primitive nodes
    skeleton.addPrimitive(operator.add, 2)
    skeleton.addPrimitive(operator.sub, 2)
    skeleton.addPrimitive(operator.mul, 2)
    #skeleton.addPrimitive(math.log, 1)
    #skeleton.addPrimitive(operator.truediv, 2)
    # Define fitness and individual classes
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # Register functions to a toolbox
    toolbox = base.Toolbox()
    toolbox.train_data = train_data
    toolbox.validate_data = validate_data
    toolbox.test_data = test_data
    toolbox.register("expr", gp.genGrow, pset=skeleton, min_=5, max_=7)  # Tree creation
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)  # Individual creation
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) #population creation
    toolbox.register("compile", gp.compile, pset=skeleton)  # Compile trees into expressions
    toolbox.register("evaluate", evaluate, toolbox=toolbox, pset=skeleton)  # Evaluate individuals
    toolbox.register("select", tools.selTournament, tournsize=4)
    toolbox.register("mate", gp.cxOnePoint)  # Crossover
    toolbox.register("mutate", gp.mutInsert, pset=skeleton)  # Mutation
    toolbox.register("expr_mut", gp.genGrow, min_=1, max_=2)  # New mutations generation

    population = toolbox.population(n=40)  # Create initial population of 100 unique individuals

    return population, toolbox, skeleton

def current_accuracy(train_data, test_data):
    # Extract features and labels for training data
    train_features = [instance[:-1] for instance in train_data]
    train_labels = [instance[-1] for instance in train_data]

    # Extract features and labels for validation data
    test_features = [instance[:-1] for instance in test_data]
    test_labels = [instance[-1] for instance in test_data]

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the classifier with raw features (no transformation)
    knn.fit(train_features, train_labels)

    # Test the classifier on test data and calculate accuracy
    accuracy = knn.score(test_features, test_labels)

    return accuracy

def extract_organ_features(metadata_file, directory, organ):
    imagesfeatures = []

    # Load the metadata file and filter for lung cases
    data = pd.read_excel(metadata_file)
    lung_data = data[data['primary_site'] == organ]

    # Create a dictionary mapping slide names to disease types
    labels_dict = pd.Series(lung_data['disease_type'].values, index=lung_data['slide_name']).to_dict()

    # Iterate over files in the specified directory
    for filename in os.listdir(directory):
        slide_id = filename.split('_', 1)[0]  # Extract slide name

        if slide_id in labels_dict:
            file_path = os.path.join(directory, filename)

            # Open and load the .pickle file
            with open(file_path, 'rb') as filecontents:
                data = pickle.load(filecontents)
                arr = []

                for _, patch in data.items():
                    if not np.any(np.isnan(patch)):  # Check for NaNs
                        arr.append(patch)

                if arr:
                    average_vector = np.mean(arr, axis=0).tolist()  # Convert NumPy array to list
                    average_vector.append(labels_dict[slide_id])  # Append label
                    imagesfeatures.append(average_vector)

    return imagesfeatures

def dataset_brain(directory, metadata_file):

    imagesfeatures = []

    # Load the metadata 
    data = pd.read_excel(metadata_file, usecols=[0, 1])  # Loading the excel info
    filename_list = data['Filename'].tolist()  # Filenames from the metadata
    labels_dict = pd.Series(data['Label'].values, index=data['Filename']).to_dict()  # Label dictionary

    # Iterate over files in the specified directory
    for filename in os.listdir(directory):
        check = filename.split('_', 1)  # Split the filename to check against the metadata

        # If the filename matches the ones in the metadata file
        if check[0] in labels_dict:
            f = os.path.join(directory, filename)  # Create full file path

            # Open and load the file
            with open(f, 'rb') as filecontents:
                data = pickle.load(filecontents)  # Loading the file data
                items = data.items()  # Converting the data to list format
                arr = []  # List to hold valid patches

                for d in items:
                    patch = d[1]
                    if not np.any(np.isnan(patch)):  # Check for NaNs in the patch
                        arr.append(patch)  # Add valid patch
                    else:
                        print(f"Skipping patch with NaN values: {patch}")

                # Calculate the average if there are valid patches
                if arr:
                    average_vector = np.mean(arr, axis=0)
                else:
                    print(f"Skipping empty patch list for file {filename}")
                    continue  # Skip file if no valid patches

                # Get the label for the current file
                label = labels_dict[check[0]]
                average_vector = np.append(average_vector, label)  # Append label to the average vector
                imagesfeatures.append(average_vector)  # Append the feature vector to the list

    return imagesfeatures

def evaluate_individual(individual, toolbox, pset):
    #Evaluate an individual formula (GP tree)

    train_features, train_labels, test_features, test_labels = transform_data(individual, toolbox.train_data, toolbox.test_data, pset)

    # Print dataset shapes
    print(f"Train features shape: {np.shape(train_features)}")
    print(f"Test features shape: {np.shape(test_features)}")

    # Initialize results dictionary
    results = {}

    # Ensure there are valid features and labels
    if len(train_features) > 0 and len(test_features) > 0:
            try:
                knn = KNeighborsClassifier(n_neighbors=3)
                knn.fit(train_features, train_labels)
                results['KNN'] = knn.score(test_features, test_labels) 
            except ValueError as e:
                print(f"Error during KNN fitting: {e}")
                return 0.0,  # in case of error, return 0 accuracy
    else:
        print("Error: No valid features to train the KNN classifier.")
        return 0.0,  # return 0 accuracy if there are no valid features

        # Model 2: Logistic Regression
        """try:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(train_features, train_labels)
            lr_preds = lr.predict(test_features)  # Use test data for predictions
            results['Logistic Regression'] = accuracy_score(test_labels, lr_preds)
        except Exception as e:
            print(f"Error with Logistic Regression: {e}")
            results['Logistic Regression'] = 0.0

        # Model 3: Decision Tree
        try:
            dt = DecisionTreeClassifier()
            dt.fit(train_features, train_labels)
            dt_preds = dt.predict(test_features)  # Use test data for predictions
            results['Decision Tree'] = accuracy_score(test_labels, dt_preds)
        except Exception as e:
            print(f"Error with Decision Tree: {e}")
            results['Decision Tree'] = 0.0"""

    return results


"""
#for checking code bottlenecks etc
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()  # Call your main function

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats("cumulative").print_stats(10)  # Show top 20 slowest functions
"""
main()