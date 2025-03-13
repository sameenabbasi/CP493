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
import re

# Cache for compiled subtrees
subtree_cache = {}

def manual_check_subtrees(ind, pset):
    subtrees = []
    subtree_cache = {}  # Define subtree_cache inside function if not global

    for i in range(len(ind)):
        subtree = ind.searchSubtree(i)  # Gets indices of the subtree
        subtree_tree = gp.PrimitiveTree(ind[subtree])  # Create a tree from the subtree
        subtree_str = str(subtree_tree)  # Convert to string for readability

        subtree_cache[subtree_str] = gp.compile(subtree_tree, pset)
        subtrees.append(subtree_tree)  # Append the tree, not just compiled code

        print(f"Subtree {i}: {subtree_str}")  # Print readable representation

    return subtrees

def save_individual(individual, index, directory="saved_individuals"):
    """Saves an individual with a unique filename."""
    os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
    
    if index is None:  # Auto-increment filename if index not given
        index = len(os.listdir(directory))
    
    filename = os.path.join(directory, f"individual_{index}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(individual, f)
    
    print(f"Individual saved as {filename}")

def load_individual(index, directory="saved_individuals"):
    """Loads an individual by index."""
    filename = os.path.join(directory, f"individual_{index}.pkl")
    
    try:
        with open(filename, "rb") as f:
            individual = pickle.load(f)
        print(f"Loaded {filename}")
        return individual
    except FileNotFoundError:
        print(f"Individual {index} not found.")
        return None

#def extract_subtrees(individual, pset):
    subtrees = []
    for i in range(len(individual)):
        subtree = individual.searchSubtree(i)
        subtree_str = str(subtree)
        
        if subtree_str not in subtree_cache:
            compiled_subtree = gp.compile(gp.PrimitiveTree(individual[subtree]), pset)
            subtree_cache[subtree_str] = compiled_subtree
        else:
            compiled_subtree = subtree_cache[subtree_str]
        subtrees.append(compiled_subtree)
    return subtrees  # Return both subtrees and extracted terminal nodes

def extract_leafs(individual, pset):
    terminalnodes = []
    
    # Iterate over all nodes in the individual
    for node in individual:
        # Check if the node is a terminal node
        if isinstance(node, gp.Terminal):
            # Compile the terminal node into a callable function
            compiled_subtree = gp.compile(gp.PrimitiveTree([node]), pset)
            terminalnodes.append(compiled_subtree)
    return terminalnodes

def inject_new_individuals(population, gen, toolbox):
    if gen % 3 == 0 and gen != 0:  
        num_new = 5  
        new_individuals = toolbox.population(n=num_new)  # generate new individuals

        fitnesses = list(toolbox.map(toolbox.evaluate, new_individuals))  
        for ind, fit in zip(new_individuals, fitnesses):
            ind.fitness.values = fit 

        #replace the worst performing individuals in the population with new unique ones
        sorted_population = sorted(population, key=lambda ind: ind.fitness.values[0])
        for i in range(num_new):
            population.remove(sorted_population[i])  # remove worst-performing individuals
            population.append(new_individuals[i])  # add new individuals
    return population

def transform_data_using_terminals(individual, train_data, testVal_data, pset):
    """Transform data using the individual's terminal nodes."""
    terminalnodes = extract_leafs(individual, pset)  # Get the terminal nodes (features)
    def process_instance(instance, terminalnodes):
        """Process a single instance (datapoint) using the subtrees."""
        datapoint = []
        features = instance[:-1]
        for node_func in terminalnodes:
            try:
                feature_value = node_func(*features)
                if np.isfinite(feature_value):
                    datapoint.append(feature_value)
            except Exception as e:
                print(f"Error applying node: {e}")
        return datapoint, instance[-1]

    # Process training and validation data in parallel
    with parallel_backend('threading', n_jobs=4):
        train_results = Parallel()(delayed(process_instance)(instance, terminalnodes) for instance in train_data)
        testVal_results = Parallel()(delayed(process_instance)(instance, terminalnodes) for instance in testVal_data)

    train_features, train_labels = zip(*train_results)
    testVal_features, testVal_labels = zip(*testVal_results)

    return list(train_features), list(train_labels), list(testVal_features), list(testVal_labels)
    
def evaluate_nodes(individual, toolbox, pset, data_type):
    if data_type == "validation":
        train_data = toolbox.train_data
        eval_data = toolbox.validate_data
    elif data_type == "test":
        train_data = toolbox.train_data
        eval_data = toolbox.test_data
    else:
        raise ValueError("data_type must be 'validation' or 'test'")

    # Use the function to transform data based on terminal nodes
    train_features, train_labels, eval_features, eval_labels = transform_data_using_terminals(individual, train_data, eval_data, pset)
    if data_type=="test":
        print(f'Shape: {np.array(train_features).shape}')
    
    if len(train_features) > 0 and len(eval_features) > 0:
        # Initialize KNN classifier
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(train_features, train_labels)  # Train the model on the transformed data
        accuracy = knn.score(eval_features, eval_labels)  # Evaluate accuracy on validation/test set
        return (accuracy,)
    else:
        print("Error: No valid features to evaluate the individual.")
        return (0.0,)

def evaluate(individual, toolbox, pset, data_type):
    if data_type == "validation":
        train_data = toolbox.train_data
        eval_data = toolbox.validate_data
    elif data_type == "test":
        train_data = toolbox.train_data
        eval_data = toolbox.test_data
    else:
        raise ValueError("data_type must be 'validation' or 'test'")

    train_features, train_labels, eval_features, eval_labels = transform_data(individual, train_data, eval_data, pset, data_type)
    #print(f'\nThe Transformed First data Point: {train_features[0]}')
    if data_type=="test":
        print(f'Shape: {np.array(train_features).shape}')
        print(f'Shape: {np.array(eval_features).shape}')
    else:
        print(f'Shape: {np.array(eval_features).shape}')

    if len(train_features) > 0 and len(eval_features) > 0:
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(train_features, train_labels)
        accuracy = knn.score(eval_features, eval_labels)
        return (accuracy,)  
    else:
        print("Error: No valid features to evaluate the individual.")
        return (0.0,) 

def transform_data_original(individual, train_data, testVal_data, pset):
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
def transform_data(individual, train_data, testVal_data, pset, data_type):
    """Transform data using the individual's subtrees, while printing debug information."""
    
    # Extract and compile subtrees from the individual
    subtrees, terminal_nodes = extract_subtrees(individual, pset)

    def process_instance(instance):
        """Process a single instance (datapoint) using the subtrees."""
        features = instance[:-1]  # Extract input features
        transformed_features = []
        #if instance==train_data[0]:
            #print("\nRaw Features:", features)

        for i, (subtree_func, terminals) in enumerate(zip(subtrees, terminal_nodes)):
            try:
                feature_value = subtree_func(*features)  # Apply subtree function
                
                if np.isfinite(feature_value):
                    transformed_features.append(feature_value)
                #if instance==train_data[0] and data_type=="test":
                    #print(f"Subtree {i}: {terminals} -> {feature_value}")  # Print transformation

            except Exception as e:
                print(f"Error applying subtree {i}: {e}")

        return transformed_features, instance[-1]  # Return transformed data & original label

    # Process training and test/validation data
    train_results = [process_instance(instance) for instance in train_data]
    testVal_results = [process_instance(instance) for instance in testVal_data]

    # Unzip results into features and labels
    train_features, train_labels = zip(*train_results)
    testVal_features, testVal_labels = zip(*testVal_results)

    return list(train_features), list(train_labels), list(testVal_features), list(testVal_labels)

def extract_subtrees(individual, pset):
    """Extract subtrees, compile them, and identify terminal nodes."""
    subtrees = []
    terminal_nodes_list = []

    for i in range(len(individual)):
        subtree = individual.searchSubtree(i)
        subtree_str = str(subtree)

        # Extract terminal nodes (features used in this subtree)
        terminal_nodes = set(node.name for node in individual[subtree] if node.arity == 0)
        terminal_nodes_list.append(terminal_nodes)

        # Cache compiled functions to avoid redundant computation
        #if subtree_str not in subtree_cache:
        compiled_subtree = gp.compile(gp.PrimitiveTree(individual[subtree]), pset)
            #subtree_cache[subtree_str] = compiled_subtree
        #else:
            #compiled_subtree = subtree_cache[subtree_str]

        subtrees.append(compiled_subtree)

    return subtrees, terminal_nodes_list  # Return compiled subtrees and their terminal nodes
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
    toolbox.register("expr", gp.genFull, pset=pset, min_=3, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evaluate, toolbox=toolbox, pset=pset, data_type="validation")
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutInsert, pset=pset)

    population = toolbox.population(n=100)  # Initial population size
    return population, toolbox, pset

def current_accuracy(train_data, test_data):
    
    train_features = [instance[:-1] for instance in train_data]
    train_labels = [instance[-1] for instance in train_data]
    test_features = [instance[:-1] for instance in test_data]
    test_labels = [instance[-1] for instance in test_data]

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_features, train_labels)
    return knn.score(test_features, test_labels)

def extract_organ_features(metadata_file, directory, organ):
    imagesfeatures = [] #gonna be the formatted data
    data = pd.read_excel(metadata_file)
    organ_data = data[data['primary_site'] == organ] #get only the right organ data
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

def dataset(directory, metadata_file):
    """
    Load dataset features and labels from the specified directory and metadata file.
    """
    imagesfeatures = []

    # Load the metadata file
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
        #else:
            #print(f"Filename {filename} not found in {data_type} data.")

    return imagesfeatures