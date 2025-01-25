from deap import gp, base, creator, tools, algorithms
import random
import pickle
import numpy as np
import os
import pandas as pd
import math
import operator
from sklearn.neighbors import KNeighborsClassifier


def main():
    train_features = dataset("TrainingData", "train_brain.xlsx", data_type='training')
    val_features = dataset("TrainingData", "validation_brain.xlsx", data_type='validation')

    accuracy = current_accuracy(train_features, val_features)
    print(accuracy)
    population, toolbox = initialization(train_features, val_features)
    algorithms.eaMuPlusLambda(
    population,
    toolbox,
    mu=40,         # Number of parents
    lambda_=80,    # Number of offspring
    cxpb=0.5,       # Crossover probability
    mutpb=0.5,      # Mutation probability
    ngen=50,        # Number of generations
    verbose=True
    )
    #print(population)
    for individual in population:
        accuracy, complexity = evaluate(individual, toolbox)
        print(accuracy, complexity)

def initialization(train_data, validate_data):
    skelton = gp.PrimitiveSet("individual", 1024)
    skelton.addPrimitive(operator.add, 2)
    skelton.addPrimitive(operator.sub, 2)
    skelton.addPrimitive(operator.mul, 2)
    skelton.addPrimitive(operator.truediv, 2)

    #gen full is one of three ways to generate the trees
    #expr = gp.genFull(skelton, min_ = 1, max_ = 5)
    #tree = gp.PrimitiveTree(expr)

    # Accuracy ↑, Features ↓, creating a fitness class
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  
    #creating an inidvudal class
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

    #registering the functions to a toolbox, and 
    toolbox = base.Toolbox()
    toolbox.train_data = train_data
    toolbox.validate_data = validate_data
    toolbox.register("expr", gp.genFull, pset=skelton, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=skelton) 
    toolbox.register("evaluate", evaluate, toolbox=toolbox)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=skelton)
    toolbox.register("expr_mut", gp.genGrow, min_=1, max_=2)

    population = toolbox.population(n=100)
    return population, toolbox

def evaluate(individual, toolbox):
    """
    Evaluate the accuracy of a model using KNN classifier.
    
    Parameters:
    - individual: The individual representing the model.
    - toolbox: The DEAP toolbox that contains the training and validation data.

    Returns:
    - accuracy: The accuracy of the model on the validation data.
    - complexity: The complexity of the model (number of nodes in the tree).
    """
    # Access the data directly from the toolbox
    train_data = toolbox.train_data
    validate_data = toolbox.validate_data
    
    # Compile the individual to get the model function
    func = toolbox.compile(expr=individual)
    
    # Prepare the training features and labels
    train_features = []
    train_labels = []
    for instance in train_data:
        features = instance[:-1]
        # Check if func is not None before using it
        if func is not None:
            try:
                # Call func with the features
                train_features.append(func(*features))
                train_labels.append(instance[-1])  # Label
            except Exception as e:
                print(f"Error during feature construction: {e}")
        else:
            print("Error: Function is None. Feature construction skipped.")
    
    # Prepare the validation features and labels
    validate_features = []
    validate_labels = []
    for instance in validate_data:
        features = instance[:-1]
        validate_features.append(func(*features))  # Construct features
        validate_labels.append(instance[-1])  # Label
    
    # Train the KNN classifier on the training data
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit([[f] for f in train_features], train_labels)
    
    # Test the KNN classifier on the validation data
    accuracy = knn.score([[f] for f in validate_features], validate_labels)
    
    # Calculate the complexity (number of nodes in the tree)
    complexity = len(individual)  # Total nodes in the tree
    
    return accuracy, complexity



def dataset(directory, metadata_file, data_type='training'):
    """
    Load dataset features and labels from the specified directory and metadata file.
    
    Parameters:
    - directory (str): The directory where the image data is stored.
    - metadata_file (str): The path to the Excel file with the filenames and labels.
    - data_type (str): Specifies whether the dataset is for 'training' or 'validation'. 
                        Default is 'training'. If 'validation', it uses a different label format.
    
    Returns:
    - list: A list of feature vectors with appended labels.
    """
    imagesfeatures = []
    
    # Load the metadata file
    data = pd.read_excel(metadata_file, usecols=[0, 1])  # loading the excel info
    filename_list = data['Filename'].tolist()  # filenames from the metadata
    labels_dict = pd.Series(data['Label'].values, index=data['Filename']).to_dict()  # label dictionary
    
    # Iterate over files in the specified directory
    for filename in os.listdir(directory):
        check = filename.split('_', 1)  # split the filename to check against the metadata
        
        # If the filename matches the ones in the metadata file
        if check[0] in labels_dict:
            f = os.path.join(directory, filename)  # create full file path
            
            # Open and load the file
            with open(f, 'rb') as filecontents:
                data = pickle.load(filecontents)  # loading the file data
                items = data.items()  # converting the data to list format
                arr = []  # list to hold valid patches
                
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
        else:
            print(f"Filename {filename} not found in {data_type} data.")
    
    return imagesfeatures


def current_accuracy(train_data, validate_data):
    # Extract features and labels for training data
    train_features = [instance[:-1] for instance in train_data]
    train_labels = [instance[-1] for instance in train_data]
    
    # Extract features and labels for validation data
    validate_features = [instance[:-1] for instance in validate_data]
    validate_labels = [instance[-1] for instance in validate_data]

    # Initialize KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)

    # Train the classifier with raw features (no transformation)
    knn.fit(train_features, train_labels)
    
    # Test the classifier on validation data and calculate accuracy
    accuracy = knn.score(validate_features, validate_labels)
    
    return accuracy
    
main()


"""
so each individual is gonna be a formula basically of different 
features and the math preformed on them. each indivudal will be a tree
with the terminal (leaf) nodes being the features and the other nodes (primitives)
will be math operations (+,-,/,* for now). 

population size can be 100 maybe? 

and we're gonna take the individual (that is the formula) and apply
it on every single data point in the training and validation set

use it to train the knn model, and then test the accuracy on validation set
"""
