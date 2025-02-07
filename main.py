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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score





def main():
    #read the data from training and validation
    train_features = dataset("TrainingData", "train_brain.xlsx", data_type='training')
    val_features = dataset("TrainingData", "validation_brain.xlsx", data_type='validation')

    #calculate the accuracy on raw features
    accuracy = current_accuracy(train_features, val_features)
    print(accuracy)
    #initalizae a population
    population, toolbox = initialization(train_features, val_features)

    #the genetic algorithn
    algorithms.eaMuPlusLambda(
    population,
    toolbox,
    mu=100,         # Number of parents
    lambda_=100,    # Number of offspring
    cxpb=0.4,       # Crossover probability
    mutpb=0.6,      # Mutation probability
    ngen=30,        # Number of generations
    verbose=True
    )
    #sub(add(ARG177, add(ARG409, ARG409)), ARG886)
    #add(add(add(add(ARG603, ARG493), add(add(add(ARG788, ARG768), ARG570), ARG920)), ARG920), add(ARG525, ARG493))
    #print(population)
    #add(sub(sub(ARG185, sub(truediv(ARG975, ARG476), add(ARG462, ARG488))), ARG920), ARG462)
    

    #after all generations, print the last population and their accuracy
    for individual in population:
        accuracy = evaluate(individual, toolbox)
        print(accuracy, individual)

    #take one best individual 
    individual = 'sub(sub(add(add(ARG750, ARG739), mul(ARG567, ARG381)), mul(sub(ARG171, ARG671), sub(ARG969, ARG19))), sub(sub(mul(ARG804, ARG900), sub(ARG447, ARG516)), sub(mul(ARG596, ARG664), mul(ARG558, ARG297))))'
    # Evaluate the individual on 3 models
    accuracy_results = evaluate_individual(individual, train_features, val_features, toolbox)
    print("Accuracy Results:")
    print(accuracy_results)

    return





def initialization(train_data, validate_data):
    skelton = gp.PrimitiveSet("individual", 1024) #set the skeleton, so max number is 1024

    #adding functions / prinmitive nodes
    skelton.addPrimitive(operator.add, 2)
    skelton.addPrimitive(operator.sub, 2)
    skelton.addPrimitive(operator.mul, 2)

    #gen full is one of three ways to generate the trees
    #expr = gp.genFull(skelton, min_ = 1, max_ = 5)
    #tree = gp.PrimitiveTree(expr)

    # Accuracy ↑, Features ↓, creating a fitness class (not for now, we are focusing just on optimzation)
    #defining the nane and goal of fitness function
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  

    #creating / defining an inidvudal class and the fitness function for its test
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    #registering the functions to a toolbox, and 
    toolbox = base.Toolbox()
    toolbox.train_data = train_data
    toolbox.validate_data = validate_data
    toolbox.register("expr", gp.genHalfAndHalf, pset=skelton, min_=4, max_=10) #tree creation
    toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.expr) #individual creation
    toolbox.register("population", tools.initRepeat, list, toolbox.individual) #population creation
    toolbox.register("compile", gp.compile, pset=skelton) #compiling trees into expressions
    toolbox.register("evaluate", evaluate, toolbox=toolbox) #sets evaluate as function to evaluates the individuals
    toolbox.register("select", tools.selLexicase) #selection, using multiobjective nsga II algorithm right now
    toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1) #crossover, swaps subtrees at 1 point
    toolbox.register("mutate", gp.mutNodeReplacement, pset=skelton) #mutation, new subtrees (there are other ways too)
    toolbox.register("expr_mut", gp.genGrow, min_=1, max_=3) #new mutations generation

    population = toolbox.population(n=100) #creates inital population of 100

    return population, toolbox


def evaluate(individual, toolbox):

    # Access the data directly from the toolbox (which we set earlier in initalization)
    train_data = toolbox.train_data
    validate_data = toolbox.validate_data
    
    # Compile the individual tree to get the expression function
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
                feature_value = func(*features)
                #print(feature_value)
                if np.isfinite(feature_value):  # Check for NaN or Infinity
                    train_features.append(feature_value)
                    train_labels.append(instance[-1])  # Label
                else:
                    print(f"Warning: Invalid feature value (NaN or Infinity) for instance {instance}")
            
            except Exception as e:
                print(f"Error during feature construction: {e}")
        
        else:
            print("Error: Function is None. Feature construction skipped.")
    

    # Prepare the validation features and labels
    validate_features = []
    validate_labels = []
    for instance in validate_data:
        features = instance[:-1]
        feature_value = func(*features)

        if np.isfinite(feature_value):  # Check for NaN or Infinity
            validate_features.append(feature_value)
            validate_labels.append(instance[-1])  # Label
        else:
            print(f"Warning: Invalid feature value (NaN or Infinity) for validation instance {instance}")
    
    if len(train_features) > 0 and len(validate_features) > 0:
        try:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit([[f] for f in train_features], train_labels)

            # Test the KNN classifier on the validation data
            accuracy = knn.score([[f] for f in validate_features], validate_labels)

            # Return accuracy as a tuple
            return accuracy,
        except ValueError as e:
            print(f"Error during KNN fitting: {e}")
            return 0.0,  # In case of error, return 0 accuracy
    else:
        print("Error: No valid features to train the KNN classifier.")
        return 0.0,  # Return 0 accuracy if there are no valid features


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

def evaluate_individual(individual, train_data, validate_data, toolbox):
    """
    Evaluate an individual formula (GP tree) on three models:
    - KNN (k=3)
    - Logistic Regression
    - Decision Tree
    
    Parameters:
    - individual: The GP tree representing the formula.
    - train_data: Training data (list of feature vectors with labels as the last element).
    - validate_data: Validation data (list of feature vectors with labels as the last element).
    - toolbox: DEAP toolbox containing the compiled function.
    
    Returns:
    - dict: A dictionary containing accuracy rates for KNN, Logistic Regression, and Decision Tree.
    """
    # Compile the individual formula into a callable function
    train_data = toolbox.train_data
    validate_data = toolbox.validate_data
    func = toolbox.compile(expr=individual)
    
    # Extract features and labels from training data
    train_features = []
    train_labels = []
    for instance in train_data:
        features = instance[:-1]  # First 1024 features
        label = instance[-1]      # Last element is the label
        try:
            value = func(*features)  # Apply the formula
            train_features.append([value])  # Wrap value to keep 2D array structure
            train_labels.append(label)
        except Exception as e:
            print(f"Error applying formula to training instance: {e}")
    
    # Extract features and labels from validation data
    validate_features = []
    validate_labels = []
    for instance in validate_data:
        features = instance[:-1]
        label = instance[-1]
        try:
            value = func(*features)
            validate_features.append([value])  # Wrap value to keep 2D array structure
            validate_labels.append(label)
        except Exception as e:
            print(f"Error applying formula to validation instance: {e}")
    
    # Initialize results dictionary
    results = {}

    # Ensure there are valid features and labels
    if len(train_features) > 0 and len(validate_features) > 0:
        # Model 1: KNN (k=3)
        try:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(train_features, train_labels)
            knn_preds = knn.predict(validate_features)
            results['KNN'] = accuracy_score(validate_labels, knn_preds)
        except Exception as e:
            print(f"Error with KNN: {e}")
            results['KNN'] = 0.0

        # Model 2: Logistic Regression
        try:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(train_features, train_labels)
            lr_preds = lr.predict(validate_features)
            results['LogisticRegression'] = accuracy_score(validate_labels, lr_preds)
        except Exception as e:
            print(f"Error with Logistic Regression: {e}")
            results['LogisticRegression'] = 0.0

        # Model 3: Decision Tree
        try:
            dt = DecisionTreeClassifier()
            dt.fit(train_features, train_labels)
            dt_preds = dt.predict(validate_features)
            results['DecisionTree'] = accuracy_score(validate_labels, dt_preds)
        except Exception as e:
            print(f"Error with Decision Tree: {e}")
            results['DecisionTree'] = 0.0
    else:
        print("Error: No valid features for training or validation.")
        results = {'KNN': 0.0, 'LogisticRegression': 0.0, 'DecisionTree': 0.0}
    
    return results


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
