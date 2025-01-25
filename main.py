from deap import gp, base, creator, tools
import random
import pickle
import numpy as np
import os
import pandas as pd
import math
import operator



Trainingdata = pd.read_excel("train_brain.xlsx", usecols=[0, 1]) #loading excel info
filename_list = Trainingdata['Filename'].tolist() #turning the filenames into a list (to check later)
labels_dict = pd.Series(Trainingdata['Label'].values, index=Trainingdata['Filename']).to_dict() #(turning it into a dictionary)

imagesfeatures = []  # will contain our whole dataset
directory = "TrainingData"  # folder with all pickle files





def main():
    data = dataset()
    population, toolbox = initialization()
    #print(population)
    for individual in population:
        constructed_features = evaluate(individual, data, toolbox)
    print(constructed_features)

def initialization():
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
    toolbox.register("expr", gp.genFull, pset=skelton, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual,toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=skelton)
    population = toolbox.population(n=100)
    return population, toolbox

def evaluate(individual, data, toolbox):
    func = toolbox.compile(expr=individual)
    constructed_features = []
    for instance in data:
        features = instance[:-1]
        print(features)
        print(len(features))
        constructed_features.append(func(*features)) 
    return constructed_features




def dataset():
    for filename in os.listdir(directory): # iterate over each file in the directory
        check = filename.split('_', 1) # split the filename to check against the ones in the excel file
        # if the filename is in the list of filenames from the triaing data list
        if check[0] in labels_dict:
            f = os.path.join(directory, filename) #make the whole path
            with open(f, 'rb') as filecontents: #opening the file
                data = pickle.load(filecontents)  # loading it
                item = data.items()  # turning it into list format
                arr = []  # list to hold all patches info for one image 
                
                for d in item: #going over each patch in one image
                    arr.append(d[1]) # append all feature patches to arr

                average_vector = np.mean(arr, axis=0)  # getting the average of all features in one image
                label = labels_dict[check[0]]  # getting the label for the current filename
                average_vector = np.append(average_vector, label)  # appending the label to the end
                imagesfeatures.append(average_vector) #appending average info for each image into dataset 
        else:
            print("p")
    return imagesfeatures



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
