from main1_2 import extract_organ_features, current_accuracy, initialization, save_individual, load_individual, evaluate, manual_check_subtrees, transform_data, evaluate_nodes, inject_new_individuals
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


def main():
    organ = "Lung"
    train_features = extract_organ_features("train_lungandkidney.xlsx", "AllKimiaPatches", organ)
    val_features = extract_organ_features("validate_lungandkidney.xlsx", "AllKimiaPatches", organ)
    test_features = extract_organ_features("test_lungandkidney.xlsx", "AllKimiaPatches", organ)

    accuracy = current_accuracy(train_features, test_features)
    print(f'Initial Accuracy on Raw Features: {accuracy}')
    
    # Initialize population and toolbox
    population, toolbox, pset = initialization(train_features, val_features, test_features)
    print("Done initializing population.")
    
    best_fitness_per_gen = []

    # Define statistics object to track the highest fitness (accuracy)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)  # Track best accuracy each generation

    # Logbook to store the statistics
    logbook = tools.Logbook()

    # Run the genetic algorithm and track stats
    pop, log = algorithms.eaMuPlusLambda(
        population,
        toolbox,
        mu=100,         # Number of parents
        lambda_=90,     # Number of offspring
        cxpb=0.8,       # Crossover probability
        mutpb=0.1,      # Mutation probability
        ngen=5,        # Number of generations
        stats=stats,    # Track stats
        halloffame=None,
        verbose=True
    )

    # Extract best fitness (max accuracy) from each generation
    best_fitness_per_gen = log.select("max")

    plt.plot(range(len(best_fitness_per_gen)), best_fitness_per_gen, marker='o', linestyle='-', color='b', label="Best Accuracy")
    plt.xlabel("Generation")
    plt.ylabel("Best Accuracy")
    plt.title("Evolution of Best Accuracy Over Generations")
    plt.legend()
    plt.grid()
    plt.show()
    
    # Evaluate the best individual in the final population
    best_ind = population[0]
    #manual_check_subtrees(best_ind, pset)
    #trainfet, trainlab, testfeat, testlab = transform_data(best_ind,train_features, test_features, pset, "test")
    #print(f'Transformed first feature of train set: {trainfet[0]}, {trainlab[0]}')
    #print(f'Raw Data Point: {train_features[0]}')

    best_accuracy = evaluate(best_ind, toolbox, pset, data_type="test")
    #print(f'Accuracy before: {best_accuracy}')
    for i in range(len(population)):
        accuracy_results = evaluate(population[i], toolbox, pset, data_type="test")
        if accuracy_results>best_accuracy:
            best_ind = population[i]
            best_accuracy = accuracy_results
    print(f'{best_ind}')
    print(f'{best_accuracy}')
    #print(f'saved best individual: {}')
    save_individual(best_ind, 11)
    #trainfet, trainlab, testfeat, testlab = transform_data(best_ind,train_features, test_features, pset)
    #print(f'Transformed first feature of train set: {trainfet[0]}, {trainlab[0]}')  
    """
    #print(f'Raw Data Point: {train_features[0]}')
    #print(evaluate(best_ind, toolbox, pset, data_type="test"))
    ind = load_individual(11)
    accuracy = evaluate_nodes(ind, toolbox, pset, data_type="test")
    print(f'Feature Selection: {accuracy}')
    accuracy = evaluate_nodes(ind, toolbox, pset, data_type="validation")
    print(f'Feature Selection Validation: {accuracy}')
    accuracy_results = evaluate(ind, toolbox, pset, data_type="validation")
    print(f'Validation Constructed: {accuracy_results}')
    #manual_check_subtrees(ind, pset)
    #trainfet, trainlab, testfeat, testlab = transform_data(ind,train_features, test_features, pset, "test")
    #print(f'Transformed first feature of train set: {trainfet[0]}, {trainlab[0]}')
    #print(f'Raw Data Point: {train_features[0]}')
    print(evaluate(ind, toolbox, pset, data_type="test"))
    print(f'{ind}')
    #accuracy_results = evaluate(ind, toolbox, pset, data_type="test")
    #print(f'{accuracy_results}')
    #print(f'{ind}')
    #"""
    

main()