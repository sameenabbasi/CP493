from main1_2 import extract_organ_features, current_accuracy, initialization, save_individual, load_individual, evaluate, manual_check_subtrees, transform_data, evaluate_nodes, inject_new_individuals, dataset, extract_features_bylabel
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


def run_experiments():
    #train_features = dataset("AllDensePatches", "train_brain.xlsx")
    #val_features = dataset("AllDensePatches", "validation_brain.xlsx")
    #test_features = dataset("AllDensePatches", "test_brain.xlsx")

    #organ = "Lung" #get lung dataset loaded
    #disease_types = ["Cholangiocarcinoma", "Liver Hepatocellular Carcinoma", "Pancreatic Adenocarcinoma"]
    #train_features = extract_features_bylabel("kimiaNet_train_data.xlsx", "AllKimiaPatches", disease_types)
    #val_features = extract_features_bylabel("kimiaNet_validation_data.xlsx", "AllKimiaPatches", disease_types)
    #test_features = extract_features_bylabel("kimiaNet_test_data.xlsx", "AllKimiaPatches", disease_types)

    organ = "Kidney" #get lung dataset loaded
    train_features = extract_organ_features("kimiaNet_train_data.xlsx", "AllDensePatches", organ)
    val_features = extract_organ_features("kimiaNet_validation_data.xlsx", "AllDensePatches", organ)
    test_features = extract_organ_features("kimiaNet_test_data.xlsx", "AllDensePatches", organ)

    val_accuracies = []
    test_accuracies = []
    nodes_only_val_accuracies = []
    nodes_only_test_accuracies = []
    #nodes_only_shapes = []
    #normal_shapes = []
    print(f'Current Accuracy Validation: {current_accuracy(train_features, val_features)}')
    print(f'Current Accuracy Test: {current_accuracy(train_features, test_features)}')
    
    for i in range(10):
        print("here 0")
        population, toolbox, pset = initialization(train_features, val_features, test_features)
        best_fitness_per_gen = []
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("max", np.max)
        logbook = tools.Logbook()
        
        pop, log = algorithms.eaMuPlusLambda(
            population,
            toolbox,
            mu=100,
            lambda_=90,
            cxpb=0.8,
            mutpb=0,
            ngen=1,
            stats=stats,
            halloffame=None,
            verbose=True
        )
        best_fitness_per_gen = log.select("max")
        """
        plt.plot(range(len(best_fitness_per_gen)), best_fitness_per_gen, marker='o', linestyle='-', color='b', label="Best Accuracy")
        plt.xlabel("Generation")
        plt.ylabel("Best Accuracy")
        plt.title("Evolution of Best Accuracy Over Generations")
        plt.legend()
        plt.grid()
        plt.show()"""
        
        best_ind = population[0]
        best_accuracy = evaluate(best_ind, toolbox, pset, data_type="test")

        for i in range(len(population)):
            accuracy_results = evaluate(population[i], toolbox, pset, data_type="test")
            if accuracy_results>best_accuracy:
                best_ind = population[i]
                best_accuracy = accuracy_results
        save_individual(best_ind, i+10)
        val_accuracy = evaluate(best_ind, toolbox, pset, data_type="validation")
        test_accuracy = evaluate(best_ind, toolbox, pset, data_type="test")
        nodes_val_accuracy = evaluate_nodes(best_ind, toolbox, pset, data_type="validation")
        nodes_test_accuracy = evaluate_nodes(best_ind, toolbox, pset, data_type="test")      
        
        val_accuracies.append(val_accuracy)
        test_accuracies.append(test_accuracy)
        nodes_only_val_accuracies.append(nodes_val_accuracy)
        nodes_only_test_accuracies.append(nodes_test_accuracy)
        #nodes_only_shapes.append(transformed_nodes_only.shape)
        #normal_shapes.append(transformed_test.shape)
    
    # Compute averages
    avg_val_accuracy = np.mean(val_accuracies)
    avg_test_accuracy = np.mean(test_accuracies)
    avg_nodes_val_accuracy = np.mean(nodes_only_val_accuracies)
    avg_nodes_test_accuracy = np.mean(nodes_only_test_accuracies)
    #avg_nodes_shape = np.mean(nodes_only_shapes, axis=0)
    #avg_normal_shape = np.mean(normal_shapes, axis=0)
    
    print(f"Average Validation Accuracy: {avg_val_accuracy:.4f}")
    print(f"Average Test Accuracy: {avg_test_accuracy:.4f}")
    print(f"Nodes Only Validation Accuracy: {avg_nodes_val_accuracy:.4f}")
    print(f"Nodes Only Test Accuracy: {avg_nodes_test_accuracy:.4f}")
    #print(f"Average Shape of Nodes Only: {avg_nodes_shape}")
    #print(f"Average Shape of Normal Testing: {avg_normal_shape}")
    
def run_once():
    organ = "Lung" #get lung dataset loaded
    train_features = extract_organ_features("train_lungandkidney.xlsx", "AllDensePatches", organ)
    val_features = extract_organ_features("validate_lungandkidney.xlsx", "AllDensePatches", organ)
    test_features = extract_organ_features("test_lungandkidney.xlsx", "AllDensePatches", organ)

    val_accuracies = []
    test_accuracies = []
    nodes_only_val_accuracies = []
    nodes_only_test_accuracies = []
    #nodes_only_shapes = []
    #normal_shapes = []
    print(f'Current Accuracy Validation: {current_accuracy(train_features, val_features)}')
    print(f'Current Accuracy Test: {current_accuracy(train_features, test_features)}')
    
    print("here 0")
    population, toolbox, pset = initialization(train_features, val_features, test_features)
    best_fitness_per_gen = []
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    logbook = tools.Logbook()
    
    pop, log = algorithms.eaMuPlusLambda(
        population,
        toolbox,
        mu=100,
        lambda_=90,
        cxpb=0.8,
        mutpb=0,
        ngen=3,
        stats=stats,
        halloffame=None,
        verbose=True
    )
    best_fitness_per_gen = log.select("max")
    
    plt.plot(range(len(best_fitness_per_gen)), best_fitness_per_gen, marker='o', linestyle='-', color='b', label="Best Accuracy")
    plt.xlabel("Generation")
    plt.ylabel("Best Accuracy")
    plt.title("Evolution of Best Accuracy Over Generations")
    plt.legend()
    plt.grid()
    plt.show()
    
    best_ind = population[0]
    best_accuracy = evaluate(best_ind, toolbox, pset, data_type="test")

    for i in range(len(population)):
        accuracy_results = evaluate(population[i], toolbox, pset, data_type="test")
        if accuracy_results>best_accuracy:
            best_ind = population[i]
            best_accuracy = accuracy_results
    save_individual(best_ind, i+11)
    val_accuracy = evaluate(best_ind, toolbox, pset, data_type="validation")
    test_accuracy = evaluate(best_ind, toolbox, pset, data_type="test")
    nodes_val_accuracy = evaluate_nodes(best_ind, toolbox, pset, data_type="validation")
    nodes_test_accuracy = evaluate_nodes(best_ind, toolbox, pset, data_type="test")
    print(val_accuracy, test_accuracy, nodes_val_accuracy, nodes_test_accuracy)
run_experiments()