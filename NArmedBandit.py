# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 17:41:25 2016

@author: Fedor Sulaev
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

armsNum = 10
armValueMean = 0
armValueSD = 1.0
armRewardSD = 1.0
tasksNum = 2000
playsNum = 1000
epsilon = 0.1
temperature = 0.5

def update_progress(progress):
    sys.stdout.write("\r%d%%" % progress)
    
def plot_results(results, title):
    plt.plot(range(0, playsNum), np.mean(results, axis=0))
    plt.xlabel("Plays")
    plt.ylabel("Average reward")
    plt.title(title)

def learn_with_epsilon_greedy():
    results = []
    for taskIndex in range(0, tasksNum):
        taskResults = []
        armMeanValues = np.random.normal(armValueMean, armValueSD, armsNum)
        estimates = [0.0]*armsNum
        tries = [[] for i in range(armsNum)]
        for playIndex in range(0, playsNum):
            armIndex = -1
            if np.random.random_sample() < epsilon:
                armIndex = np.random.randint(0, armsNum)
            else:
                armIndex = np.argmax(estimates)
            reward = np.random.normal(armMeanValues[armIndex], armRewardSD)
            tries[armIndex].append(reward)
            estimates[armIndex] = np.mean(tries[armIndex])
            taskResults.append(reward)
        results.append(taskResults)
        update_progress(float(taskIndex) / float(tasksNum-1) * 100.0)
    plot_results(results, "Learning with Epsilon-greedy")
    
def learn_with_softmax():
    results = []
    for taskIndex in range(0, tasksNum):
        taskResults = []
        armMeanValues = np.random.normal(armValueMean, armValueSD, armsNum)
        estimates = [0.0]*armsNum
        tries = [[] for i in range(armsNum)]
        for playIndex in range(0, playsNum):
            choiceProbs = []
            for i in range(armsNum):
                numerator = np.exp(estimates[i]/temperature)
                denominator = 0.0
                for j in range(armsNum):
                    if j != i:
                        denominator += np.exp(estimates[j]/temperature)
                choiceProbs.append(numerator / denominator)
            choiceProbs = np.array(choiceProbs)
            choiceProbs /= choiceProbs.sum()
            armIndex = np.random.choice(range(armsNum), p=choiceProbs)
            reward = np.random.normal(armMeanValues[armIndex], armRewardSD)
            tries[armIndex].append(reward)
            estimates[armIndex] = np.mean(tries[armIndex])
            taskResults.append(reward)
        results.append(taskResults)
        update_progress(float(taskIndex) / float(tasksNum-1) * 100.0)
    plot_results(results, "Learning with Softmax")
    
#learn_with_epsilon_greedy()
learn_with_softmax()
