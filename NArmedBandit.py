# -*- coding: utf-8 -*-
"""
Created on Fri Feb 05 17:41:25 2016

@author: Fedor Sulaev
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

def update_progress(progress):
    sys.stdout.write("\r%d%%" % progress)
    
def plot_results(results, title):
    plt.plot(range(0, len(results[0])), np.mean(results, axis=0))
    plt.xlabel("Plays")
    plt.ylabel("Average reward")
    plt.title(title)
    plt.show()

def epsilon_greedy(epsilon, armsNum, estimates):
    armIndex = -1
    if np.random.random_sample() < epsilon:
        armIndex = np.random.randint(0, armsNum)
    else:
        armIndex = np.argmax(estimates)
    return armIndex
            
    
def softmax(temperature, armsNum, estimates):
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
    return armIndex
    
def learn(tasksNum, armValueMean, armValueSD, armsNum, armRewardSD, playsNum,
          learner_func, *params):
    results = []
    for taskIndex in range(0, tasksNum):
        taskResults = []
        armMeanValues = np.random.normal(armValueMean, armValueSD, armsNum)
        estimates = [0.0]*armsNum
        tries = [[] for i in range(armsNum)]
        for palayIndex in range(0, playsNum):
            armIndex = learner_func(params[0], armsNum, estimates)
            reward = np.random.normal(armMeanValues[armIndex], armRewardSD)
            tries[armIndex].append(reward)
            estimates[armIndex] = np.mean(tries[armIndex])
            taskResults.append(reward)
        results.append(taskResults)
        update_progress(float(taskIndex) / float(tasksNum-1) * 100.0)
    return results
    
plot_results(learn(2000, 0.0, 1.0, 10, 1.0, 1000, 
                   epsilon_greedy, 0.1), 
                   "Learning with Epsilon-greedy")