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

def epsilon_greedy(epsilon, arms_num, estimates):
    arm_index = -1
    if np.random.random_sample() < epsilon:
        arm_index = np.random.randint(0, arms_num)
    else:
        arm_index = np.argmax(estimates)
    return arm_index
            
    
def softmax(temperature, arms_num, estimates):
    choice_probs = []
    for i in range(arms_num):
        numerator = np.exp(estimates[i]/temperature)
        denominator = 0.0
        for j in range(arms_num):
            if j != i:
                denominator += np.exp(estimates[j]/temperature)
        choice_probs.append(numerator / denominator)
    choice_probs = np.array(choice_probs)
    choice_probs /= choice_probs.sum()
    arm_index = np.random.choice(range(arms_num), p=choice_probs)
    return arm_index
    
def learn(tasks_num, arm_value_mean, arm_value_sd, arms_num, arm_reward_sd, 
          plays_num, step_size, with_random_walks, random_walk_step, 
          explorer_func, *params):
    results = []
    for task_index in range(0, tasks_num):
        task_results = []
        if with_random_walks:
            arm_mean_values = [arm_value_mean]*arms_num
        else:
            arm_mean_values = np.random.normal(
                arm_value_mean, arm_value_sd, arms_num)
        estimates = [0.0]*arms_num
        tries = [0]*arms_num
        for palayIndex in range(0, plays_num):
            arm_index = explorer_func(params[0], arms_num, estimates)
            reward = np.random.normal(arm_mean_values[arm_index], arm_reward_sd)
            tries[arm_index] += 1
            if step_size == 0:
                estimates[arm_index] = (estimates[arm_index] + 
                    1.0 / float(tries[arm_index]) 
                    * (reward - estimates[arm_index]))
            else:
                estimates[arm_index] = (estimates[arm_index] +
                    step_size * (reward - estimates[arm_index]))
            task_results.append(reward)
            if with_random_walks:
                for arm_index in range(0, arms_num):
                    step = np.random.choice(
                        (-random_walk_step, random_walk_step))
                    arm_mean_values[arm_index] += step
        results.append(task_results)
        update_progress(float(task_index) / float(tasks_num-1) * 100.0)
    return results
    
plot_results(learn(2000, 0.0, 1.0, 10, 1.0, 1000, 0.1, True, 0.1,
                   epsilon_greedy, 0.1), 
                   "Learning with Epsilon-greedy")