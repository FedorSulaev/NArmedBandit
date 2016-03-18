# NArmedBandit
N armed bandit problem from the book "Reinforcement Learning: An Introduction"

Exercise 2.2 (programming)   How does the softmax action selection method using the Gibbs distribution fare on the 10-armed testbed? Implement the method and run it at several temperatures to produce graphs similar to those in Figure  2.1. To verify your code, first implement the epsilon-greedy methods and reproduce some specific aspect of the results in Figure  2.1.

Exercise 2.5   Give pseudocode for a complete algorithm for the n-armed bandit problem. Use greedy action selection and incremental computation of action values with  step-size parameter. Assume a function bandit(a) that takes an action and returns a reward. Use arrays and variables; do not subscript anything by the time index t. Indicate how the action values are initialized and updated after each reward.

Exercise 2.7 (programming)   Design and conduct an experiment to demonstrate the difficulties that sample-average methods have for nonstationary problems. Use a modified version of the 10-armed testbed in which all the Q*(a) start out equal and then take independent random walks. Prepare plots like Figure  2.1 for an action-value method using sample averages, incrementally computed by , and another action-value method using a a constant step-size parameter, alpha=0.1. Use epsilon=0.1 and, if necessary, runs longer than 1000 plays.

