# Genetic Algorithm
Genetic algorithms are a type of optimisation inspired by natural selection and genetics. They are an example of an evolutionary algorithm.

Wiki [here](https://en.wikipedia.org/wiki/Genetic_algorithm)

## Theory
To apply a genetic algorithm, we need a set of parameters we are optimising, and a fitness to evaluate these parameters.

There are many different types of genetic algorithm, and often require tuning to each problem. However, the general steps are:
1. Initialise a population of parameters.
1. Evaluate the population.
1. Select the best performing parameter sets.
1. Evolve the population. There are various ways to do this, but the two most common are:
    1. Mutation - randomly alter the parameters
    1. Crossover - create "offspring" by combining parameter sets.
1. Repeat from step 2.

This lets us narrow down on well performing individuals, whilst keeping the set diverse to avoid local minima.

## Example
In the example we apply this algorithm to solve the Traveling Salesman Problem (TSP). Since, as mentioned, the implementation often is specific to the problem, our implementation is specific to TSP. Attempting to abstract above this problem is not the goal of this repository. 
