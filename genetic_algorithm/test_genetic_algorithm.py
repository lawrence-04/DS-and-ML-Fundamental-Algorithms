import numpy as np
import pytest
from genetic_algorithm import TSPGeneticAlgorithm


@pytest.fixture
def coords():
    return np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


def test_initialization(coords):
    ga = TSPGeneticAlgorithm(coords, population_size=10)
    assert ga.node_coords.shape == (4, 2)
    assert ga.population.shape == (10, 4)
    assert all(len(set(route)) == 4 for route in ga.population)


def test_compute_distance(coords):
    ga = TSPGeneticAlgorithm(coords)
    d = ga.compute_distance(coords[0], coords[1])
    assert np.isclose(d, 1.0)


def test_route_length(coords):
    ga = TSPGeneticAlgorithm(coords)
    route = np.array([0, 1, 2, 3])  # square around the perimeter
    expected_length = 4.0  # 4 sides of unit length
    assert np.isclose(ga.compute_route_length(route), expected_length)


def test_mutation_preserves_permutation(coords):
    ga = TSPGeneticAlgorithm(coords)
    route = np.array([0, 1, 2, 3])
    mutated = ga.mutate(route.copy())
    assert sorted(mutated) == [0, 1, 2, 3]


def test_crossover_produces_valid_child(coords):
    ga = TSPGeneticAlgorithm(coords)
    parent1 = np.array([0, 1, 2, 3])
    parent2 = np.array([3, 2, 1, 0])
    child = ga.crossover(parent1, parent2)
    assert sorted(child) == [0, 1, 2, 3]
