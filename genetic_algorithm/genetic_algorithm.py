import numpy as np
from tqdm import tqdm


class TSPGeneticAlgorithm:
    def __init__(
        self,
        node_coords: np.ndarray,
        population_size: int = 100,
        num_generations: int = 500,
        mutation_rate: float = 0.01,
        mutation_falloff: float = 0.9,
    ) -> None:
        """
        Genetic algorithm solver for the Traveling Salesman Problem.

        Args:
            node_coords: coordinates of nodes (cities)
            population_size: number of candidate solutions in population
            num_generations: number of generations to evolve
            mutation_rate: initial mutation probability per node swap
            mutation_falloff: rate to decrease mutation rate each generation
        """
        self.node_coords = node_coords
        self.num_nodes = node_coords.shape[0]
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.mutation_falloff = mutation_falloff

        self.half_population_size = int(self.population_size / 2)
        self.population = self.build_initial_population()

        self.best_fitnesses: list[float] | None = None

    @staticmethod
    def compute_distance(node1: np.ndarray, node2: np.ndarray) -> float:
        return np.linalg.norm(node1 - node2)

    def compute_route_length(self, route: np.ndarray) -> float:
        route_length = sum(
            self.compute_distance(
                self.node_coords[route[i]], self.node_coords[route[i + 1]]
            )
            for i in range(len(route) - 1)
        )
        # include path from end to start
        route_length += self.compute_distance(
            self.node_coords[route[-1]], self.node_coords[route[0]]
        )
        return route_length

    def build_initial_population(self) -> np.ndarray:
        population = np.array(
            [
                np.random.choice(
                    np.arange(self.num_nodes), self.num_nodes, replace=False
                )
                for _ in range(self.population_size)
            ]
        )
        return population

    def compute_fitness(self, route: np.ndarray) -> float:
        return 1 / self.compute_route_length(route)

    def sample_pair_for_crossover(
        self, population: np.ndarray, fitnesses: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        "Sample pairs weighted by fitness"
        idx = np.random.choice(
            len(population), size=2, replace=False, p=fitnesses / np.sum(fitnesses)
        )
        return population[idx[0]], population[idx[1]]

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        "Start with a segment of parent1, try to fill the rest with parent2"

        start, end = sorted(np.random.choice(np.arange(self.num_nodes), 2))
        child = [-1] * self.num_nodes
        child[start:end] = parent1[start:end]

        fill_pos = end
        for node in parent2:
            if node not in child:
                if fill_pos >= self.num_nodes:
                    fill_pos = 0
                child[fill_pos] = node
                fill_pos += 1
        return np.array(child)

    def mutate(self, route: np.ndarray) -> np.ndarray:
        "Randomly swap nodes in the route based on a mutation rate"
        for i in range(self.num_nodes):
            if np.random.uniform() < self.mutation_rate:
                j = np.random.randint(self.num_nodes)
                route[i], route[j] = route[j], route[i]
        return route

    def update_mutation_rate(self) -> None:
        self.mutation_rate *= self.mutation_falloff

    def optimise(self) -> tuple[float, np.ndarray | None]:
        """
        Run the genetic algorithm optimization process.

        Returns:
            best fitness value found,
            best route (sequence of node indices)
        """
        best_fitness = -np.inf
        best_route = None
        best_fitnesses = []
        for generation in tqdm(range(self.num_generations)):
            fitnesses = np.array(
                [self.compute_fitness(route) for route in self.population]
            )

            if np.max(fitnesses) > best_fitness:
                new_best_idx = np.argmax(fitnesses)
                best_fitness = fitnesses[new_best_idx]
                best_route = self.population[new_best_idx].copy()

            best_fitnesses.append(best_fitness)

            # breed new children
            children = []
            for _ in range(self.population_size - self.half_population_size):
                parent1, parent2 = self.sample_pair_for_crossover(
                    self.population, fitnesses
                )
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                children.append(child)
            children = np.array(children)

            # mutate top half
            top_population = self.population[np.argsort(-fitnesses)][
                : self.half_population_size
            ]
            for i in range(self.half_population_size):
                top_population[i] = self.mutate(top_population[i])

            # combine
            self.population = np.vstack([top_population, children])
            self.update_mutation_rate()

        self.best_fitnesses = best_fitnesses
        return best_fitness, best_route
