# backend/agi_core/cppn_evolver.py

import logging
import numpy as np
import random
from typing import List, Dict, Any, Callable

logger = logging.getLogger(__name__)

class CPPN:
    """
    Represents a Compositional Pattern-Producing Network (CPPN).
    A CPPN is a neural network that can generate complex patterns and structures.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int]):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.weights = self._initialize_weights()
        self.activation_functions = self._initialize_activation_functions()

    def _initialize_weights(self) -> List[np.ndarray]:
        """Initializes the weights for the network."""
        layers = [self.input_dim] + self.hidden_layers + [self.output_dim]
        return [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers) - 1)]

    def _initialize_activation_functions(self) -> List[Callable[[np.ndarray], np.ndarray]]:
        """Initializes activation functions for each layer."""
        # A variety of activation functions allows for more complex patterns
        functions = [np.tanh, np.sin, np.cos, lambda x: np.exp(-x**2)]
        return [random.choice(functions) for _ in self.hidden_layers + [self.output_dim]]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Performs a forward pass through the network."""
        for i, (w, func) in enumerate(zip(self.weights, self.activation_functions)):
            x = func(np.dot(x, w))
        return x

class EvolutionaryOptimizer:
    """
    Manages the evolution of a population of CPPNs to optimize for a given fitness function.
    """
    def __init__(self, population_size: int, input_dim: int, output_dim: int, hidden_layers: List[int], fitness_func: Callable[[CPPN], float]):
        self.population_size = population_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.fitness_func = fitness_func
        self.population = self._initialize_population()

    def _initialize_population(self) -> List[CPPN]:
        """Creates the initial population of CPPNs."""
        return [CPPN(self.input_dim, self.output_dim, self.hidden_layers) for _ in range(self.population_size)]

    def evolve(self, generations: int, mutation_rate: float = 0.01, crossover_rate: float = 0.7):
        """
        Runs the evolutionary process for a specified number of generations.

        Args:
            generations (int): The number of generations to evolve.
            mutation_rate (float): The probability of a weight being mutated.
            crossover_rate (float): The probability of two parents producing offspring.
        """
        for gen in range(generations):
            fitness_scores = [self.fitness_func(cppn) for cppn in self.population]
            
            new_population = []
            while len(new_population) < self.population_size:
                # Select parents based on fitness
                parent1, parent2 = self._select_parents(fitness_scores)
                
                # Crossover
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2
                
                # Mutation
                self._mutate(child1, mutation_rate)
                self._mutate(child2, mutation_rate)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.population_size]
            logger.info(f"Generation {gen+1}/{generations} complete. Max fitness: {max(fitness_scores)}")

    def _select_parents(self, fitness_scores: List[float]) -> tuple[CPPN, CPPN]:
        """Selects two parents from the population using tournament selection."""
        tournament_size = 5
        
        def tournament() -> CPPN:
            contenders = random.sample(list(zip(self.population, fitness_scores)), tournament_size)
            return max(contenders, key=lambda item: item[1])[0]
            
        return tournament(), tournament()

    def _crossover(self, parent1: CPPN, parent2: CPPN) -> tuple[CPPN, CPPN]:
        """Performs a single-point crossover on the weights of two parent CPPNs."""
        child1 = CPPN(self.input_dim, self.output_dim, self.hidden_layers)
        child2 = CPPN(self.input_dim, self.output_dim, self.hidden_layers)
        
        for i in range(len(parent1.weights)):
            if random.random() < 0.5:
                child1.weights[i] = parent1.weights[i]
                child2.weights[i] = parent2.weights[i]
            else:
                child1.weights[i] = parent2.weights[i]
                child2.weights[i] = parent1.weights[i]
        
        return child1, child2

    def _mutate(self, cppn: CPPN, mutation_rate: float):
        """Mutates the weights of a CPPN."""
        for i in range(len(cppn.weights)):
            for row in range(cppn.weights[i].shape[0]):
                for col in range(cppn.weights[i].shape[1]):
                    if random.random() < mutation_rate:
                        cppn.weights[i][row, col] += np.random.randn() * 0.1

    def get_fittest(self) -> CPPN:
        """Returns the fittest individual in the current population."""
        fitness_scores = [self.fitness_func(cppn) for cppn in self.population]
        return self.population[np.argmax(fitness_scores)]
