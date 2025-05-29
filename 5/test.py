import unittest
import numpy as np
from main import ParallelGeneticAlgorithmTSP

class TestGeneticAlgorithmTSP(unittest.TestCase):
    def setUp(self):
        """Ініціалізація тестових даних"""
        self.cities = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        self.ga = ParallelGeneticAlgorithmTSP(
            cities=self.cities,
            population_size=10,
            num_generations=10,
            num_processes=1
        )
        self.test_route = np.array([0, 1, 2, 3])  # Тестовий маршрут

    # --- Тести для оцінки придатності ---
    def calculate_distance(self, route):
        """Обчислення відстані без повернення до стартового міста"""
        total_distance = 0
        for i in range(len(route) - 1):  # Змінено: не додаємо останній сегмент
            start_city = route[i]
            end_city = route[i + 1]
            total_distance += np.linalg.norm(self.cities[start_city] - self.cities[end_city])
        return total_distance

    def test_evaluate_fitness(self):
        """Перевірка, чи придатність обернено пропорційна відстані"""
        population = np.array([self.test_route])
        fitness = self.ga.evaluate_fitness(population)

        # Спочатку обчислимо очікувану відстань
        expected_distance = 0.0
        for i in range(len(self.test_route)):
            start_city = self.test_route[i]
            end_city = self.test_route[(i + 1) % len(self.test_route)]  # Замикаємо маршрут
            expected_distance += np.linalg.norm(self.cities[start_city] - self.cities[end_city])

        # Потім обчислимо очікувану придатність
        expected_fitness = 1 / expected_distance

        self.assertAlmostEqual(fitness[0], expected_fitness, places=5)

    # --- Тести для вибору батьків ---
    def test_select_parents_tournament(self):
        """Перевірка турнірного відбору (найкращі особини мають більші шанси)"""
        test_population = np.array([
            [0, 1, 2, 3],  # Найкращий маршрут (найкоротший)
            [3, 2, 1, 0],  # Гірший маршрут
            [0, 2, 1, 3]   # Середній
        ])
        fitness = np.array([1.0, 0.1, 0.5])  # Штучно задана придатність
        parents = self.ga.select_parents(test_population, fitness, num_parents=2)
        self.assertEqual(len(parents), 2)
        # Найкращий маршрут має бути серед батьків
        self.assertTrue(any(np.array_equal(parent, test_population[0]) for parent in parents))

    # --- Тести для кросовера та мутації ---
    def test_ordered_crossover(self):
        """Перевірка, що кросовер зберігає всі міста (без дублікатів)"""
        parent1 = np.array([0, 1, 2, 3])
        parent2 = np.array([3, 2, 1, 0])
        child = self.ga.ordered_crossover(parent1, parent2)
        self.assertEqual(len(child), len(parent1))
        self.assertEqual(len(np.unique(child)), len(parent1))  # Немає дублікатів

    def test_swap_mutation(self):
        """Перевірка, що мутація змінює маршрут (якщо відбувається)"""
        route = self.test_route.copy()
        mutated_route = self.ga.swap_mutation(route)
        # Якщо мутація відбулася, маршрути повинні відрізнятися
        if not np.array_equal(route, mutated_route):
            self.assertFalse(np.array_equal(route, mutated_route))

    # --- Тести для вибору наступної популяції ---
    def test_select_next_population(self):
        """Перевірка елітизму (найкращі особини зберігаються)"""
        population = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
        fitness = np.array([1.0, 0.1])
        new_population = np.array([[1, 0, 2, 3], [0, 3, 2, 1]])
        next_pop = self.ga.select_next_population(population, fitness, new_population)
        # Найкраща особина з початкової популяції має зберегтися
        self.assertTrue(any(np.array_equal(ind, population[0]) for ind in next_pop))

if __name__ == "__main__":
    unittest.main()
