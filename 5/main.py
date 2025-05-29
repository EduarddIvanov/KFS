import numpy as np
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time


class ParallelGeneticAlgorithmTSP:
    def __init__(self, cities, population_size=100, num_generations=50,
                 mutation_rate=0.01, crossover_rate=0.9,
                 num_parents=50, num_elites=2, num_processes=4):
        self.cities = np.array(cities)
        self.num_cities = len(cities)
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_parents = num_parents
        self.num_elites = num_elites
        self.num_processes = num_processes

        # Ініціалізація популяції
        self.population = self.initialize_population()

        # Зберігаємо історію для статистики
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.worst_fitness_history = []

    # 1) Ініціалізація популяції
    def initialize_population(self):
        #Створення початкової популяції з випадковими маршрутами
        population = []
        for _ in range(self.population_size):
            # Створюємо випадковий маршрут (перестановку міст)
            route = np.random.permutation(self.num_cities)
            population.append(route)
        return np.array(population)

    # 2) Оцінка придатності
    def calculate_distance(self, route):
        #Обчислення загальної відстані для маршруту
        total_distance = 0
        for i in range(self.num_cities):
            start_city = route[i]
            end_city = route[(i + 1) % self.num_cities]
            total_distance += np.linalg.norm(self.cities[start_city] - self.cities[end_city])
        return total_distance

    def evaluate_fitness(self, population):
        #Обчислення придатності для кожної особини у популяції
        with Pool(self.num_processes) as pool:
            distances = pool.map(self.calculate_distance, population)
        # Придатність - це обернена величина до відстані (хочемо мінімізувати відстань)
        fitness = 1 / np.array(distances)
        return fitness

    # 3) Вибір батьків (турнірний вибір)
    def select_parents(self, population, fitness, num_parents):
        #Вибір батьків за допомогою турнірного відбору
        parents = []
        for _ in range(num_parents):
            # Вибираємо випадкові 5 осіб для турніру
            tournament_indices = np.random.randint(0, len(population), 5)
            tournament_fitness = fitness[tournament_indices]
            # Вибираємо переможця турніру (найвища придатність)
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_index])
        return np.array(parents)

    # 4) Створення нащадків (кросовер та мутація)
    def ordered_crossover(self, parent1, parent2):
        #Впорядкований кросовер для TSP
        size = self.num_cities
        child = np.zeros(size, dtype=int)
        child.fill(-1)

        # Вибираємо випадковий підмаршрут від першого батька
        start, end = sorted(np.random.randint(0, size, 2))
        child[start:end] = parent1[start:end]

        # Заповнюємо решту міст в порядку другого батька
        current_pos = 0
        for city in parent2:
            if city not in child:
                while current_pos < size and child[current_pos] != -1:
                    current_pos += 1
                if current_pos >= size:
                    break
                child[current_pos] = city

        return child

    def swap_mutation(self, route):
        #Мутація обміном двох випадкових міст
        if random.random() < self.mutation_rate:
            i, j = np.random.randint(0, self.num_cities, 2)
            route[i], route[j] = route[j], route[i]
        return route

    def create_offspring(self, parents):
        #Створення нової популяції нащадків
        offspring = []
        # Додаємо елітних батьків без змін
        for i in range(self.num_elites):
            offspring.append(parents[i])

        # Створюємо нащадків
        while len(offspring) < self.population_size:
            parent1, parent2 = parents[np.random.randint(0, len(parents), 2)]

            if random.random() < self.crossover_rate:
                child = self.ordered_crossover(parent1, parent2)
            else:
                child = parent1.copy()  # Без кросовера - копіюємо батька

            child = self.swap_mutation(child)
            offspring.append(child)

        return np.array(offspring)

    # 6) Вибір для наступної популяції (елітизм)
    def select_next_population(self, population, fitness, new_population):
        #Вибір наступної популяції з комбінації поточної та нової
        # Об'єднуємо старі та нові особини
        combined_population = np.vstack((population, new_population))
        combined_fitness = np.concatenate((fitness, self.evaluate_fitness(new_population)))

        # Вибір кращих осіб
        sorted_indices = np.argsort(combined_fitness)[::-1]  # Сортування за спаданням
        next_population = combined_population[sorted_indices[:self.population_size]]

        return next_population

    # 7) Умова зупинки
    def check_stopping_condition(self, generation):
        #Перевірка умов зупинки
        if self.num_generations == -1:  # Режим "до останнього живого"
            return False
        return generation >= self.num_generations

    # 8) Вивід результатів
    def get_best_solution(self):
        #Повертає найкращий маршрут та його відстань
        fitness = self.evaluate_fitness(self.population)
        best_idx = np.argmax(fitness)
        best_route = self.population[best_idx]
        best_distance = self.calculate_distance(best_route)
        return best_route, best_distance

    def plot_statistics(self):
        #Візуалізація статистики по поколіннях
        plt.figure(figsize=(12, 6))
        plt.plot(self.best_fitness_history, label='Найкраща придатність')
        plt.plot(self.avg_fitness_history, label='Середня придатність')
        plt.plot(self.worst_fitness_history, label='Найгірша придатність')
        plt.xlabel('Покоління')
        plt.ylabel('Придатність (1/відстань)')
        plt.title('Статистика придатності по поколіннях')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_route(self, route):
        #Візуалізація маршруту
        plt.figure(figsize=(8, 8))
        # Міста
        plt.scatter(self.cities[:, 0], self.cities[:, 1], c='red', s=100)
        for i, city in enumerate(self.cities):
            plt.text(city[0], city[1], str(i), fontsize=12)

        # Маршрут
        route_cities = self.cities[route]
        route_cities = np.vstack((route_cities, route_cities[0]))  # Замикаємо маршрут
        plt.plot(route_cities[:, 0], route_cities[:, 1], 'b-')

        plt.title(f'Маршрут комівояжера (довжина: {self.calculate_distance(route):.2f})')
        plt.xlabel('X координата')
        plt.ylabel('Y координата')
        plt.grid()
        plt.show()

    # Головний цикл генетичного алгоритму
    def run(self):
        #Запуск генетичного алгоритму
        start_time = time.time()

        for generation in range(1, self.num_generations + 1 if self.num_generations != -1 else 1000000):
            # Оцінка придатності поточної популяції
            fitness = self.evaluate_fitness(self.population)

            # Збереження статистики
            self.best_fitness_history.append(np.max(fitness))
            self.avg_fitness_history.append(np.mean(fitness))
            self.worst_fitness_history.append(np.min(fitness))

            # Вивід статистики
            best_dist = 1 / np.max(fitness)
            avg_dist = 1 / np.mean(fitness)
            print(f"Покоління {generation}: Найкраща відстань = {best_dist:.2f}, Середня відстань = {avg_dist:.2f}")

            # Перевірка умови зупинки
            if self.check_stopping_condition(generation):
                break

            # Вибір батьків
            parents = self.select_parents(self.population, fitness, self.num_parents)

            # Створення нащадків
            offspring = self.create_offspring(parents)

            # Вибір наступної популяції
            self.population = self.select_next_population(self.population, fitness, offspring)

        # Вивід результатів
        best_route, best_distance = self.get_best_solution()
        print(f"\nНайкращий маршрут знайдено за {time.time() - start_time:.2f} секунд")
        print(f"Найкраща відстань: {best_distance:.2f}")
        print(f"Найкращий маршрут: {best_route}")

        # Візуалізація
        self.plot_statistics()
        self.plot_route(best_route)

        return best_route, best_distance


# Тестовий приклад
if __name__ == "__main__":
    # Приклад міст (координати)
    cities = np.array([
        [60, 200], [180, 200], [80, 180], [140, 180], [20, 160],
        [100, 160], [200, 160], [140, 140], [40, 120], [100, 120],
        [180, 100], [60, 80], [120, 80], [180, 60], [20, 40],
        [100, 40], [200, 40], [20, 20], [60, 20], [160, 20]
    ])

    # Параметри алгоритму
    params = {
        'cities': cities,
        'population_size': 200,
        'num_generations': 50,
        'mutation_rate': 0.02,
        'crossover_rate': 0.9,
        'num_parents': 100,
        'num_elites': 5,
        'num_processes': 4
    }

    # Запуск алгоритму
    ga = ParallelGeneticAlgorithmTSP(**params)
    best_route, best_distance = ga.run()
