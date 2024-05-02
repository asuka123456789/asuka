import numpy as np

# 不满意度矩阵
dissatisfaction_matrix = np.array([
    [6, 4, 7, 7, 7],
    [6, 7, 4, 4, 7],
    [7, 3, 5, 4, 3],
    [7, 5, 5, 4, 4],
    [6, 7, 7, 7, 7],
    [5, 4, 4, 6, 3],
    [7, 3, 3, 5, 6],
    [4, 4, 3, 6, 6],
    [5, 6, 6, 5, 3],
    [7, 7, 6, 3, 3],
    [7, 6, 5, 7, 3]
])

# 遗传算法参数
population_size = 50
num_generations = 1000
mutation_rate = 0.1

# 初始化种群
population = [np.random.choice(range(len(dissatisfaction_matrix[0])), size=len(dissatisfaction_matrix)) for _ in
              range(population_size)]


def fitness(individual):
    return np.sum([dissatisfaction_matrix[i][individual[i]] for i in range(len(individual))])


# 遗传算法主循环
for generation in range(num_generations):
    # 计算适应度
    fitness_values = [fitness(individual) for individual in population]

    # 选择
    selected_indices = np.random.choice(range(population_size), size=population_size, replace=True,
                                        p=(1 / np.array(fitness_values)) / sum(1 / np.array(fitness_values)))
    selected_population = [population[i] for i in selected_indices]

    # 交叉
    offspring = []
    for _ in range(population_size):
        parent_indices = np.random.choice(range(population_size), size=2, replace=False)
        parent1, parent2 = selected_population[parent_indices[0]], selected_population[parent_indices[1]]
        crossover_point = np.random.randint(len(parent1))
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        offspring.append(child)

    # 变异
    for individual in offspring:
        for i in range(len(individual)):
            if np.random.random() < mutation_rate:
                individual[i] = np.random.choice(range(len(dissatisfaction_matrix[0])))

    # 替换
    population = offspring

# 找到最优解
best_individual = population[np.argmin([fitness(individual) for individual in population])]
best_fitness = fitness(best_individual)

# 输出最优解
print("Best solution:")
for i in range(len(best_individual)):
    print(f"Employee {i + 1} assigned to task {best_individual[i] + 1}")
print("Best fitness:", best_fitness)

