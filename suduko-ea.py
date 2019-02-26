from matplotlib import pyplot as plt
from random import shuffle, choice, random
from time import time


def analysis():
    """
    Function: analysis
    Description: Runs the testing and averaging of the evolutionary algorithm.
    Params: None
    Returns: None
    """

    # Holder variables for the testing data.
    gens = []
    fits = []
    times = []

    # Test each of the 3 grids.
    for i in range(1, 3+1):
        grid = read_grid("Grid" + str(i) + ".ss")
        sizes = [10, 100, 1000, 10000]

        # Holder variables per grid.
        avg_gen = []
        avg_fit = []
        avg_time = []

        # Iterate over each population size.
        for population_size in sizes:
            gen_found = []
            fit_found = []
            time_found = []

            for b in range(1, 5+1):
                fail_count = 0
                start = time()
                while True:
                    gen, fit = evolve(grid, population_size)

                    if fit == 0:
                        break
                    if fail_count < 3:
                        fail_count += 1
                        print("Local minimum reached - recreating seed. (Grid", i, "Pop", population_size, "Test", b,
                              ")")
                    else:
                        print("Local minimum reached - population size uneffective.")
                        break


                time_e = time() - start

                gen_found.append(gen)
                fit_found.append(fit)
                time_found.append(time_e)

                print("Grid: ", i, " Pop: ", population_size,
                      " Test ", b, " Completed in ",
                      time_e, "with fitness", fit,
                      "at generation", gen)

            avg_gen.append(sum(gen_found) / len(gen_found))
            avg_fit.append(sum(fit_found) / len(fit_found))
            avg_time.append(sum(time_found) / len(time_found))

        gens.append(avg_gen)
        fits.append(avg_fit)
        times.append(avg_time)

    # Plot the data.
    xs = ['10', '100', '1000', '10000']

    plt.figure(num=None, figsize=(8, 15))
    plt.subplot(3, 1, 1)
    plt.title("Experiments")
    plt.plot(xs, gens[0])
    plt.plot(xs, gens[1])
    plt.plot(xs, gens[2])
    plt.legend(['Grid1.ss', 'Grid2.ss', 'Grid3.ss'], loc='upper right')
    plt.ylabel('Generation Found')

    plt.subplot(3, 1, 2)
    plt.plot(xs, fits[0])
    plt.plot(xs, fits[1])
    plt.plot(xs, fits[2])
    plt.ylabel('Fitness Achieved')

    plt.subplot(3, 1, 3)
    plt.plot(xs, times[0])
    plt.plot(xs, times[1])
    plt.plot(xs, times[2])
    plt.ylabel('Time Elapsed')

    plt.xlabel("Population Size")
    plt.show()


def read_grid(file_name):
    """
    Function: read_grid
    Description: Creates the 2D array based upon the specified file name.
    Params: file_name - String
    Returns: f_grid - List of List of Int
    """
    with open(file_name) as f:
        f_grid = []
        for line in f.readlines():
            if line[0] == '-':
                continue
            f_line = []
            for char in line:
                if char == '.':
                    f_line.append(0)
                elif char in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    f_line.append(int(char))
            f_grid.append(f_line)
    f.close()
    return f_grid


def evolve(goal, pop_size, m_rate=0.075, t_rate=0.5):
    """
    Function: evolve
    Description: This function is an EA to solve the sudoku problem.
    Params: goal - List of List of Int
            pop_size - Int
            m_rate - Float
            t_rate - Float
    Returns:    gen - Int
                overall_best_fit - Int
                end - start - Float
    """
    pop = create_seed(goal, pop_size)
    fit = check_fitness(pop)

    overall_best_fit = 145          # Highest achievable fitness + 1.
    gen = 0
    fail_count = 0

    # Define the termination clause.
    if pop_size == 10:
        fail_limit = 10000
    elif pop_size == 100:
        fail_limit = 7500
    elif pop_size == 1000:
        fail_limit = 500
    elif pop_size == 10000:
        fail_limit = 250
    else:
        fail_limit = 1000

    # Begin computation.
    while overall_best_fit > 0 and fail_count < fail_limit:
        gen += 1
        pop = select_pop(pop, fit, pop_size, t_rate)
        pop = crossover_pop(pop, pop_size)
        pop = mutate_pop(goal, pop, m_rate)
        fit = check_fitness(pop)
        best_child, best_fit = best_pop(pop, fit)
        if best_fit < overall_best_fit:
            overall_best_fit = best_fit
            fail_count = 0
        else:
            fail_count += 1


    del pop, fit
    return gen, overall_best_fit


def create_seed(host, pop_size):
    """
    Function: create_seed
    Description: Creates the initial population from the host.
    Params: host - List of List of Int
            pop_size - Int
    Returns: pop - List of List of Int
    """
    pop = []
    for i in range(pop_size):
        sol = clone(host)
        for line in sol:
            # Fill each row in the solution without duplication.
            missing = list({1, 2, 3, 4, 5, 6, 7, 8, 9} - set(line))
            shuffle(missing)
            while missing:
                for j in range(len(line)):
                    if line[j] == 0:
                        line[j] = missing.pop()
                        break
        pop.append(sol)
    return pop


def best_pop(population, fitness_population):
    """
    Function: best_pop
    Description: Returns the best solution and it's fitness.
    Params: population - List of Lists of Lists of Int
            fitness_population - List of Int
    Returns: Tuple of List of List of Int, Int
    """
    return sorted(zip(population, fitness_population), key=lambda ind_fit: ind_fit[1])[0]


def select_pop(population, fitness_population, pop_size, t_rate):
    """
    Function: select_pop
    Description: Select a percentage of the solutions based on the truncation rate and the fitness.
    Params: population - List of List of List of Int
            fitness_population - List of Int
            pop_size - Int
            t_rate - Float
    Returns: List of Lists of Lists of Int
    """
    sorted_population = sorted(zip(population, fitness_population), key=lambda ind_fit: ind_fit[1])
    return [individual for individual, fitness in sorted_population[:int(pop_size * t_rate)]]


def crossover_pop(population, pop_size):
    """
    Function: crossover_pop
    Description: Carries forward the best parent and breeds to create a new population.
    Params: population - List of List of List of Int
            pop_size - Int
    Returns: cross - List of List of List of Int
    """
    cross = [crossover_ind(choice(population), choice(population)) for _ in range(pop_size - 1)]
    cross.append(population[0])
    del population
    return cross


def crossover_ind(individual1, individual2):
    """
        Function: crossover_ind
        Description: Breeds two parents.
        Params: individual1 - List of List of Int
                individual2 - List of List of Int
        Returns: child - List of List of Int
    """
    child = []
    for i in range(0, 9):
        if random() > 0.5:
            child.append(individual1[i][:])
        else:
            child.append(individual2[i][:])
    return child


def check_not_fixed(grid, x, y):
    """
    Function: check_not_fixed
    Description: Test whether the cell is fixed - a clue.
    Params: grid - List of List of Int
            x - Int
            y - Int
    Returns: Boolean
    """
    return grid[x][y] == 0


def mutate_pop(grid, pop, m_rate):
    """
    Function: mutate_pop
    Description: Mutate a population dependent on a mutation rate.
    Params: grid - List of List of Int
            pop - List of List of List of Int
            m_rate - Float
    Returns: List of List of List of Int
    """
    return [mutate_ind(grid, clone(sol), m_rate) for sol in pop]


def mutate_ind(grid, sol, m_rate):
    """
        Function: mutate_ind
        Description: Mutate some non-fixed cells dependent on a mutation rate.
        Params: grid - List of List of Int
                sol - List of List of Int
                m_rate - Float
        Returns: List of List of Int
    """
    for x in range(9):
        for y in range(9):
            if check_not_fixed(grid, x, y):
                if random() < m_rate:
                    switch = get_switch(grid, sol, x)
                    if switch != -1:
                        temp = sol[x][y]
                        sol[x][y] = sol[x][switch]
                        sol[x][switch] = temp
                    continue
    return sol


def get_switch(grid, sol, x, check=1):
    """
    Function: get_switch
    Description: Find a non-fixed cell to switch with.
    Params: grid - List of List of Int
            sol - List of List of Int
            x - Int
            check - Int
    Returns: Int
    """
    if check < 8:
        switch_y = choice(range(9))
        return switch_y if check_not_fixed(grid, x, switch_y) else get_switch(grid, sol, x, check+1)
    else:
        return -1


def clone(grid):
    """
    Function: clone
    Description: Copy a List of List of Elements
    Params: grid - List of List of Elements
    Returns: Lists of Lists of Int
    """
    return [x[:] for x in grid]


def check_fitness(pop):
    """
    Function: check_fitness
    Description: Check the fitness of a given list of solutions.
    Params: pop - List of List of List of Int
    Returns: List of Int
    """
    return [fitness(sol) for sol in pop]


def fitness(sol):
    """
        Function: fitness
        Description: Check the fitness of a given solutions.
        Params: sol - List of List of Int
        Returns: Int
        """
    return check_vertical(sol) + check_squares(sol)


def check_vertical(sol):
    """
        Function: check_vertical
        Description: Check the fitness of the columns in a solution.
        Params: sol - List of List of Int
        Returns: Int
        """
    errors = 0
    for line in [list(i) for i in zip(*sol)]:
        errors += check_line(line)
    return errors


def check_squares(sol):
    """
        Function: check_squares
        Description: Check the fitness of the squares in a solution.
        Params: pop - List of List of Int
        Returns: Int
    """
    errors = 0
    for x in range(0, 9, 3):
        for y in range(0, 9, 3):
            errors += check_line(sum([row[y:y+3] for row in sol[x:x+3]], []))
    return errors


def check_line(line):
    """
        Function: check_line
        Description: Check the fitness of a line.
        Params: line - List of Int
        Returns: Int
    """
    return 9 - len(set(line))


if __name__ == "__main__":
    analysis()
