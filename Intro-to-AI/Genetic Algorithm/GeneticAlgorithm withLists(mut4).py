import random
from prettytable import PrettyTable

class GeneticAlgorithm:

    def __init__(self, size, k, mutation_probability, generations):
        self.size = size
        self.k = k
        #self.population = self.generate_population(100)
        self.population = self.generate_population()
        self.selection = []
        self.offspring = []
        self.mutation_probability = mutation_probability
        self.generations = generations
        self.max_fitness = self.size * (self.size - 1) // 2

    def generate_population(self):

        #generate a population of inital states of queens (chromosomes)

        population = []

        for chromosome in range(self.k):

            chromosome = []

            for i in range(self.size):
                pos = random.randint(1, self.size)
                chromosome.append(str(pos))
            
            population.append(chromosome)

        return population
    
    def fitness_function(self, chromosome):
        
        '''
        input: chromosome
        return: the number of non-attacking pairs in the chromosome
        '''
        overall_fitness = 0

        for i in range(self.size):

            for j in range(i+1, self.size):

                if chromosome[i] == chromosome[j]:
                    continue
                
                horizontal_distance = abs(i-j)
                vertical_distance = abs(int(chromosome[j]) - int(chromosome[i]))
                
                if horizontal_distance != vertical_distance:
                    overall_fitness += 1
        
        return overall_fitness
    
    def calculate_probabilities(self):

        # calculates the probabilities of the odds of a chromosme advancing to the next stage
        
        fitness_scores = []

        for chromosome in self.population:
            fitness_scores.append(self.fitness_function(chromosome))

        total_fitness_scores = sum(fitness_scores)

        probabilities = [fitness_score/total_fitness_scores for fitness_score in fitness_scores]

        return self.population, probabilities
    
    def select(self):

        # selects chromosomes based on their probability of being chosen

        chromosomes, probabilities = self.calculate_probabilities()

        selection = random.choices(chromosomes, probabilities, k = self.k)

        self.selection = selection
    
    def crossover(self):

        # creates a crossover chromsome between pairs

        crossover_point = random.randint(0, self.size-1)
        pairs = [self.selection[i:i+2] for i in range(0, self.k, 2)]

        offspring = []

        for pair in pairs:
            child1_firstsplit = pair[0][0:crossover_point]
            child1_secondsplit = pair[1][crossover_point:]
            first_child = child1_firstsplit + child1_secondsplit

            child2_firstsplit = pair[1][0:crossover_point]
            child2_secondsplit = pair[0][crossover_point:]

            second_child = child2_firstsplit + child2_secondsplit

            offspring.append(first_child)
            offspring.append(second_child)

        self.offspring = offspring


    def mutate(self):

        # randomly changes an element of a chromosome to ensure diversity

        mutated_set = []

        for chromosome in self.offspring:

            if(random.random() < self.mutation_probability):

                # Select unique random positions to mutate
                rand_cols = random.sample(range(self.size), 4)

                for rand_col in rand_cols:
                    chromosome[rand_col] = str(random.randint(1, self.size))  # Mutate the position

                mutated_chromosome = chromosome

                mutated_set.append(mutated_chromosome)
            else:
                mutated_set.append(chromosome)
        
        self.offspring = mutated_set

    def run_with_print(self):

        if self.size == 2:
            print('Unsolvable Puzzle!')

        if self.size == 3:
            print('Unsolvable_puzzle!')

        for generation in range(self.generations):
            print(f'Generation {generation + 1}')

            self.select()

            self.crossover()

            self.mutate()

            self.population = self.offspring

            best_chromosome, best_fitness = max([(chromosome, self.fitness_function(chromosome)) for chromosome in self.population], key = lambda x: x[1])
            print(f"Best fitness in this generation: {best_fitness}")

            if best_fitness == self.max_fitness:
                print('The maximum fitness value is: ' + str(self.max_fitness))
                print(f'Solution found at generation {generation + 1}')
                print('Solution: ' + best_chromosome)
                return
            
        print(f'No global maximum found. Best fitness achieved was {best_fitness} with chromsome: ' + best_chromosome)

    def run_with_return(self):

        if self.size == 2:
            return self.size, 0, 0, 0, ''

        for generation in range(self.generations):

            self.select()

            self.crossover()

            self.mutate()

            self.population = self.offspring

            best_chromosome, best_fitness = max([(chromosome, self.fitness_function(chromosome)) for chromosome in self.population], key = lambda x: x[1])

            if best_fitness == self.max_fitness:
                if self.size == 10:
                    return self.size, generation + 1, best_fitness, self.max_fitness, best_chromosome
                else:
                    return self.size, generation + 1, best_fitness, self.max_fitness, ''.join(best_chromosome)
        
        if self.size == 10:
            return self.size, generation + 1, best_fitness, self.max_fitness, best_chromosome
        else:
            return self.size, generation + 1, best_fitness, self.max_fitness, ''.join(best_chromosome)




size = 4
k = 100
mutation_probability = 0.05
generations = 1000

#ga = GeneticAlgorithm(size, k, mutation_probability, generations)

#ga = GeneticAlgorithm(size, k, mutation_probability, generations)

#ga.run_with_print()


Table = PrettyTable()

Table.field_names = ['Size of the Puzzle', 'Total Number Of Steps', 'Final Heuristic Value', 'Goal Heuristic Value', 'Solution']

for n in range(5, 11):
    ga = GeneticAlgorithm(n, k, mutation_probability, generations)
    size, generation, h_val, goal_val, solution = ga.run_with_return()
    Table.add_row([size, generation, h_val, goal_val, solution])

print(Table)
