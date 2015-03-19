from numpy import *
from scipy import *
from operator import itemgetter
from random import sample
import matplotlib.pyplot as plt
from copy import deepcopy
import pylab
from mpl_toolkits.mplot3d import Axes3D
from finch import *
import os

#########################################################################
## Create the person class, which are the objects that represent individuals
## in the population. A person should be able to evaluate its value, its
## depth, its fitness, and it should be able to mutate/crossover with other
## individuals in the population.
##
## The tree has operators +, -, *, /, **, sin, cos, exp, and noise (where
## noise adds Gaussian smear which has the ability to be time dependent).
## The leaf of each tree will either be a constant or a sensor.
#########################################################################
class person(object):
	def __init__(self, root = None, left = None, right = None, age = 1, target = 0, predictability = [], engine_rank=[], uniqueness = 0, dominated_individuals = [], dominating_individuals = 0, tier_rank = 100, worst_prediction = 100000000000000000000, targetlist = [], predictionlist = [], stddiff = 10000):
		## Initialize every person with a root, a left branch, a right
		## branch, and an age.
		self.root = root
		self.left = left
		self.right = right
		self.age = age
		self.target = target
		self.predictability = predictability
		self.engine_rank = engine_rank
		self.uniqueness = uniqueness
		self.dominated_individuals=dominated_individuals
		self.dominating_individuals = dominating_individuals
		self.tier_rank = tier_rank
		self.worst_prediction = worst_prediction
		self.targetlist = targetlist
		self.predictionlist = predictionlist
		self.stddiff = stddiff

	## The evaluate function is standalone. It depends on the operator dictionary and
	## sensor dictionary.
	def evaluate(self,engine,dic=sensor_dict):
		## Recursively loop through the person, return the array obtained by symbolic
		## regression. The evaluate function does not use all of the time information
		## for each sensor, but instead only looks to a random length of time. Its 
		## fitness depends on how well it can guess the amount of time that was left
		## out.
		#distance_in = int(len(dic[engine][0])/1.5)#random.randint(len(dic[engine][0])/2,len(dic[engine][0]))#
		#self.target = len(dic[engine][0]) - distance_in
		try:
			if self.root == operator_dict[0]:
				return array(self.left.evaluate(engine)) + array(self.right.evaluate(engine))
			elif self.root == operator_dict[1]:
				return array(self.left.evaluate(engine)) - array(self.right.evaluate(engine))
			elif self.root == operator_dict[2]:
				return array(self.left.evaluate(engine)) * array(self.right.evaluate(engine))
			elif self.root == operator_dict[3]:
				return array(self.left.evaluate(engine)) / array(self.right.evaluate(engine))
			elif self.root == operator_dict[4]:
				return array(self.left.evaluate(engine)) ** array(self.right.evaluate(engine))
			elif self.root == operator_dict[5]:
				return array(self.left.evaluate(engine)) * array(sin(self.right.evaluate(engine)))
			elif self.root == operator_dict[6]:
				return array(self.left.evaluate(engine)) * array(cos(self.right.evaluate(engine)))
			elif self.root == operator_dict[7]:
				return array(self.left.evaluate(engine)) * array(exp(self.right.evaluate(engine)))
			elif self.root == operator_dict[8]:
				noise = []
				for i in array(self.right.evaluate(engine)):
					noise.extend(random.normal(1,abs(i)+0.0001,1))
				return array(self.left.evaluate(engine)) * array(noise)
			elif self.root == operator_dict[9]:
				return array(self.left.evaluate(engine)) * std(array(self.right.evaluate(engine)[-5:]))
			elif self.root == operator_dict[10]:
				return array(self.left.evaluate(engine)) * std(array(self.right.evaluate(engine)))
			elif self.root == operator_dict[11]:
				return array(self.left.evaluate(engine)) * gradient(gradient(array(self.right.evaluate(engine)[-5:])))
			elif self.root == operator_dict[12]:
				return array(self.left.evaluate(engine)) * gradient(array(self.right.evaluate(engine)[-5:]))
			elif self.root == operator_dict[13]:
				return array(self.left.evaluate(engine)) * gradient(gradient(array(self.right.evaluate(engine))))
			elif self.root == operator_dict[14]:
				return array(self.left.evaluate(engine)) * gradient(array(self.right.evaluate(engine)))

			for i in arange(15,len(operator_dict)):
				if self.root == operator_dict[i]:
					return array(dic[engine][i-15])#[:distance_in])

			else:
				return self.root

		except:
			return array([nan])


	## depth(), find_branch(), and mother_chromosome() are all helper functions for
	## crossover()
	def depth(self):
		## Return the depth of the tree
		left_depth = self.left.depth() if self.left else 0
		right_depth = self.right.depth() if self.right else 0
		return max(left_depth, right_depth) + 1

	def find_branch(self,depth):
		## Return a branch at a particular depth
		level = 0
		if level==depth:
			dice = random.randint(1,3)
			if dice == 1:
				return self.right
			else:
				return self.left
		else:
			level+=1
			dice = random.randint(1,3)
			if dice == 1:
				try:
					return self.left.find_branch(depth)
				except:
					pass
				try:
					return self.right.find_branch(depth)
				except:
					pass
				return person(self.root)
			if dice == 2:
				try:
					return self.right.find_branch(depth)
				except:
					pass
				try:
					return self.left.find_branch(depth)
				except:
					pass
				return person(self.root)

	def mother_chromosome(self,depth):
		## Returns the path to the branch that will be replaced
		## on the mother's side
		chromosome = ['child']
		level = 0
		if level==depth:
			dice = random.randint(1,3)
			if dice == 1:
				chromosome.extend(['.right'])
				return ''.join(chromosome)
			else:
				chromosome.extend(['.left'])
				return ''.join(chromosome)
		else:
			level+=1
			dice = random.randint(1,3)
			if dice == 1:
				try:
					chromosome.extend(['.left'])
					return self.left.mother_chromosome(depth)
				except:
					pass
				try:
					chromosome.extend(['.right'])
					return self.right.mother_chromosome(depth)
				except:
					pass
				return ''.join(chromosome)
			if dice == 2:
				try:
					chromosome.extend(['.right'])
					return self.right.mother_chromosome(depth)
				except:
					pass
				try:
					chromosome.extend(['.left'])
					return self.left.mother_chromosome(depth)
				except:
					pass
				return ''.join(chromosome)

	def crossover(self,spouse):
		## Takes the root of the mother and replaces a branch with that of the
		## father. The child inherits the age of the oldest parent.
		#max_depth = min(self.depth(),spouse.depth())
		mother_depth = random.randint(1,self.depth())
		father_depth = random.randint(1,spouse.depth())
		#depth = random.randint(1,max_depth)
		child = deepcopy(self)
		father = '=spouse.find_branch(father_depth)'
		mother = child.mother_chromosome(mother_depth)
		DNA = mother+father
		exec(DNA)
		inherited_age = max(self.age,spouse.age)
		child.age = inherited_age
		return child	


	def tester(self,engine,dic):
		## Same as evaluate, but it takes a specified test data set for comparison
		try:
			if self.root == operator_dict[0]:
				return array(self.left.evaluate(engine)) + array(self.right.evaluate(engine))
			elif self.root == operator_dict[1]:
				return array(self.left.evaluate(engine)) - array(self.right.evaluate(engine))
			elif self.root == operator_dict[2]:
				return array(self.left.evaluate(engine)) * array(self.right.evaluate(engine))
			elif self.root == operator_dict[3]:
				return array(self.left.evaluate(engine)) / array(self.right.evaluate(engine))
			elif self.root == operator_dict[4]:
				return array(self.left.evaluate(engine)) ** array(self.right.evaluate(engine))
			elif self.root == operator_dict[5]:
				return array(self.left.evaluate(engine)) * array(sin(self.right.evaluate(engine)))
			elif self.root == operator_dict[6]:
				return array(self.left.evaluate(engine)) * array(cos(self.right.evaluate(engine)))
			elif self.root == operator_dict[7]:
				return array(self.left.evaluate(engine)) * array(exp(self.right.evaluate(engine)))
			elif self.root == operator_dict[8]:
				noise = []
				for i in array(self.right.evaluate(engine)):
					noise.extend(random.normal(1,abs(i)+0.0001,1))
				return array(self.left.evaluate(engine)) * array(noise)
			elif self.root == operator_dict[9]:
				return array(self.left.evaluate(engine)) * std(array(self.right.evaluate(engine)[-5:]))
			elif self.root == operator_dict[10]:
				return array(self.left.evaluate(engine)) * std(array(self.right.evaluate(engine)))
			elif self.root == operator_dict[11]:
				return array(self.left.evaluate(engine)) * gradient(gradient(array(self.right.evaluate(engine)[-5:])))
			elif self.root == operator_dict[12]:
				return array(self.left.evaluate(engine)) * gradient(array(self.right.evaluate(engine)[-5:]))
			elif self.root == operator_dict[13]:
				return array(self.left.evaluate(engine)) * gradient(gradient(array(self.right.evaluate(engine))))
			elif self.root == operator_dict[14]:
				return array(self.left.evaluate(engine)) * gradient(array(self.right.evaluate(engine)))

			for i in arange(15,len(operator_dict)):
				if self.root == operator_dict[i]:
					return array(dic[engine][i-15])

			else:
				return self.root

		except:
			return array([nan])


	## mutate is standalone.
	def mutate(self,operator_mutation_rate=0,constant_mutation_rate=0.1,variable_intro_rate=0.1,variable_outro_rate=0,extension_rate=0,max_constant=200,maximum_depth=10):
		if self.root==operator_dict[0] or self.root==operator_dict[1] or self.root==operator_dict[2]\
		or self.root==operator_dict[3] or self.root==operator_dict[4] or self.root==operator_dict[5]\
		or self.root==operator_dict[6] or self.root==operator_dict[7] or self.root==operator_dict[8]\
		or self.root==operator_dict[9] or self.root==operator_dict[10] or self.root==operator_dict[11]\
		or self.root==operator_dict[12] or self.root==operator_dict[13] or self.root==operator_dict[14]:
			dice = random.random()
			if dice < operator_mutation_rate:
				op = random.randint(0,15)
				self.root = operator_dict[op]
				self.left.mutate(operator_mutation_rate,\
					constant_mutation_rate,variable_intro_rate,variable_outro_rate,\
					extension_rate,max_constant,maximum_depth)
				self.right.mutate(operator_mutation_rate,\
					constant_mutation_rate,variable_intro_rate,variable_outro_rate,\
					extension_rate,max_constant,maximum_depth)
			else:
				self.left.mutate(operator_mutation_rate,\
					constant_mutation_rate,variable_intro_rate,variable_outro_rate,\
					extension_rate,max_constant,maximum_depth)
				self.right.mutate(operator_mutation_rate,\
					constant_mutation_rate,variable_intro_rate,variable_outro_rate,\
					extension_rate,max_constant,maximum_depth)
		else:
			if type(self.root)==float or self.root==operator_dict[15] or self.root==operator_dict[16] or self.root==operator_dict[17]\
			or self.root==operator_dict[18] or self.root==operator_dict[19] or self.root==operator_dict[20]\
			or self.root==operator_dict[21] or self.root==operator_dict[22] or self.root==operator_dict[23]\
			or self.root==operator_dict[24] or self.root==operator_dict[25] or self.root==operator_dict[26]\
			or self.root==operator_dict[27] or self.root==operator_dict[28] or self.root==operator_dict[29]\
			or self.root==operator_dict[30] or self.root==operator_dict[31] or self.root==operator_dict[32]\
			or self.root==operator_dict[33] or self.root==operator_dict[34] or self.root==operator_dict[35]\
			or self.root==operator_dict[36] or self.root==operator_dict[37] or self.root==operator_dict[38]\
			or self.root==operator_dict[39] or self.root==operator_dict[40]:
				dice = random.random()
				dice1 = random.random()
				dice2 = random.random()
				if dice < constant_mutation_rate:
					const = random.uniform(-max_constant,max_constant)
					self.root=const
				if dice1 < variable_intro_rate:
					sense = random.randint(17,41)
					self.root=operator_dict[sense]
				if dice2 < extension_rate and self.depth()<maximum_depth:
					op = random.randint(0,15)
					self.root = operator_dict[op]
					self.right = person(random.uniform(0,max_constant))
					self.left = person(random.uniform(0,max_constant))
			if self.root==operator_dict[15] or self.root==operator_dict[16] or self.root==operator_dict[17]\
			or self.root==operator_dict[18] or self.root==operator_dict[19] or self.root==operator_dict[20]\
			or self.root==operator_dict[21] or self.root==operator_dict[22] or self.root==operator_dict[23]\
			or self.root==operator_dict[24] or self.root==operator_dict[25] or self.root==operator_dict[26]\
			or self.root==operator_dict[27] or self.root==operator_dict[28] or self.root==operator_dict[29]\
			or self.root==operator_dict[30] or self.root==operator_dict[31] or self.root==operator_dict[32]\
			or self.root==operator_dict[33] or self.root==operator_dict[34] or self.root==operator_dict[35]\
			or self.root==operator_dict[36] or self.root==operator_dict[37] or self.root==operator_dict[38]\
			or self.root==operator_dict[39] or self.root==operator_dict[40]:
				dice = random.random()
				if dice < variable_outro_rate:
					self.root = random.uniform(0,max_constant)
				else:
					pass

	def predictionFitness(self,engine,dic=sensor_dict):
		## Determines how close the evolved function is to the true remaining lifetime
		## of the engine # stop
		genotype = self.evaluate(engine,dic)
		try:
			if type(genotype)!=ndarray or \
			len(genotype)<5 or isnan(genotype).any() or isinf(genotype).any():
				self.predictability.extend([10000000000000000000000000000000])

			else:
				self.predictability.extend([mean(abs(abs(genotype - sorted(dic[engine][1],reverse=True))))])
		except:
			self.predictability.extend([10000000000000000000000000000000])

	
	def isDominant(self,other):
		if (self.age <= other.age and self.uniqueness <= other.uniqueness and self.predictability <= other.predictability and self.worst_prediction <= other.worst_prediction)\
		and (self.age < other.age or self.uniqueness < other.uniqueness or self.predictability < other.predictability or self.worst_prediction < other.worst_prediction):
			return True	
		else:
			return False

	def hasLeftBranch(self):
		try:
			a=self.left.root
			return True
		except:
			return False

	def hasRightBranch(self):
		try:
			a = self.right.root
			return True
		except:
			return False


	def isEquivalent(self,other):
		if (not self.hasLeftBranch()) and (not other.hasLeftBranch()):
			return (self.root == other.root)
		else:
			return self.root == other.root and self.left.isEquivalent(other.left) and self.right.isEquivalent(other.right)

	def __str__(self, depth=0):
		ret = ""

		# Print right branch
		if self.hasRightBranch():
			ret += self.right.__str__(depth + 1)
	
		# Print own value
		ret += "\n" + ("    "*depth) + str(self.root)
	
		# Print left branch
		if self.hasLeftBranch():
			ret += self.left.__str__(depth + 1)
	
		return ret




#########################################################################
## Create the population class. This is analagous to the town that holds
## all of the objects of the person class. The population class has the
## ability to rank its citizens according to fitness, to breed its citizens
## to mutate its citizens, and to evolve its population.
#########################################################################
class population(object):
	def __init__(self):
		## The population object contains a list of people
		self.people = []
		self.fitnesses = []
		self.front_dict={}
		self.evaluations=0
		self.evaluation_list = [0]
		self.winner_list = [100]
		self.winner=10000000

	def add_person(self,person):
		## Function that adds a person to the population.
		self.people.extend([person])

	def gatherTraits(self, engines = arange(0,len(sensor_dict)),dic=sensor_dict):
		## Gather the traits of each member of the population. These include
		## prediction strength, age, and engine preference
		for i in self.people:
			i.age+=1
			i.predictability = []
			i.engine_rank = []
			i.tier_rank = 100
			i.uniqueness = 0
			i.dominating_individuals = 0
			i.dominated_individuals = []
			for j in engines:
				i.predictionFitness(j,dic)
				self.evaluations+=1
			engine_preference = sorted(zip(i.predictability,engines),key=itemgetter(0))
			i.engine_rank = map(lambda engine_preference: engine_preference[1],engine_preference)
			i.predictability = mean(map(lambda engine_preference: engine_preference[0],engine_preference))#/len(i.engine_rank))*100
			i.worst_prediction = map(lambda engine_preference: engine_preference[0],engine_preference)[-1]
			for k in self.people:
				i.uniqueness += len(list(set(i.engine_rank).intersection(k.engine_rank)))


	def rank(self):
		## I want members of the population that can predict well, can predict
		## on engines that other members of the population cannot, and that are
		## young. Implement a pareto fitness based on these three characteristics.
		front_number = 0
		self.front_dict = {}
		self.front_dict[front_number] = []
		ranked_population = []


		for i in self.people:
			for j in self.people:
				if (i.age <= j.age and i.uniqueness <= j.uniqueness and i.predictability <= j.predictability and i.worst_prediction <= j.worst_prediction)\
				and (i.age < j.age or i.uniqueness < j.uniqueness or i.predictability < j.predictability or i.worst_prediction < j.worst_prediction):# or i.stddiff < j.stddiff):
					i.dominated_individuals.extend([j])

				elif (i.age >= j.age and i.uniqueness >= j.uniqueness and i.predictability >= j.predictability and i.worst_prediction >= j.worst_prediction)\
				and (i.age > j.age or i.uniqueness > j.uniqueness or i.predictability > j.predictability or i.worst_prediction > j.worst_prediction):# or i.stddiff > j.stddiff):
					i.dominating_individuals+=1

				else:
					continue

			if i.dominating_individuals == 0:
				self.front_dict[front_number].extend([i])
				i.tier_rank = front_number

		while len(self.front_dict[front_number])!=0:
			front_number+=1
			self.front_dict[front_number]=[]
			for i in self.front_dict[front_number-1]:
				for j in i.dominated_individuals:
					if j.dominating_individuals-1==0:
						self.front_dict[front_number].extend([j])
						j.tier_rank = front_number

		for i in range(front_number-1):
			members = []
			values = []
			for j in self.front_dict[i]:
				members.extend([j])
				values.extend([j.predictability])
			ziplist = sorted(zip(values,members),key=itemgetter(0))
			self.front_dict[i] = map(lambda ziplist: ziplist[1],ziplist)

		ranked_population = []
		for i in range(101):
			sortedpop = []
			sortedpopscore = []
			for j in self.people:
				if j.tier_rank == i:
					#ranked_population.extend([j])
					sortedpop.extend([j])
					sortedpopscore.extend([j.predictability])
			ranker = sorted(zip(sortedpopscore,sortedpop),key=itemgetter(0))
			sortedpop = map(lambda ranker: ranker[1],ranker)
			if len(sortedpop)!=0:
				ranked_population.extend(sortedpop)

		self.people = ranked_population
		temp = self.winner
		self.winner = ranked_population[0].predictability
		if self.winner != temp:
			os.system('say "New Best Solution"')
		self.evaluation_list.extend([self.evaluations])
		self.winner_list.extend([self.winner])


	def select(self, elite_pressure, total_pressure, max_constant):
		## Elite pressure offers the use of elitism. The top x percentage of the
		## population will have guaranteed survival if elite_pressure is nonzero.
		## Total pressure is the total number of members of the population that
		## survive to breed.
		parent_population = []

		elite = int(elite_pressure * len(self.people))
		total = int(total_pressure * len(self.people))
		remaining = total - elite

		parent_population.extend(self.people[0:elite])

		for i in range(int(.8 * remaining)):
			parent = random.randint(elite,int(0.6*len(self.people)))
			parent_population.extend([self.people[parent]])
		for i in range(int(.2 * remaining)):
			citizen=person()
			citizen.root = operator_dict[random.randint(0,15)]
			decider = random.random()
			if decider < 0.5:
				citizen.left = person(random.uniform(-max_constant,max_constant))
				citizen.right = person(operator_dict[random.randint(17,41)])
			if decider > 0.5:
				citizen.left = person(operator_dict[random.randint(17,41)])
				citizen.right = person(random.uniform(-max_constant,max_constant))
			for i in range(7):
				citizen.mutate(extension_rate = 0.05)
			parent_population.extend([citizen])
			del citizen
		self.people = parent_population

	def breed(self,population_size,operator_mutation_rate,constant_mutation_rate,variable_intro_rate,variable_outro_rate,extension_rate,max_constant,maximum_depth):
		## Breeds the parents to replenish the population with children.
		parents = len(self.people)
		num_children = population_size - parents
		growth = 0
		while growth < num_children:
			momdice = random.random()
			if momdice<0.5:
				mother_dex = random.randint(0,parents)
			else:
				mother_dex = random.randint(0,20)
			daddice = random.random()
			if daddice<0.5:
				father_dex = random.randint(0,parents)
			else:
				father_dex = random.randint(0,20)
			mother = self.people[mother_dex]
			father = self.people[father_dex]
			child = mother.crossover(father)
			child.mutate(operator_mutation_rate,constant_mutation_rate,\
                    variable_intro_rate,variable_outro_rate,extension_rate,max_constant,\
                    maximum_depth)

			equivalencies = 0
			test = child.evaluate(0)
			for j in [mother,father]:
				try:
					if test == j.evaluate(0):#child.isEquivalent(j):#
						print 'in the equivalenies loop'
						equivalencies+=1
						break
					else:
						continue
				except:
					equivalencies+=1

			if equivalencies == 0:
				self.add_person(child)
				growth+=1
			else:
				continue

	def evolve(self, generations=10000,engines=arange(0,len(sensor_dict)), elite_pressure=0.1, total_pressure=0.4, population_size=100,operator_mutation_rate=0,constant_mutation_rate=0.1,variable_intro_rate=0.05,variable_outro_rate=0,extension_rate=0,max_constant=100,maximum_depth=10,visualizer=False,dic=sensor_dict):
		if visualizer==True:
			plt.ion()
			fig = pylab.figure(figsize=(10,8))
			plt.show()

		for i in range(generations):
			mutate_rate = 0.5 - 0.5 * ((i%100)/100)
			self.fitnesses=[]
			print 'Generation Number: '+str(i)
			print 'Leading Predictability: ' + str(self.winner) +'\n'
			self.gatherTraits(engines,dic)
			self.rank()

			if visualizer==True:
				plt.clf()
				x = []
				y = []
				z = []
				for j in self.people:
					if j.tier_rank == 0:
						x.extend([j.age])
						y.extend([j.predictability])
						z.extend([j.uniqueness])
				ax = Axes3D(fig)
				ax.set_xlabel("Age (Length of Lineage)")
				ax.set_ylabel("Average Perc. Error (Training Set)")
				ax.set_zlabel("Uniqueness")
				plt.title("Pareto Front, Generation Number %s" %i)
				ax.scatter(x,y,z)
				plt.draw()
			
			self.select(elite_pressure, total_pressure, max_constant)
			self.breed(population_size,operator_mutation_rate,constant_mutation_rate,variable_intro_rate,variable_outro_rate,extension_rate,max_constant,maximum_depth)


	def hillclimber(self, generations=10000,engines=arange(0,len(sensor_dict)), operator_mutation_rate=0,constant_mutation_rate=0.1,variable_intro_rate=0.1,variable_outro_rate=0,extension_rate=0,max_constant=200,maximum_depth=10,dic=sensor_dict):
		for i in range(generations):
			mutate_rate = 0.5 - 0.5 * ((i%100)/100)
			self.fitnesses=[]
			print 'Generation Number: '+str(i)
			print 'Leading Predictability: ' + str(self.winner) +'\n'
			self.gatherTraits(engines,dic)
			self.rank()

			for j in self.people[2:]:
				j.mutate(operator_mutation_rate,constant_mutation_rate,\
                    variable_intro_rate,variable_outro_rate,extension_rate,max_constant,\
                    maximum_depth)

	def randomSearch(self,generations=10000,population_size=99,max_constant=200,engines=arange(0,len(sensor_dict)),dic=sensor_dict):
		for i in range(generations):
			print 'Generation Number: '+str(i)
			print 'Leading Predictability: ' + str(self.winner) +'\n'
			self.gatherTraits(engines,dic)
			self.rank()
			self.people = [self.people[0]]
			for j in range(population_size):
				citizen = person()
				citizen.root = operator_dict[random.randint(0,15)]
				decider = random.random()
				if decider < 0.5:
					citizen.left = person(random.uniform(-max_constant,max_constant))
					citizen.right = person(operator_dict[random.randint(17,41)])
				if decider > 0.5:
					citizen.left = person(operator_dict[random.randint(17,41)])
					citizen.right = person(random.uniform(-max_constant,max_constant))
				for k in range(7):
					citizen.mutate(extension_rate = 0.3)
				self.add_person(citizen)

########################################################
## Prognostic Methods for Comparison with Truth
########################################################
	def run_comparison(self,dic,truthdic,person_number):
		## Compares true remaining engine lifetime with predicted
		## engine lifetime
		engines = arange(0,len(dic))
		predictions = []
		for i in engines:
			predictions.extend([self.people[person_number].tester(i,dic)[-1]])
		predictions=abs(predictions)
		print predictions
		plt.plot(predictions,'r*',label="Predictions")
		plt.plot(truthdic,'g*',label="True Remaining Lifetime")
		error = abs(array(truthdic) - array(predictions))
		#print error
		print mean(error)
		plt.title("Estimated Remaining Lifetime vs. True Remaining Lifetime")
		plt.legend(loc=2)
		plt.show()




##########################################################################################
######################## END OF CLASS DEFINITIONS ########################################
##########################################################################################




########################################################
## Initialize the population
########################################################
def initialize_population(population_size=100,max_constant=200):
	town = population()
	for i in range(population_size):
		citizen = person()
		citizen.root = operator_dict[random.randint(0,15)]
		decider = random.random()
		if decider < 0.5:
			citizen.left = person(random.uniform(-max_constant,max_constant))
			citizen.right = person(operator_dict[random.randint(17,41)])
		if decider > 0.5:
			citizen.left = person(operator_dict[random.randint(17,41)])
			citizen.right = person(random.uniform(-max_constant,max_constant))
		town.add_person(citizen)
	town.people = town.people
	# for i in range(7):
	# 	for j in town.people[0:population_size/2]:
	# 		j.mutate(extension_rate = 0.3)
	# 	for k in town.people[-population_size/2:]:
	# 		k.mutate(extension_rate = 0.1)
	return town

def initialize_hc_population(population_size=100,max_constant=200):
	town = population()
	for i in range(population_size):
		citizen = person()
		citizen.root = operator_dict[random.randint(0,15)]
		decider = random.random()
		if decider < 0.5:
			citizen.left = person(random.uniform(-max_constant,max_constant))
			citizen.right = person(operator_dict[random.randint(17,41)])
		if decider > 0.5:
			citizen.left = person(operator_dict[random.randint(17,41)])
			citizen.right = person(random.uniform(-max_constant,max_constant))
		town.add_person(citizen)
	town.people = town.people
	for i in range(7):
		for j in town.people[0:population_size/2]:
			j.mutate(extension_rate = 0.3)
		for k in town.people[-population_size/2:]:
			k.mutate(extension_rate = 0.1)
	return town

def run_comparison(population,dic,truthdic,person_number):
	## Compares true remaining engine lifetime with predicted
	## engine lifetime
	engines = arange(0,len(dic))
	predictions = []
	for i in engines:
		predictions.extend([population.people[person_number].tester(i,dic)[-1]])
	predictions=abs(array(predictions))
	print predictions
	plt.plot(predictions,'r*',label="Predictions")
	plt.plot(truthdic,'g*',label="True Remaining Lifetime")
	error = abs(array(truthdic) - array(predictions))
	#print error
	print mean(error)
	plt.title("Estimated Remaining Lifetime vs. True Remaining Lifetime")
	plt.legend(loc=2)
	plt.show()

def plotter(pop1,pop2,pop3,plotlabel):
	plt.ion()
	length = len(pop1.winner_list)
	stepsize = 0
	for i in range(10,1000):
		if length%i==0:
			stepsize = i
			break
		else:
			continue

	means = []
	evals = []
	dev = []
	print stepsize

	for i in arange(0,length,stepsize):
		print i
		a=(sum(pop1.winner_list[i:i+stepsize]) + sum(pop2.winner_list[i:i+stepsize]) + sum(pop3.winner_list[i:i+stepsize]))/(stepsize*3)
		b=(sum(pop1.evaluation_list[i:i+stepsize]) + sum(pop2.evaluation_list[i:i+stepsize]) + sum(pop3.evaluation_list[i:i+stepsize]))/(stepsize*3)
		dev.extend([std(pop1.winner_list[i:i+stepsize] + pop2.winner_list[i:i+stepsize] + pop3.winner_list[i:i+stepsize])])
		errbar = array(dev)/sqrt(3)
		means.extend([a])
		evals.extend([b])

	print evals
	print means
	print errbar
	plt.errorbar(log10(evals),means,errbar/2.,fmt='',label=plotlabel)
	plt.legend(loc='upper right',prop={'size':8})
	# plt.ylim([4,30])
	plt.xlabel('log(evaluations)')
	plt.ylabel('Mean Prediction error')
	plt.title('Predictability in Training Data Set')
	plt.show()
	plt.savefig('bigplot_same_inits.png')



def go():
	town1=initialize_hc_population()
	town2=initialize_hc_population()
	town3=initialize_hc_population()

	town4=deepcopy(town1)
	town7=deepcopy(town1)
	town5=deepcopy(town2)
	town8=deepcopy(town2)
	town6=deepcopy(town3)
	town9=deepcopy(town3)

	town1.evolve(generations=1000)
	town2.evolve(generations=1000)
	town3.evolve(generations=1000)

	plotter(town1,town2,town3,"GP")

	town4.hillclimber(generations=1000)
	town5.hillclimber(generations=1000)
	town6.hillclimber(generations=1000)


	plotter(town4,town5,town6,"Hillclimber")

	town7.randomSearch(generations=1000)
	town8.randomSearch(generations=1000)
	town9.randomSearch(generations=1000)


	plotter(town7,town8,town9,"Random Search")






