from numpy import *
from scipy import *
from operator import itemgetter
from random import sample
from ast import *
import matplotlib.pyplot as plt
from treeclass import *

## Import training data from fleet of engines
#######################################################
sensors=list(loadtxt('train_FD001.txt'))

## Import test data from fleet of engines
#######################################################
tester=list(loadtxt('test_FD001.txt'))
truth=list(loadtxt('RUL_FD001.txt'))

## Declare a dictionary that will contain sensor
## data from all engines and dictionary containing
## test data.
#######################################################
sensor_dict = {}
test_dict = {}
toy_dict = {}
toy_truth_dic = {}
toy_truth = 50*ones(100)

#######################################################
## Fill that dictionary in the following manner:
##
## Each index corresponds to a separate engine
## Each index contains an array of arrays, corresponding
## to data from each sensor
##
## Array 0 is engine number
## Array 1 is time
## Array 2-4 are operational settings
## Array 5-25 are sensor readings
########################################################
engine_number=0
sensor_dict[engine_number]=[]
for k in range(len(sensors[0])):
	sensor_dict[engine_number].append([])
for i in range(len(sensors)-1):
	for j in range(len(sensors[0])):
		sensor_dict[engine_number][j].extend([sensors[i][j]])
	if sensors[i][0]!=sensors[i+1][0]:
		engine_number+=1
		sensor_dict[engine_number]=[]
		for l in range(len(sensors[0])):
			sensor_dict[engine_number].append([])

engine_number=0
test_dict[engine_number]=[]
for k in range(len(tester[0])):
	test_dict[engine_number].append([])
for i in range(len(tester)-1):
	for j in range(len(tester[0])):
		test_dict[engine_number][j].extend([tester[i][j]])
	if tester[i][0]!=tester[i+1][0]:
		engine_number+=1
		test_dict[engine_number]=[]
		for l in range(len(tester[0])):
			test_dict[engine_number].append([])

for i in range(100):
	toy_dict[i]=[]
	toy_truth_dic[i]=[]
	for j in range(26):
		toy_dict[i].append([])
		toy_truth_dic[i].append([])
		if j == 3:
			toy_dict[i][j] = list(arange(100))
			toy_truth_dic[i][j] = list(arange(50))
		else:
			toy_dict[i][j] = list(arange(100))
			toy_truth_dic[i][j] = list(arange(50))


########################################################
## Create a dictionary of operators from which genetic
## program will contruct functions.
########################################################
operator_dict={0: '+', 1: '-', 2: '*', 3: '/', 4: '**', 5: 'sin',\
6: 'cos', 7: 'exp', 8: 'noise', 9: 'windowstd', 10: 'std',\
 11: 'windowsecondgrad', 12: 'windowgrad',\
  13: 'secondgrad', 14: 'grad'}

########################################################
## Add sensors, time, and operational settings to operator
## dictionary
########################################################
ref = len(operator_dict)
for i in arange(len(operator_dict),len(operator_dict)+len(sensor_dict[0])):
	eng='sense'+str(i-ref)
	operator_dict[i]=eng
		
















