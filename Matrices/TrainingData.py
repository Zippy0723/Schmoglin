#This file mainly exists to load an excel spreadsheet into a parsable format.
#I could have the storage format for training data be something a bit less obtuse like a json file but this seems fine for now
import numpy,pandas,os
def ParseDataFromFile(SetName):
	path = 'Schmoglin/Martices/SavedTrainingData/' + SetName
	path = os.path.abspath(path)
	TrainingData = pandas.read_excel(
		io=path,sheetname=['INPUTS','EXPECTED'],header=0,index_col=0
	)
	Combined = pandas.concat([TrainingData['INPUTS'],TrainingData['EXPECTED']],axis=1,join_axes=[TrainingData['INPUTS'].index])
	return Combined