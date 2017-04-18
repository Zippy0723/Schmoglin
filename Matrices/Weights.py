import numpy,pandas,ConfigParser,io

def CreateNewWeightArray(config,upperbound,lowerbound):
	cof = ConfigParser.SafeConfigParser()
	section = 'Data'
	
	if type(config) == type(''):
		cof.readfp(io.BytesIO(config))
	else:
		cof.read(config)
	
	nLayers = cof.getint(section,'nLayers')
	WeightDictonary = {}
	
	for i in xrange(1,nLayers):
		collumLength = cof.getint(section,'layer' + str(i+1))
		indexLength = cof.getint(section,'layer' + str(i))
		indexLabels = ['n' + str(x) for x in xrange(1,indexLength+1)]
		collumLabels = ['n' + str(x) for x in xrange(1,collumLength+1)]
		
		randWeights = numpy.random.uniform(lowerbound,upperbound,(indexLength,collumLength))
		dframe = pandas.DataFrame(data = randWeights,index=indexLabels,columns=collumLabels)
		WeightDictonary[i] = dframe
	return WeightDictonary