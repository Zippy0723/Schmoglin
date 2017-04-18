import numpy,pandas,numba,math
'''
This module contains seralizable jitable, non abstarct calc functions. Sometimes python will Crash with cache is enabled on too many of them
I don't really know why, cacheing is not critical to operation

There can be no inline function deffenitions or else jit will break. There can be not list comprehension statements either.
Therefore you will see no map statements or lambda definitions. This might seem counter intutitive
but JIT makes using for loops for this actions much, much faster than base python
'''

#ALIASES
ActivationFunction = None
ActivationDerivitive = None
RegressionFunction = None
RegressionDerivitve = None
CostFunction = None
CostDerivitive = None

@numba.guvectorize(['void(float64[:],float64[:,:],float64[:,:])'],'(x),(x,y)->(x,y)',nopython=True,fastmath=True) #This does magical GIL magic, let's not touch it
def MultiplyByWeightVector(Values,Weights,Result):
	for i in xrange(Values.size):
		Result[i] = Values[i] * Weights[i]
		
def SumResults(MultArray):
	return numpy.sum(MultArray,axis=0)
	
@numba.vectorize(cache=True,nopython=True,fastmath=True)
def SigmoidFunction(Values):
	Result = math.e**(-1*Values)
	Result += 1
	Result = 1/Result
	return Result
	
@numba.jit(cache=True,nopython=True,fastmath=True) #This won't vectorize for some reason, will fix later maybe
def SoftmaxFunction(Values):
	Values = math.e**Values
	summation = Values.sum()
	coefficant = 1/summation
	return Values * coefficant
	
def FeedForward(Input,WeightDataFrame):
	nLayers = len(WeightDataFrame.keys()) #This does not include the input layer, and that dosn't matter
	lstLayerFinal = Input
	nsdict = {}
	nsdict[1] = lstLayerFinal
	for layer in xrange(1,nLayers):
		Weights = numpy.array(WeightDataFrame[layer])
		Res = MultiplyByWeightVector(lstLayerFinal,Weights)
		summed = SumResults(Res)
		lstLayerFinal = ActivationFunction(summed)
		nsdict[layer+1] = lstLayerFinal
		
	Weights = numpy.array(WeightDataFrame[nLayers])
	Res = MultiplyByWeightVector(lstLayerFinal,Weights)
	summed = SumResults(Res)
	FinalReg = RegressionFunction(summed)
	nsdict[nLayers+1]  = FinalReg
	
	'''
	Implementation/devnote/thingy that exists in case I ever release this program since this is like the tenth rewrite:
	If any of the values in FinalReg become a 1, or become so close to 1 that they eventually get rounded down into
	1, the program will break and start yielding nan values, just due to the way the softmax function works there
	is no way around this. It would be possible to impliment a check here to just stop the program if a value ever reached
	1, but i decided againts it because on large, real datasets and with a variable learning rate, it is incredibly unlikley that the network
	will ever become efficent enough to yield a 1 during softmax, and if statements that are checked as much as this one
	would add too much overhead for how fast this program needs to compute
	'''
		
	return (nsdict,summed)
	
@numba.jit(cache=True,nopython=True,fastmath=True) #GUvectorize can't return a scalar, well, it might be able to but I can't get it to do so.
def QuadraticCostFunction(Observed,Expected):
	Vec = Observed - Expected
	Vec = Vec **2
	Cost = Vec.sum()
	return Cost * 0.5
	
@numba.vectorize(cache=True,nopython=True,fastmath=True)
def QuadraticCostDerivitive(Observed,Expected):
	Vec=-(Observed-Expected)
	return Vec
	
@numba.vectorize(cache=True,nopython=True,fastmath=True)
def SigmoidDerivitive(Value): #There is a slight issue here if I ever do a different network config with this project, in that this can also be a regression function but it does not take the same amount of arguments as Softmax. Adding dummy args might break JIT as well
	return Value * (1-Value)
	
def SoftmaxDerivitive(Observed,Expected,OrignalValues):
	intendedClass = numpy.where(Expected==1)
	intendedClass = numpy.asscalar(intendedClass[0])
	out = numpy.zeros(Observed.shape)
	for obs in xrange(Observed.size):
		out[obs] = -(OrignalValues[obs] * (obs==intendedClass) - Observed[obs])
	return out
	
def ComputeHiddenError(WeightFrame,OutputError): #This function is less jitable than its brothers and gets run in object mode, will change if it causes slowdowns, it might
	Transposition = WeightFrame.T  
	index = Transposition.index
	WeightArray = numpy.array(Transposition)
	for i in xrange(len(index)):
		WeightArray[i] *= OutputError[i]
	summed = SumResults(WeightArray)
	return summed
	
def ComputeWeightChanges(WeightDict,ErrorDict,NetState): #This function assumes that Errordict has already been multiplied by the partial dervs
	DeltaDict = {}
	for i in WeightDict.keys():
		Weights = numpy.zeros(WeightDict[i].shape)
		DeltaDict[i] = Weights
	for i in WeightDict.keys():
		ActivationValues = NetState[i]
		ForwardErrors = ErrorDict[i+1]
		toChange = DeltaDict[i].T
		for node in xrange(ForwardErrors.size):
			toChange[node] = ForwardErrors[node] * ActivationValues
		DeltaDict[i] = toChange.T
	return DeltaDict
		
def BackPropigate(NetState, originalLast, Expected,WeightDict,learningRate):
	lastLayer = max(NetState.keys())
	observedRegression = NetState[lastLayer]
	ErrorGradient = CostDerivitive(observedRegression,Expected)
	PartialDerivitives = {}
	for layer in xrange(2,len(NetState.keys())):
		PartialDerivitives[layer] = ActivationDerivitive(NetState[layer])
	PartialDerivitives[lastLayer] = RegressionDerivitve(observedRegression,Expected,originalLast)
	
	ErrorTerms = {}
	ErrorTerms[lastLayer] = PartialDerivitives[lastLayer] * ErrorGradient
	for layer in xrange(lastLayer-1,1,-1):
		layerBefore = layer +1
		Error = ComputeHiddenError(WeightDict[layer],ErrorTerms[layerBefore])
		ErrorTerms[layer] = Error * PartialDerivitives[layer]
	DeltaWeights = ComputeWeightChanges(WeightDict,ErrorTerms,NetState)
	for key,value in DeltaWeights.iteritems():
		DeltaWeights[key] = value * learningRate
	return DeltaWeights
	
def ApplyChanges(WeightDict,DeltaWeights): #I could use collections.counter() here but it might not jit and I don't want to waste import overhead
	for key,value in WeightDict.iteritems():
		WeightDict[key] = value + DeltaWeights[key] #There must be a sign inversion somewhere because every bloody page on these equations has this as a minus sign, but this program only works if this is a plus.
	return WeightDict

	