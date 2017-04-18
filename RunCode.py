from multiprocessing.connection import Client
import numpy as np
import pandas as pd
import time,multiprocessing,win32api,win32process,win32con,Serial,copy,psutil,cPickle

def Worker(Weights,LrQueues,Blocks,FirstBlock,InterfacerQueue,NumInputs,FuncAliases,TerminationCommander):
	BlockCosts = {
		1:float('inf'),
		2:float('inf'),
		3:float('inf'),
		4:float('inf'),
	}
	BlockNum = FirstBlock
	ReverseWeights = copy.deepcopy(Weights)
	LastTotalCost = float('inf')
	
	Serial.ActivationFunction = getattr(Serial,FuncAliases['afunc'])
	Serial.ActivationDerivitive = getattr(Serial,FuncAliases['ader'])
	Serial.RegressionFunction = getattr(Serial,FuncAliases['rfunc'])
	Serial.RegressionDerivitve = getattr(Serial,FuncAliases['rder'])
	Serial.CostFunction = getattr(Serial,FuncAliases['cfunc'])
	Serial.CostDerivitive = getattr(Serial,FuncAliases['cder'])
	PID = win32api.GetCurrentProcessId()
	handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, PID)
	win32process.SetPriorityClass(handle, win32process.REALTIME_PRIORITY_CLASS)
	
	while True:
		Block = Blocks[BlockNum]
		LearningRate = LrQueues[BlockNum].get()
		SumCost = 0
		for example in Block.values:
			input = example[:NumInputs]
			expected = example[NumInputs:]
			NetState,UnregressedOutput = Serial.FeedForward(input,Weights)
			Observed = NetState[len(NetState)]
			SumCost += Serial.QuadraticCostFunction(Observed,expected)
			DeltaWeight = Serial.BackPropigate(NetState,UnregressedOutput,expected,Weights,LearningRate)
			Weights = Serial.ApplyChanges(Weights,DeltaWeight)
		
		if SumCost < BlockCosts[BlockNum]:
			LrQueues[BlockNum].put(LearningRate*1.01)
			BlockCosts[BlockNum] = SumCost
		else:
			LrQueues[BlockNum].put(LearningRate*.99)
			
		if BlockNum == 4:
			BlockNum = 1
		else:
			BlockNum += 1
			
		if BlockNum == FirstBlock:
			TotalCost = sum(BlockCosts.values())
			if TotalCost > LastTotalCost:
				Weights = copy.deepcopy(ReverseWeights)
			else:
				for k,w in Weights.iteritems():
					Weights[k] = w * .999
				ReverseWeights = copy.deepcopy(Weights)
				LastTotalCost = TotalCost
			InterfacerQueue.put([FirstBlock,TotalCost])
			
		if not TerminationCommander.empty():
			toPICKLE = open('ParamMatrix' + str(FirstBlock),'wb')
			cPickle.dump(ReverseWeights,toPICKLE)
			break
	
def Interfacer(InterfacerQueue,TerminationCommander):
	import os
	CLS = lambda: os.system('cls')
	Epoch = 0
	
	PID = win32api.GetCurrentProcessId()
	handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, PID)
	win32process.SetPriorityClass(handle, win32process.REALTIME_PRIORITY_CLASS)
	
	while True:
		GOT = []
		while len(GOT) < 4:
			GOT.append(InterfacerQueue.get())
		AllCosts = {}
		for n in GOT:
			AllCosts[n[0]] = n[1]
			
		Done = [True for x in AllCosts.itervalues() if x <= 80]
		CLS()
		print 'Alpha: ' + str(AllCosts[1])
		print 'Beta: ' + str(AllCosts[2])
		print 'Gamma: ' + str(AllCosts[3])
		print 'Delta: ' + str(AllCosts[4])
		print 'Epoch: ' + str(Epoch)
		Epoch += 1
		
		if any(Done):
			TerminationCommander.put(1)
			break
			
if __name__ == '__main__':
	adress = ('localhost',1500)
	con = Client(adress)
	Information = con.recv()
	con.close()
	
	Weights = Information[0]
	DataSet = Information[1]
	FuncAliases = Information[2]
	NumInputs = len(filter(lambda label: label.startswith('IN'),DataSet.columns))
	
	PID = win32api.GetCurrentProcessId()
	handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, PID)
	win32process.SetPriorityClass(handle, win32process.REALTIME_PRIORITY_CLASS)
	
	LRQUEUE_BlockOne = multiprocessing.Queue()
	LRQUEUE_BlockTwo = multiprocessing.Queue()
	LRQUEUE_BlockThree = multiprocessing.Queue()
	LRQUEUE_BlockFour = multiprocessing.Queue()
	
	LrQueues = {
		1:LRQUEUE_BlockOne,
		2:LRQUEUE_BlockTwo,
		3:LRQUEUE_BlockThree,
		4:LRQUEUE_BlockFour,
	}

	blocklst = np.array_split(DataSet,4)
	lbls = [1,2,3,4]
	Blocks = dict(zip(lbls,blocklst))
	
	InterfacerQueue = multiprocessing.Queue(4)
	TerminationCommander = multiprocessing.Queue(1)
	
	Alpha = multiprocessing.Process(target=Worker,args=(Weights,LrQueues,Blocks,1,InterfacerQueue,NumInputs,FuncAliases,TerminationCommander))
	Alpha.start()
	
	Beta = multiprocessing.Process(target=Worker,args=(Weights,LrQueues,Blocks,2,InterfacerQueue,NumInputs,FuncAliases,TerminationCommander))
	Beta.start()
	
	Gamma = multiprocessing.Process(target=Worker,args=(Weights,LrQueues,Blocks,3,InterfacerQueue,NumInputs,FuncAliases,TerminationCommander))
	Gamma.start()
	
	Delta = multiprocessing.Process(target=Worker,args=(Weights,LrQueues,Blocks,4,InterfacerQueue,NumInputs,FuncAliases,TerminationCommander))
	Delta.start()
	
	StartTime = time.clock()
	for i in xrange(1,5):
		LrQueues[i].put(0.001)
		
	InterfacerThread = multiprocessing.Process(target=Interfacer,args=(InterfacerQueue,TerminationCommander))
	InterfacerThread.start()
	parent = psutil.Process()
	parent.nice(psutil.REALTIME_PRIORITY_CLASS)
	for child in parent.children():
		child.nice(psutil.REALTIME_PRIORITY_CLASS)
	win32process.SetPriorityClass(handle, win32process.IDLE_PRIORITY_CLASS)
	InterfacerThread.join()
	
	print 'DONE IN: ' + str(time.clock() - StartTime)