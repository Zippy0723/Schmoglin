from subprocess import Popen
import os
from multiprocessing.connection import Listener
	
def RUNMAIN(StartingWeights, DataSet, FuncAliases):
	pth = os.path.abspath('Schmoglin')
	Proc = Popen('RUN.bat',cwd=pth,shell=True)
	
	adress = ('localhost',1500)
	lstnr = Listener(adress)
	con= lstnr.accept()
	
	Communication = [StartingWeights,DataSet,FuncAliases]
	
	con.send(Communication)
	con.close()
	lstnr.close()