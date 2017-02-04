import multiprocessing
from train import train
#Change this to actual name
from main import nameMethod


def worker(num,name):
	print ('Worker: ' + str(num) + ', Game: ' + name) 
	train()		
	return
def nameWorker():
	print ('nameWorker') 
	return
if __name__ == '__main__':
	jobs = []
	gameName = 'Breakout'
	instNum = 3
	p2 = multiprocessing.Process(target=nameWorker)
	p2.start()
	for i in range(instNum):
		p = multiprocessing.Process(target=worker, args=(i,gameName))
		jobs.append(p)
		p.start()
	

