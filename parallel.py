import multiprocessing
from train import train
def worker(num,name):
	print ('Worker: ' + str(num) + ', Game: ' + name) 
	train()	
	return
if __name__ == '__main__':
	jobs = []
	gameName = 'Breakout'
	instNum = 3
	for i in range(instNum):
		p = multiprocessing.Process(target=worker, args=(i,gameName))
		jobs.append(p)
		p.start()


