import numpy as np
import multiprocessing as mp
from multiprocessing import Pool,Process#,shared_memory


class IsingMC:
	
	def __init__(self,Nsites=100,temp=[1],Nsteps=1000,eq_steps=50,lag=5,full_Zexp=False,graph=None):
		
		
		# number of sites
		self.Nsites=Nsites
		
		# number of MC steps
		self.Nsteps=1000
		
		# number of initial steps to wait to equilibriate
		self.eq_steps=eq_steps
		
		# collect data after lag steps
		self.lag=lag
		
		# temperature array
		self.temp=temp 
		
		# probability of flipping for each temperature (storing so that calls are faster)
		# also this is independent of the type of graph
		self.p_arr=[min(1,1-np.exp(-2*T)) for T in self.temp]
		
		# if Z expectation is requried for each MC step
		if(full_Zexp==True):
			self.Zexp=np.zeros((len(self.temp),Nsteps))
			
		else:
			self.Zexp=np.zeros(len(self.temp))
			
		self.status="model initialized but not run"
		
		# build square lattice graph (2D) by default
		if graph is None:
		
			self.graph=self.buildSquare()
			
			# update Nsites to nearest square
			self.Nsites=int(np.sqrt(Nsites))**2

			
		else:
			self.graph=graph
			self.Nsites=len(graph) # update number of sites if graph in case mismatch
	
	# build square lattice graph
			
	def buildSquare(self):
		G={}
		# setup square lattice
		L=int(np.sqrt(self.Nsites))
		
		# add nearest neighbors    
		for i in range(L):
			for j in range(L):
		    
				n=i*L+j
				
				G[n]=set()

				xp=(i+1)%L
				xm=(i-1)%L
				yp=(j+1)%L
				ym=(j-1)%L

				G[n]=set([xp*L+j,xm*L+j,i*L+yp,i*L+ym])
				
				
		
		return G

		
	def buildCube(self):
		G={}
		# setup square lattice
		L=int(np.pow(self.Nsites,1/3))
		
		# add the six nearest neighbors    
		for i in range(L):
			for j in range(L):
				for k in range(L):		    
					n=i*(L**2)+j*L+k
					
					G[n]=set()

					xp=(i+1)%L
					xm=(i-1)%L
					yp=(j+1)%L
					ym=(j-1)%L
					zp=(k+1)%L
					zm=(k-1)%L


					G[n]=set([xp*(L**2)+j*L+k,xm*(L**2)+j*L+k,i*(L**2)+yp*L+k,i*(L**2)+ym*L+k,i*(L**2)+j*L+zp,i*(L**2)+j*L+zm])
				
		
		return G
	

	# do a metropolis run for a given temperature, uses similar structure as the cluster algorithm
	def metropolis(self,t_ind):
		# t_ind is the index of the temperature 


		# initialize random configuration
		spins=np.random.choice([-1,1],size=self.Nsites)

		for n in range(self.Nsteps):

			# choose random site i
			i=np.random.randint(self.Nsites)

			# measure change in energy upon flipping
			deltaE=0
			for neighbor in self.graph[i]:			
				if(spins[i]*spins[neighbor]>0):
					deltaE+=2*self.temp[t_ind]
				
			# flip with MC probability
			if(np.random.rand()<min(1,np.exp(-deltaE))):
			    spins[i]*=-1

			# sample expectation value from current configuration if full data not required
			if(len(self.Zexp.shape)==1):
				if(n>self.eq_steps and n%self.lag==0):
				    self.Zexp[t_ind]+=np.abs(np.average(spins))
		    	# store expectation value at each step otherwise
			else:
				self.Zexp[t_ind,n]=np.average(spins)
			
		# normalize over Nsteps if only average expectation value desired
		#if(len(self.Zexp.shape)==1):
		#	self.Zexp/=((self.Nsteps-self.eq_steps)//self.lag)			

	
	
	# run for Nsteps for a given temperature
	def cluster(self,t_ind):
		# t_ind is the index of the temperature 

		print("Temperature ",self.temp[t_ind])
		# initialize random configuration
		spins=np.random.choice([-1,1],size=self.Nsites)

		for n in range(self.Nsteps):
			print("n here",n)
			print("graph",self.graph)
			# choose random site i
			i=np.random.randint(self.Nsites)

			# initalize bfs search with current site        
			curr_nodes=[i]

			# initialize the current boundary nodes in the cluster
			visited=set()        
			visited.add(i)

			# flip starting state
			spins[i]*=-1


			# bfs loop
			while(len(curr_nodes)>0):

				next_nodes=[]

				# search neighbors with opposite spins and flip with MC probability
				for node in curr_nodes:
					for neighbor in self.graph[node]:
						if((spins[neighbor]!=spins[i]) and (neighbor not in visited)):
							visited.add(neighbor)

						# add to cluset with MC prob
						if(np.random.rand()<self.p_arr[t_ind]):
							spins[neighbor]*=-1
							next_nodes.append(neighbor)
					    
				curr_nodes=next_nodes

			# sample expectation value from current configuration if full data not required
			if(len(self.Zexp.shape)==1):
				if(n>self.eq_steps and n%self.lag==0):
				    self.Zexp[t_ind]+=np.abs(np.average(spins))
		    	# store expectation value at each step otherwise
			else:
				self.Zexp[t_ind,n]=np.average(spins)

		# normalize over Nsteps if only average expectation value desired
		if(len(self.Zexp.shape)==1):
			self.Zexp/=(self.Nsteps-self.eq_steps)//lag

	def run(self,algo='cluster',n_workers=1):
		
		algo_fun={'cluster':self.cluster,'metropolis':self.metropolis} # currently supports 'cluster' and 'metropolis'
		
		if(algo not in algo_fun):
			raise ValueError("Only 'cluster' and 'metropolis' algorithms supported!")
		
		# run sequentially for all temperatures
		if(n_workers==1):
			
			for t_ind,temp in enumerate(self.temp):
				algo_fun[algo](t_ind)
				
			
			self.status= algo + " algorithm was performed successfully"
			
		else:
			n_workers=min(mp.cpu_count()-1,n_workers)
			pool=Pool(num_workers)
			shm = shared_memory.SharedMemory(create=True, size=self.Zexp.nbytes)

			shared_Zexp = np.ndarray(self.Zexp.shape, dtype=float, buffer=shm.buf)
			shared_Zexp[:] = self.Zexp[:]
			self.Zexp=shared_Zexp
			
			pool.starmap(self.run, range(len(self.temp)))

			shm.close()
			
			
if __name__=='__main__':
	
	
	Nsites=100
	Nsteps=10000
	lag=100
	eq_steps=100
	
	temp=np.linspace(0,1,10)
	
	model=IsingMC(Nsites=Nsites,temp=temp,Nsteps=Nsteps,eq_steps=eq_steps,lag=lag,full_Zexp=False,graph=None)
	
	
	num_workers=1
	model.run(algo='metropolis',n_workers=num_workers)
	print("expectation value", model.Zexp,model.Nsites,model.p_arr)
	
