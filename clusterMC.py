import numpy as np
import multiprocessing as mp
from multiprocessing import Pool,Process
import time

def p_add(J):

    return min(1,1-np.exp(-2*J))


def mc_sample(G,N,J,nsteps,lag,eq_time):

    # intialize Z expectation value to be returned
    Z_exp=0

    spins=np.random.choice([-1,1],size=N)
    
    for n in range(nsteps):
    
        # choose random site i
        i=np.random.randint(N)

        # initalize bfs search with current site        
        curr_nodes=[i]

        # initialize the current boundary nodes in the cluster
        visited=set()        
        visited.add(i)

        # flip starting state
        spins[i]=-1*spins[i]
        

        # bfs loop
        while(len(curr_nodes)>0):
            
            next_nodes=[]

            # search neighbors with opposite spins and flip with MC probability
            for node in curr_nodes:
                for neighbor in G[node]:
                    if((spins[neighbor]!=spins[i]) and (neighbor not in visited)):
                        visited.add(neighbor)
                        
                        if(np.random.rand()<p_add(J)):
                            spins[neighbor]=-1*spins[neighbor]
                            next_nodes.append(neighbor)
                            
            curr_nodes=next_nodes

        # sample expectation value from current configuration
        if(n>eq_time and n%lag==0):
            Z_exp+=np.abs(np.average(spins))
    print("J = ",J,Z_exp/((nsteps-eq_time)//lag),flush=True)        
    return Z_exp/((nsteps-eq_time)//lag)


if __name__=='__main__':


    L=16
    N=L**2
    
    G={}
    # setup square lattice    
    for i in range(L):
        for j in range(L):
            
            n=i*L+j
            G[n]=set()
                
            xp=(i+1)%L
            xm=(i-1)%L
            yp=(j+1)%L
            ym=(j-1)%L
    
            G[n]=set([xp*L+j,xm*L+j,i*L+yp,i*L+ym])
            
    
    J_arr=np.linspace(0,1,20)
    lag=L**2
    nsteps=5000
    eq_time=3*L*L
    
    num_workers=mp.cpu_count()-1
    print("reached here, number of workers = ",num_workers)
    #with Pool(num_workers) as p:
    #    print(p.map(mc_sample, [(G,N,J,nsteps,lag,eq_time,) for J in J_arr]))
    t0=time.time()
    pool=Pool(num_workers)
    Z_exp=pool.starmap(mc_sample, [(G,N,J,nsteps,lag,eq_time) for J in J_arr])
    print("parallel time ", time.time()-t0)
    
    t0=time.time()
    Z_exp=[mc_sample(G,N,J,nsteps,lag,eq_time) for J in J_arr]
    print("serial time", time.time()-t0)
    #processes = [Process(target=mc_sample, args=(G,N,J,nsteps,lag,eq_time,)) for J in J_arr]

    # start all processes
    #for process in processes:
    #    process.start()
    ## wait for all processes to complete
    #for process in processes:
    #    process.join()
    # report that all tasks are completed
    print("Z_exp",Z_exp)
    print('Done', flush=True)        
