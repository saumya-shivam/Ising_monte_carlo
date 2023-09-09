# Monte Carlo Simulation of the Ising Model

Monte Carlo simulation of the Ising model for an arbitrary graph using the Metropolis/Wolff cluster algorithm. 

An arbitrary dictonary based graph can be used to initialize an IsingMC class object, but by default uses a square lattice. Also has a built-in function to build a 3D cubic graph. For the cluster algorithm, a breadth first search is performed to keep track of the boundary of the cluster.

## How to run?

Initialize an IsingMC class object with the following arguments.

| Variable | Description    | Type    | Default Value |
| :---:   | :---: | :---: | :---: |
| Nsites | Total number of sites in the graph (rounds to nearest square for square lattice)   | int   | 100    |
| temp | List of inverse temperatures to use   | List[int]   | [1]   |
| Nsteps | Total number of MC steps   | int   | 1000   |
| eq_steps | Number of steps to wait before collecting date   | int   | 50  |
| lag | collect samples periodically after lag steps   | int   | 5   |
| full_Zexp | If magnetization(Zexp) is collected at each step   | bool   | False   |
| graph | Interaction graph   | dict   | None   |

Once a model is initialized with these parameters, perform MC sampling using model.run(), which takes the following arguments
| Variable | Description    | Type    | Default Value |
| :---:   | :---: | :---: | :---: |
| algo | Name of the algorithm (currently 'cluster' or 'metropolis')  | str   | 'cluster'    |
| n_workers | Number of workers to use for multiprocessing   | int   | 1   |

Once the model has been run, use model.Zexp to access the collected data (could be corresponding to each MC step if full_Zexp==True). use model.status to check which algorithm was run most recenctly.
