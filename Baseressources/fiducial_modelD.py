import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import multiprocessing as mp
import ext_model
import test_1ANN_2D.MyDataset2D


fmodel="ext_model2D.pickle"
if os.path.isfile(fmodel):
    print("Using existing model")
    newmodel = 0
else:
    newmodel = 1 #set to zero
    print("Creating new model")
    
    
# Create new model
if newmodel==1:    
    model = ext_model.extmy_model(100)
    #print(model.rho,model.a1)
    with open(fmodel,"wb") as file:
        pickle.dump(model,file)
        file.close()
        

# Load model
with open(fmodel,"rb") as file:
    model = pickle.load(file)
    file.close()
    
X,Y = np.mgrid[-5:5.1:0.1, -5:5.1:0.1]
dens = X*0.

for i in range(len(X[:,1])):
    for j in range(len(X[1,:])):
        dens[i,j] = ext_model.ext_model(X[i,j],Y[i,j],0.,model)

plt.pcolormesh(X, Y, dens, shading='auto', cmap=plt.cm.gist_yarg)
plt.show()

dtype = torch.float
#device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device("cpu") # Uncomment this to run on CPU
print(device)



print("Number of processors: ", mp.cpu_count())

results=[]
pool = mp.Pool(mp.cpu_count())

# call back function for asynchroneous parallel processing
def collect_result(result):
    global results
    results.append(result)

# define a training sample of n points -> using 3D model to construct 2D model
n=20000
ell = torch.rand(n,device=device)*360.
b = torch.zeros(n,device=device,dtype=dtype)
dist = torch.rand(n,device=device)*5.5

K = torch.zeros(n,device=device,dtype=dtype)
error = torch.zeros(n,device=device,dtype=dtype)


print("Start processing")
# use loop for parallel processing  
for i in range(n):
    pool.apply_async(ext_model.integ_d_async, args=(i,ext_model.ext_model,ell[i].data,b[i].data,dist[i].data,model),callback=collect_result)

# close pool
pool.close()
pool.join() # wait for all processes to be completed 

# sort the result
print("Sorting results")
results.sort(key=lambda x: x[0])
K = [r for i,r in results]

print("Adding errors")
for i in range(n):
    error[i] = K[i].item()*np.random.uniform(low=0.01,high=0.1)
    K[i] = K[i].item()+np.random.normal(scale=error[i].item())


# prepares the dataset used for training
dataset = MyDataset2D(ell, dist, K, error)
torch.save(dataset,"fiducial_model2D.pt")
