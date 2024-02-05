import pickle
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

#utility functions for coordinate change and integration

def lbd_xyz(l,b,d):
    #assumes l and b in deg [0,360],[-90,90]
    return d*math.cos(b*math.pi/180.)*math.cos(l*math.pi/180.),d*math.cos(b*math.pi/180.)*math.sin(l*math.pi/180.),d*math.sin(b*math.pi/180.)

def ld_xy(l,d):
    #assumes l in deg [0,360]
    return d*math.cos(l*math.pi/180.),d*math.sin(l*math.pi/180.)

def xyz_lbd(x,y,z):
    R = math.sqrt(x**2+y**2)
    ell = math.atan2(y,x)
    if ell<0:
        ell = ell + 2.*math.pi
    return ell*180./math.pi,math.atan2(z,R)*180./math.pi,math.sqrt(x**2+y**2+z**2)

def xy_ld(x,y):
    R = math.sqrt(x**2+y**2)
    ell = math.atan2(y,x)
    if ell<0:
        ell = ell + 2.*math.pi
    return ell*180./math.pi,R

def integ_d(f,l,b,dmax,model,dd=0.01):
    #uses trapezoidal rule WARNING - dmax/dd might not be an integer
    n = int(dmax/dd)
    x,y,z = lbd_xyz(l,b,dmax)
    s = 0.5*(f(0.,0.,0.,model) + f(x,y,z,model))
    for i in range(1,n,1):
        x,y,z = lbd_xyz(l,b,i*dd)
        s = s + f(x,y,z,model)
    return dd*s

def integ_d_async(idx,f,l,b,dmax,model,dd=0.01):
    #uses trapezoidal rule WARNING - dmax/dd might not be an integer
    n = int(dmax/dd)
    x,y,z = lbd_xyz(l,b,dmax)
    s = 0.5*(f(0.,0.,0.,model) + f(x,y,z,model))
    for i in range(1,n,1):
        x,y,z = lbd_xyz(l,b,i*dd)
        s = s + f(x,y,z,model)
    return (idx,dd*s)

# construct the extinction model we will try to recover and save it to a file
def gauss3d(x,y,z,x0,y0,z0,rho,s1,s2,s3,a1,a2):
    v=[x-x0,y-y0,z-z0]
    r1 = R.from_euler('x', a1, degrees=True)
    r2 = R.from_euler('z', a2, degrees=True)
    xx=r2.apply(r1.apply(v))
    return rho/((2*math.pi)**1.5*s1*s2*s3)*math.exp(-0.5*(xx[0]**2/s1+xx[1]**2/s2+xx[2]**2/s3))    
      
def ext_model(x,y,z,model):
    #global model
    
    # first assumes a double exponential disk model with hr=2.5kpc hz=0.05 kpc 
    # with an absoption of 0.2mag/kpc near the Sun radius (-8,0,0)
    R = math.sqrt((x-8.)**2 + y**2)
    d = 0.2 * math.exp(-(R-8.)/2.5)*math.exp(-abs(z)/0.05)
     
    for i in range(len(model.x0)):
        d = d + gauss3d(x,y,z,model.x0[i],model.y0[i],model.z0[i],model.rho[i],model.s1[i],model.s2[i],model.s3[i],model.a1[i],model.a2[i])
    
    return d

### CLASS TO STORE A GIVEN MODEL
class extmy_model:
    def __init__(self,N):
        self.rho = np.random.rand(N)*0.1 # core density
        self.x0 = np.random.rand(N)*8.-4. # cartesian location of cloud [-4,4] kpc
        self.y0 = np.random.rand(N)*8.-4. # cartesian location of cloud [-4,4] kpc
        self.z0 = np.random.rand(N)-0.5   # cartesian location od cloud [-0.5,0.5] kpc
        self.s1 = 0.05+np.random.rand(N)*0.05 # size along axis kpc
        self.s2 = 0.05+np.random.rand(N)*0.05
        self.s3 = 0.05+np.random.rand(N)*0.05
        self.a1 = np.random.rand(N)*60.-30. # orientation [-30,30] deg
        self.a2 = np.random.rand(N)*90.-45. # orientation [-45,45] deg

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.rho)

 
