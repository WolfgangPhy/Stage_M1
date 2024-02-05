import pickle
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

#utility functions for coordinate change and integration

def ConvertGalacticToCartesian3D(l, b, d):
    """Converts from galactic coordinates to cartesian coordinates

    Args:
        l (float): Galactic longitude in degrees (0 to 360)
        b (float): Galactic latitude in degrees (-90 to 90)
        d (float): Distance in kpc

    Returns:
        tuple[float, float, float] : Cartesian coordinates (x,y,z) in kpc
    """
    return  d * math.cos(b*math.pi/180.) * math.cos(l*math.pi/180.), \
            d * math.cos(b*math.pi/180.) * math.sin(l*math.pi/180.), \
            d * math.sin(b*math.pi/180.)

def ConvertGalacticToCartesian2D(l, d):
    """Converts from galactic coordinates to cartesian coordinates

    Args:
        l (float): Galactic longitude in degrees (0 to 360)
        d (float): Distance in kpc

    Returns:
        tuple[float, float]: Cartesian coordinates (x,y) in kpc
    """
    return  d * math.cos(l*math.pi/180.), \
            d * math.sin(l*math.pi/180.)

def ConvertCartesianToGalactic3D(x, y, z):
    """Converts from cartesian coordinates to galactic coordinates

    Args:
        x (float): x coordinate in kpc
        y (float): y coordinate in kpc
        z (float): z coordinate in kpc

    Returns:
        tuple[float, float, float]: Galactic coordinates (l,b,d) in degrees and kpc
    """
    R = math.sqrt(x**2 + y**2)
    ell = math.atan2(y,x)
    if ell < 0:
        ell = ell + 2.*math.pi
    return  ell*180./math.pi, \
            math.atan2(z,R) * 180./math.pi, \
            math.sqrt(x**2 + y**2 + z**2)

def ConvertCartesianToGalactic2D(x, y):
    """Converts from cartesian coordinates to galactic coordinates

    Args:
        x (float): x coordinate in kpc
        y (float): y coordinate in kpc

    Returns:
        tuple[float, float]: Galactic coordinates (l,d) in degrees and kpc
    """
    R = math.sqrt(x**2 + y**2)
    ell = math.atan2(y, x)
    if ell < 0:
        ell = ell + 2.*math.pi
    return  ell * 180./math.pi, \
            R

def integ_d(func , l, b, dmax, model, dd=0.01):
    """Integrates a function f over a line of sight in the galactic plane
    
    Args:
        func (function): Function to integrate
        l (float): Galactic longitude in degrees (0 to 360)
        b (float): Galactic latitude in degrees (-90 to 90)
        dmax (float): Maximum distance in kpc
        model (extmy_model): Model to use
        dd (float, optional): Step size in kpc. Defaults to 0.01.

    Returns:
        float: Value of the integral
    """
    #uses trapezoidal rule WARNING - dmax/dd might not be an integer
    n = int(dmax/dd)
    x, y, z = ConvertGalacticToCartesian3D(l, b, dmax)
    s = 0.5 * (func(0., 0., 0., model) + func(x, y, z, model))
    for i in range(1, n, 1):
        x, y, z = ConvertGalacticToCartesian3D(l, b, i*dd)
        s = s + func(x, y, z, model)
    return dd * s

def integ_d_async(idx,f,l,b,dmax,model,dd=0.01):
    #uses trapezoidal rule WARNING - dmax/dd might not be an integer
    n = int(dmax/dd)
    x,y,z = ConvertGalacticToCartesian3D(l,b,dmax)
    s = 0.5*(f(0.,0.,0.,model) + f(x,y,z,model))
    for i in range(1,n,1):
        x,y,z = ConvertGalacticToCartesian3D(l,b,i*dd)
        s = s + f(x,y,z,model)
    return (idx,dd*s)

def gauss3d(x, y, z, x0, y0, z0, rho, s1, s2, s3, a1, a2):
    """3D Gaussian function

    Args:
        x (float): x coordinate in kpc
        y (float): y coordinate in kpc
        z (float): z coordinate in kpc
        x0 (float): x coordinate of the center in kpc
        y0 (float): y coordinate of the center in kpc
        z0 (float): z coordinate of the center in kpc
        rho (float): Density at the center
        s1 (float): Size along x axis in kpc
        s2 (float): Size along y axis in kpc
        s3 (float): Size along z axis in kpc
        a1 (float): Rotation angle around x axis in degrees
        a2 (float): Rotation angle around z axis in degrees

    Returns:
        float : Value of the Gaussian function
    """
    v=[x-x0, y-y0, z-z0]
    r1 = R.from_euler('x', a1, degrees=True)
    r2 = R.from_euler('z', a2, degrees=True)
    xx=r2.apply(r1.apply(v))
    return rho/((2*math.pi)**1.5 * s1 * s2 * s3) * math.exp(-0.5 * (xx[0]**2/s1 + xx[1]**2/s2 + xx[2]**2/s3))    
      
def ext_model(x, y, z, model):
    """Computes the extinction model
    
    Args:
        x (float): x coordinate in kpc
        y (float): y coordinate in kpc
        z (float): z coordinate in kpc
        model (extmy_model): Model to use

    Returns:
        float : Value of the extinction model
    """
    #Assumes a double exponential disk model near the Sun
    hr = 2.5 #kpc
    hz = 0.05 #kpc
    absorp = 0.2 #mag/kpc
    r_sun = 8.
    # first assumes a double exponential disk model with hr=2.5kpc hz=0.05 kpc 
    # with an absoption of 0.2mag/kpc near the Sun radius (-8,0,0)
    R = math.sqrt((x-r_sun)**2 + y**2)
    d = absorp * math.exp(-(R - r_sun)/hr) * math.exp(-abs(z)/hz)
     
    for i in range(len(model.x0)):
        d = d + gauss3d(x, y, z, model.x0[i], model.y0[i], model.z0[i], model.rho[i], model.s1[i], model.s2[i], model.s3[i], model.a1[i], model.a2[i])
    
    return d


class extmy_model:
    """
    Class to store a given model.

    Attributes:
        rho (numpy.ndarray): Core density for each model.
        x0 (numpy.ndarray): Cartesian location of cloud along the x-axis [-4, 4] kpc for each model.
        y0 (numpy.ndarray): Cartesian location of cloud along the y-axis [-4, 4] kpc for each model.
        z0 (numpy.ndarray): Cartesian location of cloud along the z-axis [-0.5, 0.5] kpc for each model.
        s1 (numpy.ndarray): Size along the first axis for each model in kpc.
        s2 (numpy.ndarray): Size along the second axis for each model in kpc.
        s3 (numpy.ndarray): Size along the third axis for each model in kpc.
        a1 (numpy.ndarray): Orientation angle along the first axis [-30, 30] degrees for each model.
        a2 (numpy.ndarray): Orientation angle along the second axis [-45, 45] degrees for each model.

    Methods:
        __len__(): Returns the total number of samples, which is the length of the 'rho' array.
    """
    def __init__(self,N):
        """
        Initializes an instance of the extmy_model class.

        Args:
            N (int): Number of models to generate.
        """
        self.rho = np.random.rand(N)*0.1 
        self.x0 = np.random.rand(N)*8.-4. 
        self.y0 = np.random.rand(N)*8.-4. 
        self.z0 = np.random.rand(N)-0.5   
        self.s1 = 0.05+np.random.rand(N)*0.05 
        self.s2 = 0.05+np.random.rand(N)*0.05
        self.s3 = 0.05+np.random.rand(N)*0.05
        self.a1 = np.random.rand(N)*60.-30. 
        self.a2 = np.random.rand(N)*90.-45. 

    def __len__(self):
        """
        Returns the total number of samples.

        Returns:
            int: The length of the 'rho' array.
        """
        return len(self.rho)

 
