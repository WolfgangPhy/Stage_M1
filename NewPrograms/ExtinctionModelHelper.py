import numpy as np
import math
from scipy.spatial.transform import Rotation as R

class ExtinctionModelHelper:
    """Utility functions for coordinate change and integration"""
    @staticmethod
    def ConvertGalacticToCartesian3D(l, b, d):
        """Converts from galactic coordinates to cartesian coordinates

        Args:
            l (float): Galactic longitude in degrees (0 to 360)
            b (float): Galactic latitude in degrees (-90 to 90)
            d (float): Distance in kpc

        Returns:
            tuple[float, float, float] : Cartesian coordinates (x, y, z) in kpc
        """
        return (
            d * math.cos(b*math.pi/180.) * math.cos(l*math.pi/180.),
            d * math.cos(b*math.pi/180.) * math.sin(l*math.pi/180.),
            d * math.sin(b*math.pi/180.)
        )
    
    @staticmethod
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
                
    @staticmethod
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

    @staticmethod
    def integ_d(func, l, b, dmax, model, dd=0.01):
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
        x, y, z = ExtinctionModelHelper.ConvertGalacticToCartesian3D(l, b, dmax)
        s = 0.5 * (func(0., 0., 0., model) + func(x, y, z, model))
        for i in range(1, n, 1):
            x, y, z = ExtinctionModelHelper.ConvertGalacticToCartesian3D(l, b, i*dd)
            s = s + func(x, y, z, model)
        return dd * s

    @staticmethod
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
    
