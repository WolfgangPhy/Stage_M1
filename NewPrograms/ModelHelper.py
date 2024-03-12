import math
from scipy.spatial.transform import Rotation as Rot


class ModelHelper:
    """
    Utility functions for coordinate change and integration.
    
    # Methods:
        - `convert_galactic_to_cartesian_3D(ell, b, d)`: Converts from galactic coordinates to
            cartesian coordinates. (Static)
        - `convert_cartesian_to_galactic_3D(x, y, z)`: Converts from cartesian coordinates to
            galactic coordinates. (Static)
        - `convert_cartesian_to_galactic_2D(x, y)`: Converts from cartesian coordinates to
            galactic coordinates. (Static)
        - `convert_galactic_to_cartesian_2D(ell, d)`: Converts from galactic coordinates to
            cartesian coordinates. (Static)
        - `integ_d(func, ell, b, dmax, model, dd=0,01)`: Integrates a function over a line of sight
            in the galactic plane. (Static)
        - `integ_d_async(idx, func, ell, b, dmax, model, dd=0.01)`: Integrates a function over a line of sight
            in the galactic plane. (Static)
        - `gauss3d(x, y, z, x0, y0, z0, m_tot, s1, s2, s3, a1, a2)`: Return the value of the density of a cloud
            at a given point in the Galactic plane. (Static)
        - `compute_extinction_model_density(extinction_model, x, y, z)`: Computes the density of the model
            at a given point in the Galactic plane. (Static)
            
    # Examples:
        The following example shows how to convert from galactic coordinates to cartesian coordinates.
        
        >>> x, y, z = ModelHelper.convert_galactic_to_cartesian_3D(ell, b, d)
        
        The following example shows how to compute the density of the model at a given point in the Galactic plane.
        
        >>> density = ModelHelper.compute_extinction_model_density(extinction_model, x, y, z)
    
    """
    @staticmethod
    def convert_galactic_to_cartesian_3D(ell, b, d):
        """
        Converts from galactic coordinates to cartesian coordinates.

        # Args:
            - `ell (float)`: Galactic longitude in degrees (0 to 360).
            - `b (float)`: Galactic latitude in degrees (-90 to 90).
            - `d (float)`: Distance in kpc.

        " Returns:
            `tuple[float, float, float]` : Cartesian coordinates (x, y, z) in kpc.
        """
        return d * math.cos(b*math.pi/180.) * math.cos(ell*math.pi/180.), \
            d * math.cos(b*math.pi/180.) * math.sin(ell*math.pi/180.), \
            d * math.sin(b*math.pi/180.)
    
    @staticmethod
    def convert_cartesian_to_galactic_3D(x, y, z):
        """
        Converts from cartesian coordinates to galactic coordinates.

        # Args:
            - `x (float)`: x coordinate in kpc.
            - `y (float)`: y coordinate in kpc.
            - `z (float)`: z coordinate in kpc.

        # Returns:
            `tuple[float, float, float]`: Galactic coordinates (ell, b, d) in degrees and kpc.
        """
        R = math.sqrt(x**2 + y**2)
        ell = math.atan2(y, x)
        if ell < 0:
            ell = ell + 2.*math.pi
        return ell*180./math.pi, \
            math.atan2(z, R) * 180./math.pi, \
            math.sqrt(x**2 + y**2 + z**2)
                
    @staticmethod
    def convert_cartesian_to_galactic_2D(x, y):
        """
        Converts from cartesian coordinates to galactic coordinates.

        # Args:
            - `x (float)`: x coordinate in kpc.
            - `y (float)`: y coordinate in kpc.

        # Returns:
            `tuple[float, float]`: Galactic coordinates (ell,d) in degrees and kpc.
        """
        R = math.sqrt(x**2 + y**2)
        ell = math.atan2(y, x)
        if ell < 0:
            ell = ell + 2.*math.pi
        return ell * 180./math.pi, \
            R

    @staticmethod
    def convert_galactic_to_cartesian_2D(ell, d):
        """
        Converts from galactic coordinates to cartesian coordinates.

        # Args:
            - `ell (float)`: Galactic longitude in degrees (0 to 360).
            - `d (float)`: Distance in kpc.

        " Returns:
            `tuple[float, float]`: Cartesian coordinates (x,y) in kpc.
        """
        return d * math.cos(ell*math.pi/180.), \
            d * math.sin(ell*math.pi/180.)
                
    @staticmethod
    def integ_d(func, ell, b, dmax, model, dd=0.01):
        """
        Integrates a function f over a line of sight in the galactic plane.
        
        # Args:
            - `func (callable)`: Function to integrate.
            - `ell (float)`: Galactic longitude in degrees (0 to 360).
            - `b (float)`: Galactic latitude in degrees (-90 to 90).
            - `dmax (float)`: Maximum distance in kpc.
            - `model (extmy_model)`: Model to use.
            - `dd (float, optional)`: Step size in kpc. Defaults to 0.01.

        # Returns:
            `float`: Value of the integral.
        """
        n = int(dmax/dd)
        x, y, z = ModelHelper.convert_galactic_to_cartesian_3D(ell, b, dmax)
        s = 0.5 * (func(model, 0., 0., 0.) + func(model, x, y, z))
        for i in range(1, n, 1):
            x, y, z = ModelHelper.convert_galactic_to_cartesian_3D(ell, b, i*dd)
            s = s + func(model, x, y, z)
        
        return dd * s
    
    @staticmethod
    def integ_d_async(idx, func, ell, b, dmax, model, dd=0.01):
        """
        Integrates a function f over a line of sight in the galactic plane.
        
        # Args:
            - `idx (int)`: Index of the integral.
            - `func (callable)`: Function to integrate.
            - `ell (float)`: Galactic longitude in degrees (0 to 360).
            - `b (float)`: Galactic latitude in degrees (-90 to 90).
            - `dmax (float)`: Maximum distance in kpc.
            - `model (extmy_model)`: Model to use.
            - `dd (float, optional)`: Step size in kpc. Defaults to 0.01.

        # Returns:
            `tuple[int, float]`: Index and value of the integral.
        """
        n = int(dmax/dd)
        x, y, z = ModelHelper.convert_galactic_to_cartesian_3D(ell, b, dmax)
        s = 0.5 * (func(model, 0., 0., 0.) + func(model, x, y, z))
        for i in range(1, n, 1):
            x, y, z = ModelHelper.convert_galactic_to_cartesian_3D(ell, b, i*dd)
            s = s + func(model, x, y, z)
        return idx, \
            dd * s

    @staticmethod
    def gauss3d(x, y, z, x0, y0, z0, m_tot, s1, s2, s3, a1, a2):
        """
        Return the value of the density of a cloud at a given point in the Galactic plane.

        # Args:
            - `x (float)`: x coordinate in kpc.
            - `y (float)`: y coordinate in kpc.
            - `z (float)`: z coordinate in kpc.
            - `x0 (float)`: x coordinate of the center in kpc.
            - `y0 (float)`: y coordinate of the center in kpc.
            - `z0 (float)`: z coordinate of the center in kpc.
            - `m_tot (float)`: Total mass of the cloud.
            - `s1 (float)`: Size along x axis in kpc.
            - `s2 (float)`: Size along y axis in kpc.
            - `s3 (float)`: Size along z axis in kpc.
            - `a1 (float)`: Rotation angle around x axis in degrees.
            - `a2 (float)`: Rotation angle around z axis in degrees.

        # Returns:
            `float` : Value of the density of the cloud at the given point.
        """
        v = [x-x0, y-y0, z-z0]
        r1 = Rot.from_euler('x', a1, degrees=True)
        r2 = Rot.from_euler('z', a2, degrees=True)
        xx = r2.apply(r1.apply(v))
        return m_tot/((2*math.pi)**1.5 * s1 * s2 * s3) * math.exp(-0.5 * (xx[0]**2/s1 + xx[1]**2/s2 + xx[2]**2/s3))    
    
    @staticmethod
    def compute_extinction_model_density(exctinction_model, x, y, z):
        """
        Computes the density of the model at a given point in the Galactic plane.

        # Args:
            - `extinction_model (ExctinctionModel)`: Model to use.
            - `x (float)`: x coordinate in kpc.
            - `y (float)`: y coordinate in kpc.
            - `z (float)`: z coordinate in kpc.

        # Returns:
            `float` : Value of the density.
        """
        hr = 2.5  # kpc
        hz = 0.05  # kpc
        absorp = 0.2  # mag/kpc
        x_sum = 8.  # kpc (Sun x coordinate relative to the Galactic center)
        R = math.sqrt((x-x_sum)**2 + y**2)
        density = absorp * math.exp(-(R - x_sum)/hr) * math.exp(-abs(z)/hz)
        
        for i in range(len(exctinction_model.x0)):
            density += ModelHelper.gauss3d(x, y, z, exctinction_model.x0[i], exctinction_model.y0[i],
                                           exctinction_model.z0[i], exctinction_model.m_tot[i],
                                           exctinction_model.s1[i],
                                           exctinction_model.s2[i], exctinction_model.s3[i],
                                           exctinction_model.a1[i], exctinction_model.a2[i]
                                           )
        return density
    
