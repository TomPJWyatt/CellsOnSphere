import numpy as np
from scipy.spatial.transform import Rotation as R


def vecs2ang(v1,v2):
    """
    Calculates the angle in radians between 2 vectors.
    
    Parameters
    ----------
    v1,v2 : array_like, shape (3,) or (N,3)
        The vectors in cartesian coordinates. (Basic dot product doesn't give 
        angle in spherical coords!)
    """
    dot = np.dot(v1,v2)
    v1m = np.linalg.norm(v1)
    v2m = np.linalg.norm(v2)
    return np.arccos(dot/(v1m*v2m))


def vectortoplaneAng(v,n):
    """
    Give the angle between a vector and a plane.
    
    Parameters
    ----------
    v : array_like, shape (3,) or (N,3)
        The vector in cartesian coordinates.
    n : array_like, shape (3,) or (N,3)
        The normal vector of the plane.
    """
    dot = np.dot(v1,v2)
    v1m = np.linalg.norm(v1)
    v2m = np.linalg.norm(v2)
    return np.arcsin(dot/(v1m*v2m))    


def sphereDistance(x1,y1,z1,x2,y2,z2,r):
    """
    Calculate the distance between two points (represented in cartesian 
    coordinates) that lie on the surface of a sphere of radius r, as the 
    shortest distance across surface of sphere. 
    
    Parameters
    ----------
    x1,y1,z1,x2,y2,z2 : float or array_like, shape (1,), (N,1) or (N,N)
        Coordinates of the two points, in cartesian coordinates. 
        t=theta, p=phi. In radians. (N,N) is just for finding the distance 
        matrix of a set of points (see core.findDistanceMatrix()).
    r : float
        The radius of the sphere which the points lie on and along whose 
        surface you measure the distance.
    """
    r1,lon1,lat1 = np.moveaxis(cart2lonlat(x1,y1,z1),-1,0)
    r2,lon2,lat2 = np.moveaxis(cart2lonlat(x2,y2,z2),-1,0)
    cosdlon = np.cos(lon2-lon1)
    cossig = np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*cosdlon
    
    if isinstance(cossig,np.ndarray):
        cossig[cossig>1] = 1
    
    # set diagonal elements to 1 to stop zero errors in distance matrix calc
    if isinstance(cossig,np.ndarray) and len(cossig.shape) > 1:
        np.fill_diagonal(cossig,1)
    
    sig = np.arccos(cossig)
    return r*sig
    

def cart2sph(x,y,z):
    """
    Converts cartesian coordinates to spherical polar.
    
    Parameters
    ----------
    x,y,z : float or array_like, shape (1,) or (N,1)
        The cartesian coordinates. 
        
    Returns
    -------
    v2 : array_like, shape (3,) or (N,3)
        The coordinates in spherical polars [r,theta,phi]. Angles in radians.
    """
    dxy = np.sqrt(x**2 + y**2)
    r = np.sqrt(dxy**2 + z**2)
    theta = np.arctan2(dxy,z)
    phi = np.arctan2(y,x)
    return np.moveaxis(np.array([r,theta,phi]),0,-1)


def cart2lonlat(x,y,z):
    """
    Converts cartesian coordinates to longitude and latitude.
    
    Parameters
    ----------
    x,y,z : float or array_like, shape (1,) or (N,1)
        The cartesian coordinates. 
        
    Returns
    -------
    v2 : array_like, shape (3,) or (N,3)
        The coordinates in longitude and latitude [r,lon,lat]. Angles in 
        radians.
    """
    dxy = np.sqrt(x**2 + y**2)
    r = np.sqrt(dxy**2 + z**2)
    lat = np.arctan2(z,dxy)
    lon = np.arctan2(y,x)
    return np.moveaxis(np.array([r,lon,lat]),0,-1)


def lonlat2cart(r,lon,lat):
    """
    Converts longitude and latitude to cartesian coordinates.
    
    Parameters
    ----------
    r,lon,lat : float or array_like, shape (1,) or (N,1) or (N,N)
        The longitude-latitude coordinates. 
        
    Returns
    -------
    v2 : array_like, shape (3,) or (N,3) or (N,N,3)
        The coordinates in longitude and latitude [x,y,z]. Angles in 
        radians.
    """
    z = r*np.sin(lat)
    rcoslat = r*np.cos(lat)
    x = rcoslat*np.cos(lon)
    y = rcoslat*np.sin(lon)
    return np.moveaxis(np.array([x,y,z]),0,-1)


def sph2cart(r,theta,phi):
    """
    Converts spherical polar coordinates to cartesian.
    
    Parameters
    ----------
    r,theta,phi : float or array_like, shape (1,) or (N,1)
        The coordinates in sperical polar as [r,theta,phi]. Angles in radians.
        
    Returns
    -------
    v2 : array_like, shape (3,) or (N,3)
        The coordinates in cartesian [x,y,z].
    """    
    z = r*np.cos(theta)
    rsintheta = r*np.sin(theta)
    x = rsintheta*np.cos(phi)
    y = rsintheta*np.sin(phi)
    return np.array([x,y,z]).T

    
def pointOnPlaneQ(v,n):
    """
    Tests whether point with position vector v lies on the plane with normal n.
    
    Parameters
    ----------
    v : [float,float,float]
        The coordinates in cartesian [x,y,z] of the point.
    n : [float,float,float]
        The normal vector of the plane in cartesian [x,y,z].
    """
    if np.dot(v,n)==0:
        return True
    else:
        return False


def rotateAboutAxis(v,n,ang):
    """
    Rotates a point (or points) with coordinates v by angle ang radians, 
    about the axis which passes through origin and is defined by vector n.
    
    Parameters
    ----------
    v : array_like, shape (3,) or (N, 3)
        The coordinates in cartesian [x,y,z] of the point(s).
    n : array_like, shape (3,) or (N, 3)
        The vector of the axis of rotaion in cartesian [x,y,z].
    ang : float or array-like, shape (1,) or (N,1)
        The angle to rotate through, in radians.
    
    Returns
    -------
    v2 : array_like, shape (3,) or (N, 3)
        The new point(s) in cartesian.
    """
    if len(n.shape)==1:
        norm = np.linalg.norm(n)
        n2 = n/norm
    else:
        norm = np.linalg.norm(n,axis=1)
        n2 = n/norm[:,None]
    
    r = R.from_rotvec(ang*n2)
    return r.apply(v)
    