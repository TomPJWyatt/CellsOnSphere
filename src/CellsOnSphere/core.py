import math
from math import acos,pi
import random
import numpy as np
from scipy.spatial import SphericalVoronoi
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from . import utils as u

# minimum allowed distance between cell centres as fraction of radius
MIN_DIST = 1 


class Cell():
    """
    Represents an endoderm cell.
    
    Parameters
    ----------
    pSphere : zerbendo.Sphere
        The embryo that the cell belongs to.
    N : int
        The index of the cell within self.pSphere.cellsList.
    pos : [float,float,float]
        The position of the cell on the sphere-embryo surface, [x,y,z].
        i.e. in cartesian coordinates.
    radius : int
        The radius in um of the cell.        
    vel : float
        The magnitude of velocity of the cell along the surface of the embryo.
    gCirc : [float,float,float]
        The great circle that the cell is travelling on, represented as the 
        unit normal of the plane of the great circle, in cartesian coords.
    persistence : float or bool
        A kind of persistence length. Currently if you take a step whose size 
        is equal to the persistence length then your new orientation is chosen 
        from a normal distribution with mean = 0 and std = pi. For smaller 
        steps the variance scales as step/persistence to ensure total 
        variance of many steps if same as one big one (by sum of normals rule).
        If False then you do not change direction when you step.
    repel : bool or float
        Whether the cell is repelled from other cells. If False then cells 
        will not change direction when they collide. If True then they will 
        move in opposite directions along the line between their centres. If 
        float (must be between 0-1) then they will change direction by max. 
        this fraction of 180 degrees to reach that direction described before.
    repRand : bool or float
        If repel is a bool then this will choose a random angle between 
        -pi/2<a<pi/2 and add this to the direction after the repel opposite 
        direction has been set.
    age : float
        The time since the last cell division.
    dividing : bool
        Whether this cell will divide or not.
    cycleT : float
        The average time between divisions.
    divTime : float
        The sphere-embryo-time at which this cell will divide.
    passProb : False or float from 0 to 1
        If False and repel==False then cells can't pass through each other 
        ever. If passProb is a float then this is the probability that moves 
        directly towards the centre of another cell will be accepted. Moves 
        away from or tangental to the cell will be accepted and the 
        probability varies linearly between those angles.
    """
    
    def __init__(self,
                 pSphere=None,
                 N = 0,
                 pos=None,
                 radius=6.6,
                 vel=0,
                 gCirc = None,
                 persistence=1000,
                 repel=True,
                 repRand=False,
                 age = 0,
                 dividing = True,
                 cycleT = 1500,
                 divTime = None,
                 passProb = False
                ):
        self.pSphere = pSphere
        self.N = N
        self.pos = pos
        if pos is None:
            pos = [radius,0,0]
        self.radius = radius
        self.vel = vel
        self.gCirc = gCirc 
        if gCirc is None:
            if self.pos[1]==0 and self.pos[2]==0:
                gCirc = np.cross(self.pos,[0,1,0])
                gCirc = gCirc/np.linalg.norm(gCirc)                
            else:
                gCirc = np.cross(self.pos,[1,0,0])
                gCirc = gCirc/np.linalg.norm(gCirc)
        self.persistence = persistence
        self.repel = repel
        self.repRand = repRand
        self.age = age
        self.dividing = dividing
        self.cycleT = cycleT
        self.divTime = divTime
        if self.divTime is None:
            self.resetDivTime()
        self.passProb = passProb

        
    def resetDivTime(self):
        """
        Sets the sphere-embryo-time that the cell will divide based on the 
        current time and the cell's age and cycleT.
        """
        std = self.cycleT/5
        lifetime = -1
        while lifetime<0:
            lifetime = self.cycleT + np.random.normal(0,std,1)
        self.divTime = self.pSphere.time + lifetime - self.age
        if self.divTime<self.pSphere.time:
            print('Warning: divTime was reset to less than current time!')

            
    def resolveOverlap(self):
        """
        Checks if cell overlaps beyond MIN_DIST and moves to nearest place 
        with no overlaps if so.
        """
        pE = self.pSphere
        r = pE.radius
        cellsList = [c for i,c in enumerate(pE.cellsList) if i!=self.N]
        ds = [u.sphereDistance(*c.pos,*self.pos,r) for c in cellsList]
        if not any([d<self.radius*MIN_DIST for d in ds]):
            return
        NAng = 30
        angI = 0
        step = 1
        while any([d<self.radius*MIN_DIST for d in ds]):
            angG = angI*2*pi/NAng
            gc = u.rotateAboutAxis(self.gCirc,self.pos,angG)
            angS = step*self.radius/(2*r)
            if angS>pi:
                raise Exception('Couldn\'t find room for cell')
            newP = u.rotateAboutAxis(self.pos,gc,angS)
            ds = [u.sphereDistance(*c.pos,*newP,r) for c in cellsList]
            
            angI += 1
            if angI>=NAng:
                angI = 0
                step += 1
        self.gCirc = gc
        self.pos = newP
    
    
    def moveCell(self,dt):
        """
        Move the cell along it's great circle for time dt. Then change its 
        great circle according to it's persistence. This function is not 
        currently used in Sphere.evolveOneStep().
        
        Parameters
        ----------
        dt : float
            The time interval, which determines the distance cells will move.
            
        Notes
        -----
        Requires all cell radii are the same.
        """
        # do checks
        radCheck = len(set([c.radius for c in self.pSphere.cellsList]))==1
        assert radCheck, 'moveCell : all cells must have same radius'
        overlaps = np.where(self.pSphere.distanceMatrix[N,:]<2*self.radius)
        
        # move cell
        S = dt*self.vel
        ang = S / self.pSphere.radius
        pos2 = u.rotateAboutAxis(self.pos,self.gCirc,ang)
        
        # change great circle
        if self.persistence:
            std = math.pi*(math.sqrt(S/P))
            ang = np.random.normal(0,std,1)
            ang = math.fmod(ang,math.pi)
            self.gCirc = u.rotateAboutAxis(self.gCirc,self.pos,ang)
        
        self.age += dt

        
    def divideCell(self):
        """
        Simulate a cell division. Orientation of division will be the new 
        direction of the old cell. New cell will be in opposite direction. 
        New cell is added to sphere-embryo. They are 2*radius distance from each 
        other, along their great circles.
        """
        
        # choose a random division direction 
        ang = pi*random.randint(-100,100)/100
        self.gCirc = u.rotateAboutAxis(self.gCirc,self.pos,ang)
        
        # move cell to make way for new one
        ang2 = self.radius / self.pSphere.radius
        self.pos = u.rotateAboutAxis(self.pos,self.gCirc,ang2)
        
        self.age = 0
        self.resetDivTime()
        
        # add new cell
        pE = self.pSphere
        N = len(pE.cellsList)
        r = self.radius
        gc = u.rotateAboutAxis(self.gCirc,self.pos,pi)
        p = u.rotateAboutAxis(self.pos,gc,2*ang2)
        v = self.vel
        per = self.persistence
        rep = self.repel
        repR = self.repRand
        age = 0
        cT = self.cycleT
        cell = Cell(pE,N,p,r,v,gc,per,repel=rep,repRand=repR,age=age,dividing=True,cycleT=cT)
        self.pSphere.cellsList.append(cell)
        
        self.resolveOverlap()
        cell.resolveOverlap()
        


class Sphere():
    """
    Represents a spherical embryo.
    
    Parameters
    ----------
    radius : int
        The radius of the sphere-embryo in um.
    regionsList : list of regions
        All regions that you specify to have certain properties.
    cellsList : list of zebrendo.Cell
        All the cells belonging to the sphere-embryo.
    distanceMatrix : numpy.array
        Matrix of distances between cells. Element [i,j] is distance between 
        cells i,j where i,j is given by position in self.cellsList.
    time : float
        The current time of the sphere-embryo in seconds.
    NT : int
        The number of time steps that have been taken.
    epiboly : float
        Percentage as decimal that epiboly has reached. 0 means theta (of 
        spherical polar coords) = 0 and 1 mean theta = pi.
    epibolyV : float
        Speed at which epiboly progresses (percent per second). (Note: NOT as 
        decimal per second, divide by 100 for that).
    """
    def __init__(self,
                 radius=1000,
                 regionsList=[],
                 cellsList=[],
                 startTime=0,
                 epiboly=1,
                 epibolyV=0):
        """
        Parameters
        ----------
        radius : int
            The radius of the sphere-embryo in um.
        regionsList : list of regions
            All regions that you specify to have certain properties.
        cellsList : list of zebrendo.Cell
            All the cells belonging to the sphere-embryo.
        startTime : float
            The time where the self.time counter starts.
        """
        self.radius = radius
        self.regionsList = regionsList
        self.cellsList = cellsList
        self.time = startTime
        self.NT = 0
        self.distanceMatrix = None
        self.history = []
        self.epiboly = epiboly
        self.epibolyV = epibolyV
        
        
    def initiateRing(self,
                     N=20,
                     r=10,
                     v=10,
                     p=1000,
                     rep=True,
                     repRand=False,
                     cT=1500,
                     *args,
                     **kwargs):
        """
        Initiates a sphere-embryo with a ring of N equally spaced cells around 
        the circumference. Each cell has initial direction north.
        
        Parameters - see class Cell for many of theses
        ----------
        N : int
            The number of cells initiated.
        r,v : int
            The radius/velocity of each cell.
        cT = the cell cycle time in seconds
        """
        cs = np.array([[self.radius,pi/2,i*2*pi/N] for i in range(N)])
        cs = u.sph2cart(*cs.T)
        gCs = [np.cross(c,[0,0,1])/np.linalg.norm(c) for c in cs]
        cs = [Cell(self,i,c,r,v,g,p,rep,repRand=repRand,cycleT=cT,*args,**kwargs) for i,[c,g] in enumerate(zip(cs,gCs))]
        self.cellsList = cs
        self.findDistanceMatrix()
        
        
    def initiateFibonacci(self,cell=Cell,N=21,*args,**kwargs):
        """
        Uses the Fibonacci sphere to distribute any number of cells roughly 
        evenly onto surface of sphere.
        
        Parameters
        ----------
        cell : Class
            All cells made on the embryo will be this class.
        N : int
            Number of cells to initiate.
        cellArgs : list
            Any other arguments to pass to the cells.
        """
        i = np.arange(0,N,dtype=float) + 0.5
        phi = np.arccos(1 - 2*i/N)
        goldenRatio = (1 + 5**0.5)/2
        theta = 2*pi*i/goldenRatio
        x = np.cos(theta)*np.sin(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(phi)
        pos = np.moveaxis(np.array([x,y,z]),0,-1)
        
        cells = [cell(self,i,p,*args,**kwargs) for i,p in enumerate(pos)]
        self.cellsList = cells

        
    def initiateFromPosList(self,posList,*args,**kwargs):
        """
        Initiates a sphere-embryo with cells placed according to positions in 
        posList.
        
        Parameters
        ----------
        posList : list of [x,y,z] or numpy array-like, shape (N,3)
            The positions of each cell.
        """
        cells = [Cell(self,i,p,*args,**kwargs) for i,p in enumerate(posList)]
        self.cellsList = cells
    
    
    def findDistanceMatrix(self,pts=None,retn=False):
        """
        Calculates all distances between cells to update self.distanceMatrix.
        
        Parameters
        ----------
        pts : array_like, shape (3,) or (N,3)
            List of points that you want to build a distance matrix from, in 
            cartesian coordinates. If None then it uses the positions of all 
            cells in the sphere-embryo.
        retn : bool
            If True then it returns the distance matrix without setting 
            self.distanceMatrix. Otherwise it sets self.distanceMatrix and 
            returns None.
        """
        if pts is None:
            pts = np.array([c.pos for c in self.cellsList])
        x1,x2 = np.meshgrid(pts[:,0],pts[:,0])
        y1,y2 = np.meshgrid(pts[:,1],pts[:,1])
        z1,z2 = np.meshgrid(pts[:,2],pts[:,2])
        if retn:
            return u.sphereDistance(x1,y1,z1,x2,y2,z2,self.radius)
        self.distanceMatrix = u.sphereDistance(x1,y1,z1,x2,y2,z2,self.radius)
    
    
    def updateHistory(self):
        """
        Saves the current state to the embryo history. Element i is the 
        position of cell i and element -2,-1 are the time and NT.
        """
        data = [c.pos for c in self.cellsList] + [self.time,self.NT]
        self.history.append(data)
    
    
    def evolveOneStep(self,dt):
        """
        Moves all cells in the sphere-embryo by dt.
        The sequence of actions is:
            1. do checks
            2. update times and ages
            3. update great circles according to persistences
            4. if repel then set gcircs in overlapping cells to shortest path 
                out (plus optional random bit).
            5. do test move according to new gcircs
            6. behaviour to test moves if overlap
            7. advance epiboly
            8. if test move beyond epiboly then reject and randomise direction
            9. assign test moves as new positions
            10. divide cells
        
        Parameters
        ----------
        dt : float
            The time interval of the step in seconds.
        """
        # checks: step size not too big,  all radii and repel the same
        Q = [dt*c.vel>c.radius for c in self.cellsList]
        if any(Q):
            print('Warning: S>radius, cells may get jammed.')
        rep = set([c.repel for c in self.cellsList])
        assert len(rep)==1,'evolveOneStep: can\'t yet handle varied repel.'
        rep = list(rep)[0]
        repRand = set([c.repRand for c in self.cellsList])
        assert len(repRand)==1,'evolveOneStep: can\'t yet handle varied repRand.'
        repRand = list(repRand)[0]
        radC = set([c.radius for c in self.cellsList])
        assert len(radC)==1,'currently cell radii must all be equal'
        radC = list(radC)[0] 
        
        # update time
        self.time = self.time + dt
        self.NT += 1
        for c in self.cellsList:
            c.age += dt            
            
        # update all great circles according persistences
        S = np.array([dt*c.vel for c in self.cellsList])
        for i,c in enumerate(self.cellsList):
            if c.persistence:
                #std = pi*(1-math.exp(-S[i]/c.persistence))/4 #one way
                #std = pi*S[i]/(2*c.persistence) # another way
                std = (pi/2)*(math.sqrt(S[i]/c.persistence))
                ang = np.random.normal(0,std,1)
                ang = math.fmod(ang,math.pi)
                c.gCirc = u.rotateAboutAxis(c.gCirc,c.pos,ang)        
        
        # if repel, check overlaps and update great circles as necessary
        if rep:
            self.findDistanceMatrix()
            idi,idj = np.where(self.distanceMatrix<2*radC)
            for i,j in zip(idi,idj):
                if not i==j:
                    v = np.cross(self.cellsList[j].pos,self.cellsList[i].pos)
                    if isinstance(rep,bool):
                        self.cellsList[i].gCirc = v/np.linalg.norm(v)
                        if repRand:
                            a1 = (np.random.rand()-0.5)*repRand*2
                            c1 = self.cellsList[i]
                            self.cellsList[i].gCirc = u.rotateAboutAxis(c1.gCirc,c1.pos,a1)
                    elif isinstance(rep,float):
                        angGC2new = u.vecs2ang(v,self.cellsList[i].pos)
                        if angGC2new < pi*rep:
                            self.cellsList[i].gCirc = v/np.linalg.norm(v)
                        else:
                            gc1 = self.cellsList[i].gCirc
                            p1 = self.cellsList[i].pos
                            v2 = u.rotateAboutAxis(gc1,p1,pi*rep)
                            self.cellsList[i].gCirc = v2/np.linalg.norm(v2)
                    else:
                        assert False, 'rep must be bool or float'
        
        # do test move on all cells
        pos = np.array([c.pos for c in self.cellsList])
        gCircs = np.array([c.gCirc for c in self.cellsList])
        ang = S/self.radius
        ang = np.tile(ang,(3,1)).T
        newPos = u.rotateAboutAxis(pos,gCircs,ang)
        
        # if there is cell overlap then reject the move according to passProb 
        # and the direction of travel. See __init__ for details
        passProb = self.cellsList[0].passProb
        if not rep and passProb:
            newDistMat = self.findDistanceMatrix(newPos,retn=True)
            idi,idj = np.where(newDistMat<radC*2)
            for i,j in zip(idi,idj):
                if not i==j:
                    stepI = newPos[i] - self.cellsList[i].pos
                    vecIJ = self.cellsList[j].pos - self.cellsList[i].pos
                    angDR = u.vecs2ang(stepI,vecIJ)
                    prob = passProb + (1-np.cos(angDR))*(1-passProb)
                    if random.random()>prob:
                        newPos[i] = self.cellsList[i].pos
        elif not rep and not passProb:
            newDistMat = self.findDistanceMatrix(newPos,retn=True)
            idi,idj = np.where(newDistMat<radC*2)
            for i,j in zip(idi,idj):
                if not i==j:
                    newPos[i] = self.cellsList[i].pos
        
        # advance epiboly
        self.epiboly += dt*self.epibolyV/100
        
        # check cells not beyond epiboly
        for i,[c,n] in enumerate(zip(self.cellsList,newPos)):
            if u.cart2sph(*n)[1]>self.epiboly*math.pi:
                newPos[i] = c.pos
                ang = pi*random.randint(-100,100)/100
                c.gCirc = u.rotateAboutAxis(c.gCirc,c.pos,ang)
        
        # assign new cell positions
        for i,c in enumerate(self.cellsList):
            c.pos = newPos[i]
            
        # divide cells
        for c in self.cellsList:
            if c.dividing and self.time > c.divTime:
                c.divideCell()
        
        # update distance matrix
        self.findDistanceMatrix()
    
    
    def evolveReaggregateStep(self,dt):
        """
        Imagine a line on the sphere along phi=0. For each cell find the 
        closest part of the line and set your great circle towards it and move 
        dt towards it. If you end up further from the line then set position 
        to sitting on the line.
        
        Parameters
        ----------
        dt : int
            The time in seconds that you evolve by.
        """
        # update time
        self.time = self.time + dt
        self.NT += 1
        for c in self.cellsList:
            c.age += dt
        
        # create the line
        Ni = 100
        line = [[self.radius,pi*i/Ni,0] for i in range(Ni)]+[[self.radius,pi,0]]
        line = np.array(line)
        line = u.sph2cart(*line.T)
        
        # for each cell find its relations to the line
        inds = []
        ds = []
        lps = []
        for c in self.cellsList:
            dsts = u.sphereDistance(*line.T,*np.tile(c.pos,(Ni+1,1)).T,self.radius)
            inds.append(np.argmin(dsts))
            ds.append(dsts[inds[-1]])
            lps.append(line[inds[-1]])
        
        # convert to numpy for speed
        inds = np.array(inds)
        ds = np.array(ds)
        lps = np.array(lps)
        
        # required data for the move
        S = np.array([dt*c.vel for c in self.cellsList])
        pos = np.array([c.pos for c in self.cellsList])
        ang = S/self.radius
        ang = np.tile(ang,(3,1)).T
        
        # new direction given by cross product
        gcs = np.cross(pos,lps)
        gcs[np.all(gcs==np.array([0,0,0]),axis=1)] = np.array([1,0,0])
        ang[np.all(gcs==np.array([0,0,0]),axis=1)] = np.array([0])
        gcs = gcs/np.linalg.norm(gcs,axis=1)[:,np.newaxis]
        newPos = u.rotateAboutAxis(pos,gcs,ang)
        
        # if you're now further from the line then go to the line
        newDsts = u.sphereDistance(*lps.T,*newPos.T,self.radius)
        newPos[newDsts>=ds] = lps[newDsts>=ds]
        
        # set the new great circles and positions
        for g,c,p in zip(gcs,self.cellsList,newPos):
            c.gCirc = g
            c.pos = p
        
        # advance epiboly
        self.epiboly += dt*self.epibolyV/100        

        # don't divide cells during this
        #for c in self.cellsList:
        #    if self.time > c.divTime:
        #        c.divideCell()        
        
        # update distance matrix
        self.findDistanceMatrix()
        


    def evolveReaggregateStep2(self,dt):
        """
        Imagine a line on the sphere along phi=0. Each cell moves towards the 
        line by moving purely east-west.
        
        Parameters
        ----------
        dt : int
            The time in seconds that you evolve by.
            
        Notes
        -----
        No need for test for stopping when it gets to line because next step would 
        be in reverse direction if it passes the line.
        """
        # update time
        self.time = self.time + dt
        self.NT += 1
        for c in self.cellsList:
            c.age += dt
        
        angs = []
        for c in self.cellsList:
            _,theta,phi = u.cart2sph(*c.pos)
            S = dt*c.vel
            ang = S/(self.radius*math.sin(theta))
            if phi>0:
                ang = -ang
            angs.append([ang])
            
        gc = np.array([0,0,1])
        angs = np.array(angs)
        pos = np.array([c.pos for c in self.cellsList])
        newPos = u.rotateAboutAxis(pos,gc,angs)
        
        # set the new positions
        for c,p in zip(self.cellsList,newPos):
            c.pos = p
        
        # advance epiboly
        self.epiboly += dt*self.epibolyV/100    
        
        # don't divide cells during this
        #for c in self.cellsList:
        #    if self.time > c.divTime:
        #        c.divideCell()        
        
        # update distance matrix
        self.findDistanceMatrix()
        
        
    def latitudeDensity(self,NSeg=10):
        """
        Finds the cell density on the sphere for equi-latitude segments.
        
        Parameters
        ----------
        NSeg : int
            The number of segments (of equal change in theta, not equal area) 
            that the sphere is divided into.
        returns : list of [theta,density]
            The [theta,density] for every segment, where theta is the angle at 
            the midpoint of the segment and density is the cells per um^2.
        
        """
        allTheta = [u.cart2sph(*c.pos)[1] for c in self.cellsList]
        delt = math.pi/NSeg
        
        densities = []
        
        for i in range(NSeg):
            minT = i*delt
            maxT = (i+1)*delt
            theta = minT + (maxT-minT)/2
            
            # find number of cells
            nCells = sum([aT>minT and aT<maxT for aT in allTheta])
            
            # calculate area of segment
            a1 = 2*math.pi*(self.radius**2)*(1-math.cos(minT))
            a2 = 2*math.pi*(self.radius**2)*(1-math.cos(maxT))
            area = a2 - a1
            
            densities.append([theta,nCells/area])
        
        return densities
    
    
    def measureHomogeneity(self,S=6,R=5):
        """
        A grid based measure of cell position distribution homogeneity (see 
        Schilcherab et al.)
        
        Parameters
        ----------
        S : int
            Will split sphere into S^2 regions. (Really you should take a 
            weighted mean of many S.)
        """
        
        phis = [[2*pi*i/S-pi,2*pi*(i+1)/S-pi] for i in range(S)]
        thetas = [[acos(1-2*i/S),acos(1-2*(i+1)/S)] for i in range(S)]
        cells = [u.cart2sph(*c.pos) for c in self.cellsList]
        
        delRs = []
        X = np.array([1,0,0])
        for r in range(R):
            cells = [u.rotateAboutAxis(c.pos,X,(pi*r)/R) for c in self.cellsList]
            dels = []
            cells = [u.cart2sph(*c) for c in cells]
            for p in phis:
                for t in thetas:
                    count = 0
                    for c in cells:
                        if c[1]>=t[0] and c[1]<t[1] and c[2]>=p[0] and c[2]<p[1]:
                            count += 1
                    dels.append(abs(count - (len(cells)/(S**2))))
            delRs.append(sum(dels)/(2*len(cells)))
        
        return max(delRs)
    
    
    def plotImage(self,
                  resFac=1,
                  figsize=(10,10),
                  save=False,
                  phiLine=False,
                  thetaLine=False,
                  epiboly=False,
                  time=False,
                  tilt1=0,
                  tilt2=0,
                  lineGrad=False,
                  line=None):
        """
        Plots an image of the whole sphere-embryo with cells drawn as circles 
        according to their radii.
        
        Parameters
        ---------
        resFac : int
            Minimal resolution is One point per degree. The resolution will be 
            this minimal resolution multiplied by resFac.
        figsize : (int,int)
            The figure size as in matplotlib.
        save : str or bool
            If str then it should be the file path to save the plot to. The 
            time step will be added to name. This will also stop the output 
            being displayed.
        phi/thetaLine : float of list of floats
            Draw a line of constant phi/theta. One line for each element. 
            Value is in radians and is phi/theta not lon/lat.
        epiboly = bool
            Whether to draw epiboly or not.
        time : bool
            Whether to plot the embryo time (= self.time + 5hrs).
        tilt1/2 : float
            Tilt in degrees of the projection. In degrees.
        lineGrad : bool
            Whether to plot line gradient.
        line : list
            The line to plot a gradient from.            
        """
        # extract cell positions
        pts = np.array([c.pos for c in self.cellsList])

        # make an xyz grid from lon-lat grid, this defines pixel locations
        nrow,ncol = (90*resFac,180*resFac)
        linlon = np.linspace(0,2*pi,ncol)
        linlat = np.linspace(-pi/2,pi/2,nrow)
        lon,lat = np.meshgrid(linlon,linlat)     
        
        rg = np.full(lon.shape,self.radius)
        xg,yg,zg = np.moveaxis(u.lonlat2cart(rg,lon,lat),-1,0)
        
        # this will be the heatmap
        Ti = np.full_like(lon,0)
        if len(self.cellsList)==0:
            radiusC = 1
        else:
            radiusC = list(set([c.radius for c in self.cellsList]))[0]
        v = np.moveaxis([xg,yg,zg],0,-1)
        vnorm = np.linalg.norm(v,axis=2)

        if epiboly:
            thr = pi/2 - pi*(self.epiboly+0.01)
            Ti[lat>thr] = Ti[lat>thr] + 0.2        
        
        for p in pts:
            pnorm = np.linalg.norm(p)
            lengths = self.radius*np.arccos(np.matmul(v,p)/(vnorm*pnorm))
            Ti[lengths<radiusC] = 0.6
        
        lev = None
        if lineGrad:
            dists = np.zeros(lat.shape+(len(line),))
            for i,l in enumerate(line):
                dists[:,:,i] = self.radius*np.arccos(np.matmul(v,l)/(vnorm*pnorm))
            dists = np.min(dists,axis=2)
            dists = 0.45*(1 - dists/(self.radius*pi))
            Ti[Ti!=0.6] = dists[Ti!=0.6]   
            lev = 100
        
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon) 
        
        # set up map projection
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,1,1,projection=ccrs.Orthographic(tilt1,tilt2))
        ax.gridlines()
        ax.contourf(lon,lat,Ti,
                    transform=ccrs.PlateCarree(),
                    cmap='nipy_spectral',vmax=1,vmin=0,levels=lev)
        
        if phiLine:
            if isinstance(phiLine,int) or isinstance(phiLine,float):
                phiLine = [phiLine]
            for p in phiLine:
                phis = [180*p/pi for i in range(180)]
                thetas = [90-i for i in range(180)]
                plt.plot(phis,thetas,
                         color='red',
                         linewidth=2,
                         marker=None,
                         transform=ccrs.Geodetic())      
                
        if thetaLine:
            if isinstance(thetaLine,int) or isinstance(thetaLine,float):
                thetaLine = [thetaLine]
            for t in thetaLine:
                phis = [i for i in range(360)]
                thetas = [180*((pi/2)-t)/pi for i in range(360)]
                plt.plot(phis,thetas,
                         color='red',
                         linewidth=2,
                         marker=None,
                         transform=ccrs.Geodetic())                     
                
        if time:
            timeStr = str((self.time+(3600*5))//3600).zfill(2)
            timeStr += ':'
            timeStr += str(int(((self.time+(3600*5))/60)%60)).zfill(2)
            plt.gcf().text(0.13,0.13,timeStr, fontsize=20)   
            
        if save:
            save = save[:-4]+'_'+str(self.NT)+save[-4:]
            plt.savefig(save)
            plt.close(fig)
        else:
            plt.show()

            
    def plotVoronoi(self,resFac=1,figsize=(10,10),thickness=2,save=False):
        """
        Plots an image of the whole sphere-embryo with cells as a voronoi 
        tesselation.
        
        Prameters
        ---------
        resFac : int
            Minimal resolution is One point per degree. The resolution will be 
            this minimal resolution multiplied by resFac.
        figsize : (int,int)
            The figure size as in matplotlib.
        thickness : float
            The thickness of the edges you draw in um.
        save : str or bool
            If str then it should be the file path to save the plot to. The 
            time step will be added to name. This will also stop the output 
            being displayed.
        """
        
        # extract cell positions
        pts = np.array([c.pos for c in self.cellsList])
        
        center = np.array([0,0,0])
        sv = SphericalVoronoi(pts,self.radius,center)
        
        r1 = sv.regions[0]
        e11 = [sv.vertices[r1[0]],sv.vertices[r1[1]]]
        #  each edge is [p1,p2,gCirc,theta]
        e11 += [np.cross(e11[0],e11[1]),]
        e11[2] = e11[2]/np.linalg.norm(e11[2])
        e11 += u.vecs2ang(e11[0],e11[1])
        
        #pixP
        pixTheta = u.vecs2ang(e11[0],pixP)
        edgeDist = u.vectortoplaneAng(pixP,e11[2])*self.radius
        if edgeDist<thickness/2 and (pixTheta>e11[3] or pixTheta<0):
            #paint black
            pass
        
        # 1. get every edge, remove duplicates
        # 2. get all great circles of edges
        # 3. get all edge lengths and express as angles and divide by resFac*2
        # 4. draw dots at all first points of edges then apply rotation
        # 5. draw dots at new positions and rotate - repeat resFac*2 times
        # 6. start again but after rotating it perp. direction to add thickness
        
        # make an xyz grid from lon-lat grid, this defines pixel locations
        nrow,ncol = (90*resFac,180*resFac)
        linlon = np.linspace(0,2*pi,ncol)
        linlat = np.linspace(-pi/2,pi/2,nrow)
        lon,lat = np.meshgrid(linlon,linlat)
        rg = np.full(lon.shape,self.radius)
        xg,yg,zg = np.moveaxis(u.lonlat2cart(rg,lon,lat),-1,0)
        
        # this will be the heatmap
        Ti = np.full_like(lon,-200)
        
        radiusC = list(set([c.radius for c in self.cellsList]))[0]
        v = np.moveaxis([xg,yg,zg],0,-1)
        vnorm = np.linalg.norm(v,axis=2)
        for p in pts:
            pnorm = np.linalg.norm(p)
            lengths = self.radius*np.arccos(np.matmul(v,p)/(vnorm*pnorm))
            Ti[lengths<radiusC] = 100   
        
        # set up map projection
        fig = plt.figure(figsize=(10,10))
        map = Basemap(projection='ortho', lat_0=30, lon_0=0)
        # draw lat/lon grid lines every 30 degrees.
        map.drawmeridians(np.arange(0, 360, 30))
        map.drawparallels(np.arange(-90, 90, 30))
        # compute native map projection coordinates of lat/lon grid.
        x, y = map(lon*180/pi,lat*180/pi)
        # contour data over the map.
        cs = map.contourf(x,y,Ti,2)
        if save:
            save = save[:-4]+'_'+str(self.NT)+save[-4:]
            plt.savefig(save)
            plt.close(fig)
        else:
            plt.show()
            