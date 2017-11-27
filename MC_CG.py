
"""
Monte Carlo and Conjugate gradient methods calculator of the ground state of the Square Classical Heisenberg Lattice.
Author: Yaroslav V. Zhumagulov (2017), yaroslav.zhumagulov@gmail.com
"""
# ----------------------------------------------------------------------
import numpy as np
import progressbar
import matplotlib.pyplot as plt
from numba import jit
from scipy.optimize import minimize
from skimage.feature.peak import peak_local_max
# ----------------------------------------------------------------------
@jit(nopython=True)
def energy(configuration,L,J,K,D,B):
    lattice=np.zeros((3,L,L))
    theta=configuration[:L*L];phi=configuration[-L*L:];
    theta=theta.reshape((L,L));phi=phi.reshape((L,L));
    lattice[0]=np.sin(theta)*np.cos(2*phi);
    lattice[1]=np.sin(theta)*np.sin(2*phi);
    lattice[2]=np.cos(theta);
    Energy=0
    for i in range(L):
        for j in range(L):
            for k in range(3):
                Energy+=J*(lattice[k,i,j]*(lattice[k,(i+1)%L,j]+lattice[k,i,(j+1)%L]))
            Energy+=K*lattice[2,i,j]**2;
            Energy-=B*lattice[2,i,j];
            Energy-=D*(lattice[2,i,j]*lattice[0,(i+1)%L,j]-lattice[0,i,j]*lattice[2,(i+1)%L,j]);
            Energy+=D*(lattice[1,i,j]*lattice[2,i,(j+1)%L]-lattice[2,i,j]*lattice[1,i,(j+1)%L]);
    return Energy/(L*L)
# ----------------------------------------------------------------------
@jit(nopython=True)
def spin_flip(lattice,beta,L,J,K,D,B):
    i,j=np.random.randint(L),np.random.randint(L)
    theta,phi=np.random.random()*np.pi,np.random.random()*np.pi

    spin=np.zeros(3);
    spin[0]=np.sin(theta)*np.cos(2*phi);
    spin[1]=np.sin(theta)*np.sin(2*phi);
    spin[2]=np.cos(theta);

    delta_energy=0
    for k in range(3): delta_energy+=J*(spin[k]-lattice[k,i,j])*(lattice[k,(i+1)%L,j]+lattice[k,i,(j+1)%L])
    delta_energy+=K*(spin[2]**2-lattice[2,i,j]**2);
    delta_energy-=B*(spin[2]-lattice[2,i,j]);
    delta_energy-=D*((spin[2]-lattice[2,i,j])*lattice[0,(i+1)%L,j]-(spin[0]-lattice[0,i,j])*lattice[2,(i+1)%L,j]);
    delta_energy+=D*((spin[1]-lattice[1,i,j])*lattice[2,(i+1)%L,j]-(spin[2]-lattice[2,i,j])*lattice[1,(i+1)%L,j]);

    if delta_energy<0:
        lattice[:,i,j]=spin
        return 1,lattice
    elif np.exp(-beta*delta_energy)>=np.random.random():
        lattice[:,i,j]=spin
        return 1,lattice
    else:
        return 0,lattice




class SquareClassicalHeisenbergLattice (object):
# ----------------------------------------------------------------------
#    Square Classical Heisenberg Lattice
# ----------------------------------------------------------------------
    def __init__(self,L,J,K,D,B):
        self.L=L;self.J=J;self.K=K;self.D=D;self.B=B
        self.configuration=np.random.random((2*self.L**2))
        self._lattice()
# ----------------------------------------------------------------------
    def _lattice(self):
        self.lattice=np.zeros((3,self.L,self.L))
        theta=self.configuration[:self.L*self.L];phi=self.configuration[-self.L*self.L:]
        theta=theta.reshape((self.L,self.L));phi=phi.reshape((self.L,self.L))
        self.lattice[0]=np.sin(theta)*np.cos(2*phi);
        self.lattice[1]=np.sin(theta)*np.sin(2*phi);
        self.lattice[2]=np.cos(theta);
# ----------------------------------------------------------------------
    def _configuration(self):
        self.configuration=np.zeros((2*L**2))
        buff=self.lattice[1].flatten()/self.lattice[0].flatten();
        buff[np.isnan(buff)]=10**8;
        self.configuration[-self.L*self.L:]=np.arctan(buff);
        self.configuration[:self.L*self.L]=np.arccos(self.lattice[2].flatten());
# ----------------------------------------------------------------------
    def _MonteCarlo(self,beta,beta_steps,time_steps):
        for step in range(beta_steps):
            success_steps=0
            while success_steps<time_steps:
                success,self.lattice=spin_flip(self.lattice,beta/(beta_steps-step),self.L,self.J,self.K,self.D,self.B)
                success_steps+=success
# ----------------------------------------------------------------------
    def _ConjugateGradient(self):
        result=minimize(energy, self.configuration, args=(self.L,self.J,self.K,self.D,self.B), method='CG')
        self.configuration=result.x
# ----------------------------------------------------------------------
    def ground_state(self,niter=10):
        self.E=np.inf;final_result=None
        bar = progressbar.ProgressBar()
        for i in bar(range(niter)):
            self._MonteCarlo(beta=200,beta_steps=200,time_steps=10*self.L*self.L)
            self._ConjugateGradient()
            E=energy(self.configuration,self.L,self.J,self.K,self.D,self.B)
            if E<self.E:
                self.E=E
                final_result=np.copy(self.configuration)
        self.configuration=final_result
        self._lattice()
# ----------------------------------------------------------------------
    def plot_lattice(self):
        plt.figure(figsize=(8,8))
        plt.imshow(self.lattice[2])
        plt.colorbar()
        plt.clim([-1,1])
        X,Y=np.meshgrid(range(self.L),range(self.L))
        plt.quiver(X, Y, self.lattice[0], self.lattice[1], units='width')
# ----------------------------------------------------------------------
    def plot_fourier(self):
        self.fourier=np.abs(np.fft.fft2(self.lattice[2]))
        plt.figure(figsize=(8,8))
        plt.imshow(self.fourier)
# ----------------------------------------------------------------------
    def detect_phase(self):
        self.fourier=np.abs(np.fft.fft2(self.lattice[2]))
        self.peaks=peak_local_max(self.fourier,threshold_rel=0.9)
        if len(self.peaks)==1:
            if self.E<(-2*self.J):
                self.phase='antiferromagnet'
            else:
                self.phase='spin-flip'
        elif len(self.peaks)==2:
            self.phase='spiral'
        elif len(self.peaks)==4:
            self.phase='2q'
        else:
            self.phase='unknown'
        return self.phase
# ----------------------------------------------------------------------
    def detect_skyrmion(self):
        nx=np.zeros_like(self.lattice)
        ny=np.zeros_like(self.lattice)

        nx[0]=np.gradient(self.lattice[0],axis=0)
        nx[1]=np.gradient(self.lattice[1],axis=0)
        nx[2]=np.gradient(self.lattice[2],axis=0)
        ny[0]=np.gradient(self.lattice[0],axis=1)
        ny[1]=np.gradient(self.lattice[1],axis=1)
        ny[2]=np.gradient(self.lattice[2],axis=1)

        self.w=0
        for i in range(self.L):
            for j in range(self.L):
                D=np.zeros((3,3))
                D[0]=self.lattice[:,i,j]
                D[1]=nx[:,i,j];D[2]=ny[:,i,j]
                self.w+=np.linalg.det(D.T)/(4*np.pi)
        return self.w
