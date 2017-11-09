import subprocess
import numpy as np
import matplotlib.pyplot as plt
class Square_Skyrmion_Annealing(object):
    def __init__(self,size,J,K,B,D,n_warmup_cycles=5*10**6,beta=0.01,beta_step=2,iter_step=10**4):
        self.n_warmup_cycles=n_warmup_cycles;self.beta=beta;self.beta_step=beta_step;self.iter_step=iter_step
        self.size=size;self.J=J;self.K=K;self.B=B;self.D=D
    def solve(self):
        command_line="./Square_Skyrmion_Annealing"
        for argument in [" "+str(value) for value in self.__dict__.values()]:
            command_line+=argument
        process=subprocess.Popen(command_line, shell=True)
        out,error=process.communicate()
    def results(self):
        with open('magnetization.dat') as f:
            content = f.readlines()
        i=0
        while i < len(content):
            content[i]=content[i].replace('\n','')
            content[i]=content[i].replace('[','')
            content[i]=content[i].replace(']','')
            if content[i]=='':
                del content[i]
            else:
                content[i]=np.array(content[i].split(',')).astype(np.float)
                i+=1
        self.Sx=np.array(content[:self.size]);
        self.Sy=np.array(content[self.size:2*self.size]);
        self.Sz=np.array(content[-self.size:]);
    def plot_lattice_real(self):
        X, Y = np.meshgrid(range(self.size), range(self.size))
        plt.figure(figsize=(8,8))
        plt.title('Real')
        plt.imshow(self.Sz)
        plt.quiver(X, Y, self.Sx, self.Sy, units='width')
        plt.show()
    def plot_lattice_staggered(self):
        indexes=np.arange(self.size)
        X, Y = np.meshgrid(range(self.size), range(self.size))
        self.Sz_staggered=np.copy(self.Sz)
        for i in range(self.size):
            if i%2==0:
                self.Sz_staggered[i][indexes%2==0]*=-1;
            else:
                self.Sz_staggered[i][indexes%2!=0]*=-1;
        plt.figure(figsize=(8,8))
        plt.title('Staggered')
        plt.imshow(self.Sz_staggered)
        plt.quiver(X, Y, self.Sx, self.Sy, units='width')
        plt.show()
    def plot_lattice_fft_real(self):
        plt.figure(figsize=(8,8))
        plt.title('Real fourier')
        plt.imshow(np.abs(np.fft.fft2(self.Sz).real))
        plt.show()
