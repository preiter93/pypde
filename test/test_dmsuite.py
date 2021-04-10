import numpy as np
from pypde.bases.dmsuite import *
import unittest

N = 100    # Grid size
EPS = 1e-2 # Tolerance

class TestDMSuiteCheby(unittest.TestCase):
    def setUp(self):
        
        self.L = 1.0           # Domain size
        self.x,_ = chebdif(N,1,L=self.L) # Obtain x
        
        arg = 2*np.pi/self.L
        self.y = np.sin(np.pi*self.x)
        self.y = np.sin(arg*self.x)
        self.dy = []  # deriv 1-4
        self.dy.append(  arg**1*np.cos(arg*self.x) )
        self.dy.append( -arg**2*np.sin(arg*self.x) )
        self.dy.append( -arg**3*np.cos(arg*self.x) )
        self.dy.append(  arg**4*np.sin(arg*self.x) )


    def test_chebdif(self):
        print("-------------------")
        print("     chebdif       ")
        print("-------------------")
        for i,dy in enumerate(self.dy):
            
            # Calculate derive
            _,D  = chebdif(N,i+1,L=self.L)
            dy_c = D@self.y

            # Calculate norm
            norm = np.linalg.norm(dy_c-dy)/N
            print("{:1d}. deriv: Norm |y-ya| {:5.2e}"
            .format(i+1,norm))

            # Assert
            assert norm<EPS

    def test_chebdifft(self):
        print("-------------------")
        print("     chebdifft     ")
        print("-------------------")
        for i,dy in enumerate(self.dy):
            
            # Calculate derive
            dy_c = chebdifft(self.y,i+1,L=self.L)

            # Calculate norm
            norm = np.linalg.norm(dy_c-dy)/N
            print("{:1d}. deriv: Norm |y-ya| {:5.2e}"
            .format(i+1,norm))

            # Assert
            assert norm<EPS

class TestDMSuiteFourier(unittest.TestCase):
    def setUp(self):
        
        self.L = 1.0           # Domain size
        self.x,_ = fourdif(N,1,L=self.L) # Obtain x
        
        arg = 1.0*2*np.pi/self.L
        self.y = np.sin(arg*self.x)
        self.dy = []  # deriv 1-4
        self.dy.append(  arg**1*np.cos(arg*self.x) )
        self.dy.append( -arg**2*np.sin(arg*self.x) )
        self.dy.append( -arg**3*np.cos(arg*self.x) )
        self.dy.append(  arg**4*np.sin(arg*self.x) )


    def test_fourdif(self):
        print("-------------------")
        print("     fourdif       ")
        print("-------------------")
        for i,dy in enumerate(self.dy):
            
            # Calculate derive
            _,D  = fourdif(N,i+1,L=self.L)
            dy_c = D@self.y

            # Calculate norm
            norm = np.linalg.norm(dy_c-dy)/N
            print("{:1d}. deriv: Norm |y-ya| {:5.2e}"
            .format(i+1,norm))

            # Assert
            assert norm<EPS

    def test_fourdifft(self):
        print("-------------------")
        print("     chebdifft     ")
        print("-------------------")
        for i,dy in enumerate(self.dy):
            
            # Calculate derive
            dy_c = fourdifft(self.y,i+1,L=self.L)

            # Calculate norm
            norm = np.linalg.norm(dy_c-dy)/N
            print("{:1d}. deriv: Norm |y-ya| {:5.2e}"
            .format(i+1,norm))

            # Assert
            assert norm<EPS

if __name__ == '__main__':
    unittest.main()

# import matplotlib.pyplot as plt 
# plt.plot(self.x,dy  ,"k")
# plt.plot(self.x,dy_c,"r--")
# plt.show()