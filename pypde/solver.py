from .utils.abstract import ABCMeta, abstract_attribute
from .operator import OperatorImplicit
import numpy as np 

class SolverBase(metaclass=ABCMeta):

    def __get__(self, *args, **kwargs):
            raise NotImplementedError("Not implemented on child")

    def update(self):
        raise NotImplementedError

    def set_bc(self):
        raise NotImplementedError

    @abstract_attribute
    def t(self):
        pass

    @abstract_attribute
    def dt(self):
        pass

    @abstract_attribute
    def v(self):
        pass

    @abstract_attribute
    def fields(self):
        pass

    def iterate(self,maxtime):
        ''' Iterate till max time'''
        while self.t<maxtime:
            
            self.update()
            
            if (self.t%self.tsave<self.dt):
                self.save()
                print("Time: {:5.3f}".format(self.t))

                if self.check_field(): 
                    print("\nNan or large value detected! STOP\n")
                    break

    def update_time(self):
        for key in self.fields:
            self.fields[key].t += self.dt
        self.t += self.dt

    def save(self):
        for key in self.fields:
            self.fields[key].save()

    def check_field(self):
        return any( [self.fields[key].check() 
            for key in self.fields])

class SolverExplicit(SolverBase):
    ''' 
    Class for explicit approach
    '''
    def rhs(self):
        raise NotImplementedError

    def update(self):
        ''' 
        Update pde by 1 timestep 
        '''
        self._set_bc()
        self.v += self.dt*self._rhs()
        self.update_time()


class SolverImplicit(SolverBase):
    ''' 
    Class for implicit approach
    '''
    def lhs(self):
        raise NotImplementedError

    def rhs(self):
        raise NotImplementedError

    def update(self):
        ''' 
        Update pde by 1 timestep 
        '''
        self.set_bc()
        Op = self.lhs(self.dt)
        if not isinstance(Op, list): Op = [Op]
        self.v = (self.v+self.dt*self.rhs())
        for O in Op:
            self.v = O.solve( self.v )
        self.update_time()