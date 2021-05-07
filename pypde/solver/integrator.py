class Integrator():
    def __init__(self):
        self.time = 0.0

    def update(self):
        raise NotImplementedError
        
    def iterate(self,maxtime,*args,**kwargs):
        ''' Iterate till max time'''
        while self.time<maxtime:
            
            self.update(*args,**kwargs)
            self.update_time()

            if self.tsave is not None:
                if ( (self.time+1e-3*self.dt)%self.tsave<self.dt*0.5):
                    self.save()
                    print("Time: {:5.3f}".format(self.time))

                # if self.field.check(): 
                #     print("\nNan or large value detected! STOP\n")
                #     break

    def update_time(self):
        self.field.t += self.dt
        self.time += self.dt
        
    def save(self):
        self.field.save()