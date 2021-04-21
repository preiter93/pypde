class SolverBase():
    
    def update(self):
        raise NotImplementedError
        
    def iterate(self,maxtime):
        ''' Iterate till max time'''
        while self.time<maxtime:
            
            self.update()
            
            if (self.time%self.tsave<self.dt):
                self.save()
                print("Time: {:5.3f}".format(self.time))

                if self.field.check(): 
                    print("\nNan or large value detected! STOP\n")
                    break
    
    def update_time(self):
        self.field.t += self.dt
        self.time += self.dt
        
    def save(self):
        self.field.save()