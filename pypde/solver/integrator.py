from ..field import Field, MultiField

class Integrator():
    def __init__(self):
        self.time = 0.0

    def update(self):
        raise NotImplementedError

    def callback(self):
        pass

    def iterate(self,maxtime,*args,**kwargs):
        ''' Iterate till max time'''
        eps = 1e-3*self.dt
        while (self.time+eps)<maxtime:

            self.update(*args,**kwargs)
            self.update_time()

            if self.tsave is not None:
                if ( (self.time+eps)%self.tsave<self.dt*0.5):
                    self.save()
                    print("Time: {:5.3f}".format(self.time))

                    self.callback()

                # if self.field.check():
                #     print("\nNan or large value detected! STOP\n")
                #     break

    def update_time(self):
        if isinstance(self.field,Field):
            self.field.t += self.dt
        elif isinstance(self.field,MultiField):
            self.field.update_time(self.dt)
        else:
            raise ValueError("fields must be of type Field or MultiField.")
        self.time += self.dt

    def save(self):
        self.field.save()
