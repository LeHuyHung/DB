import torch

from concern.config import Configurable, State


class OptimizerScheduler(Configurable):
    optimizer = State()
    optimizer_args = State(default={})
    learning_rate = State(autoload=False)

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.load('learning_rate', cmd=cmd, **kwargs)
        if 'lr' in cmd:
            self.optimizer_args['lr'] = cmd['lr']

    def create_optimizer(self, parameters):
        #optimizer = getattr(torch.optim, self.optimizer)(
        #        parameters, **self.optimizer_args)
        # from ranger import Ranger  # this is from ranger.py
        # from ranger import RangerVA  # this is from ranger913A.py
        # from ranger import RangerQH  # this is from rangerqh.py
        # Each of the Ranger, RangerVA, RangerQH have different parameters.
        
        print("=======optimizer=====")
        import torch_optimizer as optim
        optimizer = optim.RAdam(parameters, lr=0.005)
        #optimizer = Ranger(parameters,alpha=0.5)        
        print(optimizer)
        if hasattr(self.learning_rate, 'prepare'):
            self.learning_rate.prepare(optimizer)
        return optimizer
