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
        # optimizer = getattr(torch.optim, self.optimizer)(
                # parameters, **self.optimizer_args)
        import torch_optimizer as optim
        optimizer = optim.RAdam(parameters, lr=0.000005)
        if hasattr(self.learning_rate, 'prepare'):
            self.learning_rate.prepare(optimizer)
        return optimizer
