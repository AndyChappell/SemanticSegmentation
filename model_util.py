import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable

def get_exponential_factor(start, stop, n, epoch):
    ratio = stop / start
    factor = ratio ** (1. / (n - 1))
    return factor ** epoch

def get_linear_factor(start, stop, n, epoch):
    target = start + (epoch / (n - 1.)) * (stop - start)
    return target / start

class LRFinder():
    def __init__(self, model, bunch, loss_fn, optim, num_iter, void_code,
                 low_lr=1e-7, high_lr=2, is_exponential=True):
        self.num_iter = num_iter
        if is_exponential:
            self.lambda_func = lambda batch : get_exponential_factor(low_lr, high_lr, self.num_iter, batch)
        else:
            self.lambda_func = lambda batch : get_linear_factor(low_lr, high_lr, self.num_iter, batch)
        self.scheduler = LambdaLR(optim, lr_lambda=self.lambda_func)
        self.learner = UNetLearner(model, bunch, loss_fn, optim, self.scheduler, void_code)
    
    def find(self):
        num_epochs = int(np.floor(self.num_iter/len(self.learner.bunch.train_dl)))
        for epoch in list(range(num_epochs)):
            # This is ugly, finder needs to know too much about learner
            train_dl = self.learner.bunch.train_dl
            for i, batch in enumerate(train_dl):
                self.learner.is_training = True
                self.learner._batch(epoch, i, batch)
                self.learner.evaluate(0)
    
    def get_lrs(self):
        return torch.Tensor(self.learner.history["lr"]).cpu()

    def get_losses(self):
        num_batches = len(self.learner.bunch.valid_dl)
        x = torch.Tensor(self.learner.history["val_loss"])
        x = x.reshape(x.size(0) // num_batches, num_batches).mean(axis = 1)
        return x.cpu()

    def get_accuracies(self):
        num_batches = len(self.learner.bunch.valid_dl)
        x = torch.Tensor(self.learner.history["val_acc"])
        x = x.reshape(x.size(0) // num_batches, num_batches).mean(axis = 1)
        return x.cpu()
