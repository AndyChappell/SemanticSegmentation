import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable

def get_factor(start, stop, n):
    ratio = stop / start
    factor = ratio ** (1 / (n - 1))
    return factor

def get_running_factor(start, stop, n, epoch):
    factor = get_factor(start, stop, n)
    return factor ** epoch

def multiplicative_slice(start, stop, n):
    factor = get_factor(start, stop, n)
    slices = [start * (factor ** i) for i in range(n)]
    return np.array(slices)

def ewma(current, new, alpha=0.1):
    return current + alpha * (new - current)

class LRFinder():
    def __init__(self, optim, num_iter):
        self.optim = optim
        self.num_iter = num_iter
        self.lambda_func = lambda batch : get_running_factor(1e-7, 10, self.num_iter, batch)
        self.scheduler = LambdaLR(optim, lr_lambda=self.lambda_func)
        self.learning_rates = []
        self.losses = []
    
    def find(self, model, input, loss_fn):
        smooth_loss = None
        best_loss = np.Inf
        stop_training = False
        num_epochs = int(np.floor(self.num_iter/len(input)))
        for epoch in list(range(num_epochs)):
            if stop_training:
                break
            model.train()
            self.optim.zero_grad()
            for i, batch in enumerate(input):
                print("Batch {} has learning rate {}".format(i, self.scheduler.get_lr()))
                images, masks = batch
                x = Variable(images)
                y = Variable(masks)
                pred = model.forward(x)

                loss = loss_fn(pred, y)
                if smooth_loss is not None:
                    smooth_loss = ewma(smooth_loss, loss.item())
                else:
                    smooth_loss = loss.item()
                if smooth_loss < best_loss:
                    best_loss = smooth_loss
                if smooth_loss > 4 * best_loss or np.isnan(smooth_loss):
                    stop_training = True
                    break

                self.learning_rates.append(self.scheduler.get_lr())
                self.losses.append(loss.item())

                loss.backward()
                self.optim.step()
                self.scheduler.step()
        self.learning_rates = np.array(self.learning_rates).flatten()
        self.losses = np.array(self.losses)
    
    def get_learning_rates(self):
        return self.learning_rates

    def get_losses(self):
        return self.losses

#niter = 100
#optim = opt.Adam(model.parameters(), lr=1e-7)
#lr_lambda = lambda batch : get_running_factor(1e-7, 10, niter, batch)
#lr_finder.find(model = model, input = bunch.train_dl, loss_fn=loss_fn)