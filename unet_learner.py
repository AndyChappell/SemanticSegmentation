SHOWER = 1
TRACK = 2

def switch_batch_norm(model, track_running_stats):
    for group in model.children():
        if type(group) == nn.Sequential:
            for child in group:
                if type(child) == nn.BatchNorm2d:
                    child.track_running_stats = track_running_stats

def summarize_epoch(history, n_train_batches, n_valid_batches):
    epoch = len(history["train_loss"]) // n_train_batches
    train_loss = torch.Tensor(history["train_loss"][-n_train_batches:]).mean()
    train_acc = torch.Tensor(history["train_acc"][-n_train_batches:]).mean()
    valid_loss = torch.Tensor(history["val_loss"][-n_valid_batches:]).mean()
    valid_acc = torch.Tensor(history["val_acc"][-n_valid_batches:]).mean()
    valid_acc_shower = torch.Tensor(history["val_acc_shower"][-n_valid_batches:]).mean()
    valid_acc_track = torch.Tensor(history["val_acc_track"][-n_valid_batches:]).mean()

    print("Epoch {} : Training Loss {:.3f} Acc {:.3f} Validation Loss {:.3f} " \
          "Acc {:.3f} T Acc {:.3f} S Acc {:.3f}".format(epoch, train_loss,
          train_acc, valid_loss, valid_acc, valid_acc_track, valid_acc_shower))

def save_network(model, input, filename):
    eval_model = model.eval()
    torch.save(eval_model.state_dict(), f"{filename}.pkl")
    # Load later with:
    # model.load_state_dict(torch.load(PATH))
    # model.eval()

    sm = torch.jit.trace(eval_model, input)
    sm.save(f"{filename}_traced.pt")

    #sm = torch.jit.script(model)
    #sm.save(f"{filename}_script.pt")

class UNetLearner:
    def __init__(self, model, bunch, loss_fn, optim, scheduler, void_code=0):
        self.model = model
        self.bunch = bunch
        self.loss_fn = loss_fn
        self.optim = optim
        self.scheduler = scheduler
        self.void_code = void_code
        self.is_training = True
        self.verbose = 0
        self.history = {"lr" : [], "train_loss" : [], "train_acc" : [],
                        "train_acc_shower" : [], "train_acc_track" : [],
                        "val_loss" : [], "val_acc" : [],
                        "val_acc_shower" : [], "val_acc_track" : []}

    def train(self, n_epochs):
        self.model = self.model.cuda()
        for e in range(n_epochs):
            # Train
            self.is_training = True
            self.model = self.model.train()
            self._epoch(e)
            # Save the network
            for batch in self.bunch.valid_dl:
                example, _ = batch
                example = Variable(example).cuda()
                save_network(model, example, f"unet_{e}")

            # Validate
            self.model = self.model.eval()
            switch_batch_norm(self.model, False)
            with torch.no_grad():
                self.evaluate(e)
            switch_batch_norm(self.model, True)
            summarize_epoch(self.history, len(self.bunch.train_dl),
                            len(bunch.valid_dl))
    
    def evaluate(self, e):
        self.is_training = False
        self._epoch(e)

    def _batch(self, e, b, batch):
        images, truth = batch
        x = Variable(images).cuda()
        y = Variable(truth).cuda()
        pred = self.model.forward(x)
        loss = self.loss_fn(pred, y)

        key = "train" if self.is_training else "val"
        if self.is_training:
            self.history["lr"].append(self.scheduler.get_lr()[0])
        self.history[f"{key}_loss"].append(loss.item())
        self.history[f"{key}_acc"].append(self.accuracy(pred, y))
        self.history[f"{key}_acc_track"].append(self.accuracy(pred, y, TRACK))
        self.history[f"{key}_acc_shower"].append(self.accuracy(pred, y, SHOWER))

        if not self.is_training and self.verbose > 0:
            if self.verbose > 1: #and b == (len(self.bunch.valid_dl) - 1):
                net_input = x.cpu().detach().numpy()
                net_pred = pred.cpu().detach().numpy()
                net_mask = y.cpu().detach().numpy()
                show_batch(e, b, net_input, net_pred, net_mask, self.void_code,
                            self.is_training, n = images.shape[0], randomize=False)
                label = "Training" if self.is_training else "Validation"
                print("Batch {}: {} Loss: {:.3f} Acc: {:.3f} S Acc: {:.3f} T Acc: {:.3f}".format(
                    b + 1, label, loss.item(), self.history[f"{key}_acc"][-1],
                    self.history[f"{key}_acc_shower"][-1], self.history[f"{key}_acc_track"][-1]))
            elif e == 4 and b == 0:    # Should add epochs to self on call to train
                net_input = x.cpu().detach().numpy()
                net_pred = pred.cpu().detach().numpy()
                net_mask = y.cpu().detach().numpy()
                #iu.show_batch(e, b, net_input, net_pred, net_mask, self.void_code,
                #            self.is_training, n = images.shape[0], randomize=False)
                show_batch(e, b, net_input, net_pred, net_mask, self.void_code,
                            self.is_training, n = images.shape[0], randomize=False)
                label = "Training" if self.is_training else "Validation"
                print("Batch {}: {} Loss: {:.3f} Acc: {:.3f} S Acc: {:.3f} T Acc: {:.3f}".format(
                    b + 1, label, loss.item(), self.history[f"{key}_acc"][-1],
                    self.history[f"{key}_acc_shower"][-1], self.history[f"{key}_acc_track"][-1]))
            elif e == 10:
                net_input = x.cpu().detach().numpy()
                net_pred = pred.cpu().detach().numpy()
                net_mask = y.cpu().detach().numpy()
                label = "Training" if self.is_training else "Validation"
                print("Batch {}: {} Loss: {:.3f} Acc: {:.3f} S Acc: {:.3f} T Acc: {:.3f}".format(
                    b + 1, label, loss.item(), self.history[f"{key}_acc"][-1],
                    self.history[f"{key}_acc_shower"][-1], self.history[f"{key}_acc_track"][-1]))
            #elif e == 4:
            #    net_input = x.cpu().detach().numpy()
            #    net_pred = pred.cpu().detach().numpy()
            #    net_mask = y.cpu().detach().numpy()
            #    show_batch(e, b, net_input, net_pred, net_mask, self.void_code,
            #                self.is_training, n = images.shape[0], randomize=False)

        if self.is_training:
            loss.backward()
            self.optim.step()
            self.scheduler.step()
            self.optim.zero_grad()

    def _epoch(self, epoch):
        dl = self.bunch.train_dl if self.is_training else self.bunch.valid_dl
        for i, batch in enumerate(dl):
            self._batch(epoch, i, batch)
    
    def set_verbose(self, verbose):
        self.verbose = verbose

    def accuracy(self, input, truth, type=None):
        target = truth.squeeze(1)
        if type is None:
            mask = target != self.void_code
        else:
            mask = target == type
        return (input.argmax(dim=1)[mask] == target[mask]).float().mean()
