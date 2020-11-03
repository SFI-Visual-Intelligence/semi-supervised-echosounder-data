import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_accu_max = 0
        self.delta = delta
        self.trace_func = trace_func
        self.path = os.path.join(
            os.getcwd(),
            'checkpoints_early', 'checkpoint_early.pth.tar')
        if not os.path.isdir(os.path.join(os.getcwd(), 'checkpoints_early')):
            os.makedirs(os.path.join(os.getcwd(), 'checkpoints_early'))

    def __call__(self, val_accu, epoch, args, model, optimizer_body, optimizer_category):
        score = val_accu
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_accu, epoch, args, model, optimizer_body, optimizer_category)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_accu, epoch, args, model, optimizer_body, optimizer_category)
            self.counter = 0

    def save_checkpoint(self, val_accu, epoch, args, model, optimizer_body, optimizer_category):
        '''Saves model when validation increase increase.'''
        if self.verbose:
            self.trace_func(f'Validation accu increased ({self.val_accu_max:.6f} --> {val_accu:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer_body': optimizer_body.state_dict(),
                    'optimizer_category': optimizer_category.state_dict(),
                    }, self.path)
        self.val_accu_max = val_accu
