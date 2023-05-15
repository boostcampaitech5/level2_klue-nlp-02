import torch

class WarmupConstantLR(torch.optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        

        def lr_lambda(step):
            step+=1

            if step < warmup_steps+1:
                return float(step) / float(max(1.0, warmup_steps+1))
            return 1.

        super(WarmupConstantLR, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)