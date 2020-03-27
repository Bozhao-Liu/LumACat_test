class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __call__(self):
        return self.avg

def UnevenWeightBCE_loss(outputs, labels, weights = (1, 1)):
    '''
    Cross entropy loss with uneven weigth between positive and negative result to manually adjust precision and recall
    '''
    loss = [torch.sum(torch.add(weights[0]*torch.mul(labels[:, i],torch.log(outputs[:, i])), weights[1]*torch.mul(1 - labels[:, i],torch.log(1 - outputs[:, i])))) for i in range(outputs.shape[1])]
    return -torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def Exp_UEW_BCE_loss(outputs, labels, weights = (1, 1)):
    '''
    Cross entropy loss with uneven weigth between positive and negative result, add exponential function to positive to manually adjust precision and recall
    '''
    loss = [torch.sum(torch.add(weights[0]*torch.exp(-torch.mul(labels[:, i],torch.log(outputs[:, i])))-1, -weights[1]*torch.mul(1 - labels[:, i],torch.log(1 - outputs[:, i])))) for i in range(outputs.shape[1])]
    return torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def save_AUC(params, model_dir, network, CViter, data, label):
    return 0
