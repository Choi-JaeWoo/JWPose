import torch

class HMLoss(torch.nn.Module):
    def __init__(self):
        super(HMLoss, self).__init__()
    def forward(self, hm_preds, hm):
        l = ((hm_preds-hm)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1).mean()
        return l
if __name__ == '__main__':
    print("HI")