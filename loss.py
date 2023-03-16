from turtle import forward
import torch 
import torch.nn as nn

class softdiceloss(nn.Module):
    def __init__(self):
        super(softdiceloss, self).__init__()
    
    def forward(self, probs, targets):
        num = targets.size()[0]
        smooth = 1e-5

        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1*m2
        N_dice_eff = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - N_dice_eff.sum() / num
        return score



