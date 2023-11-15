"""
Code referenced from: 
https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyLoss(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.kl_loss = nn.KLDivLoss(log_target=True)

    def forward(self, student_outputs, teacher_outputs):
        student_outputs = F.log_softmax(student_outputs / self.T, dim=1)
        teacher_outputs = F.log_softmax(teacher_outputs / self.T, dim=1)
        loss = self.kl_loss(student_outputs, teacher_outputs) * (self.T ** 2)
        return loss