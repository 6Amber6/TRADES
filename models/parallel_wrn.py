import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wideresnet_update import WideResNet


class WRNWithEmbedding(WideResNet):
    def forward(self, x, return_embedding=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        emb = out.view(out.size(0), -1)
        logits = self.fc(emb)
        if return_embedding:
            return emb, logits
        return logits


class ParallelFusionWRN(nn.Module):
    def __init__(self, model4, model6):
        super().__init__()
        self.m4 = model4
        self.m6 = model6

        for p in self.m4.parameters():
            p.requires_grad = False
        for p in self.m6.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(640 * 2, 10)

    def forward(self, x, return_aux=False):
        e4, out4 = self.m4(x, return_embedding=True)
        e6, out6 = self.m6(x, return_embedding=True)
        emb = torch.cat([e4, e6], dim=1)
        out = self.fc(emb)
        if return_aux:
            return out4, out6, out
        return out
