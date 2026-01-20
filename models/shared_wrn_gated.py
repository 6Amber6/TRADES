import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.wideresnet_update import BasicBlock, NetworkBlock


class SharedWRNGated(nn.Module):
    """
    WRN-34-10 with shared backbone (conv1 + block1 + block2),
    split block3 heads, and a lightweight gating network.
    """
    def __init__(self,
                 depth=34,
                 widen_factor=10,
                 num_vehicle=4,
                 num_animal=6,
                 gate_hidden=128,
                 dropRate=0.0,
                 vehicle_classes=(0, 1, 8, 9),
                 animal_classes=(2, 3, 4, 5, 6, 7)):
        super().__init__()
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6

        self.vehicle_classes = list(vehicle_classes)
        self.animal_classes = list(animal_classes)

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # Shared backbone
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], BasicBlock, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], BasicBlock, 2, dropRate)

        # Split heads (block3)
        self.block3_vehicle = NetworkBlock(n, nChannels[2], nChannels[3], BasicBlock, 2, dropRate)
        self.block3_animal = NetworkBlock(n, nChannels[2], nChannels[3], BasicBlock, 2, dropRate)

        self.bn_vehicle = nn.BatchNorm2d(nChannels[3])
        self.bn_animal = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        self.fc_vehicle = nn.Linear(nChannels[3], num_vehicle)
        self.fc_animal = nn.Linear(nChannels[3], num_animal)

        # Gating network on shared features
        self.gate_fc1 = nn.Linear(nChannels[2], gate_hidden)
        self.gate_fc2 = nn.Linear(gate_hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / k))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def shared_parameters(self):
        return list(self.conv1.parameters()) + \
            list(self.block1.parameters()) + \
            list(self.block2.parameters())

    def vehicle_parameters(self):
        return list(self.block3_vehicle.parameters()) + \
            list(self.bn_vehicle.parameters()) + \
            list(self.fc_vehicle.parameters())

    def animal_parameters(self):
        return list(self.block3_animal.parameters()) + \
            list(self.bn_animal.parameters()) + \
            list(self.fc_animal.parameters())

    def gate_parameters(self):
        return list(self.gate_fc1.parameters()) + list(self.gate_fc2.parameters())

    def _gate(self, shared_feat):
        pooled = F.adaptive_avg_pool2d(shared_feat, 1).view(shared_feat.size(0), -1)
        gate = torch.sigmoid(self.gate_fc2(F.relu(self.gate_fc1(pooled))))
        return gate

    def _head_forward(self, shared_feat, block3, bn, fc):
        out = block3(shared_feat)
        out = self.relu(bn(out))
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        return fc(out)

    def forward(self, x, return_aux=False, head=None):
        shared = self.conv1(x)
        shared = self.block1(shared)
        shared = self.block2(shared)

        gate = self._gate(shared)  # (N, 1)

        logits_vehicle = self._head_forward(shared, self.block3_vehicle,
                                            self.bn_vehicle, self.fc_vehicle)
        logits_animal = self._head_forward(shared, self.block3_animal,
                                           self.bn_animal, self.fc_animal)

        if head == 'vehicle':
            return logits_vehicle
        if head == 'animal':
            return logits_animal

        # Dynamic fusion: place logits in CIFAR-10 order
        gate_v = gate.expand_as(logits_vehicle)
        gate_a = (1.0 - gate).expand_as(logits_animal)
        out = logits_vehicle.new_zeros((logits_vehicle.size(0), 10))
        out[:, self.vehicle_classes] = logits_vehicle * gate_v
        out[:, self.animal_classes] = logits_animal * gate_a

        if return_aux:
            return logits_vehicle, logits_animal, out
        return out
