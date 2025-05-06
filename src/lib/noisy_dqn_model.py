import torch
import torch.nn as nn
import typing as tt
from torchrl.modules import NoisyLinear

class NoisyDQN(nn.Module):
    def __init__(self, input_shape: tt.Tuple[int, ...],
                 n_actions: int):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        size = self.conv(torch.zeros(1, *input_shape)).size()[-1]
        self.noisy_layers = [
            NoisyLinear(size, 512),
            NoisyLinear(512, n_actions)
        ]
        self.fc = nn.Sequential(
            self.noisy_layers[0],
            nn.ReLU(),
            self.noisy_layers[1],
        )

    def forward(self, x: torch.ByteTensor):
        xx = x / 255.0
        return self.fc(self.conv(xx))

    def reset_noise(self):
        for n in self.noisy_layers:
            n.reset_noise()

    @torch.no_grad()
    def noisy_layers_sigma_snr(self) -> tt.List[float]:
        return [
            ((layer.weight_mu ** 2).mean().sqrt() /
             (layer.weight_sigma ** 2).mean().sqrt()).item()
            for layer in self.noisy_layers
        ]