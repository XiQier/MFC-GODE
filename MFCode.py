import torch
from torch import nn
from torchdiffeq import odeint

class FrequencyDomainAlpha(nn.Module):
    def __init__(self, k,data_name):
        super().__init__()
        self.k = k  # Top-k 高频率
        # 两层 MLP 处理 freq_feat1 和 freq_feat2
        if data_name == 'Beauty':
            self.freq_net_1 = nn.Sequential(
                nn.Linear(k * 2, k),
                nn.BatchNorm1d(k),  # 加入 BatchNorm
                nn.Linear(k, 1),
            )
            self.freq_net_2 = nn.Sequential(
                nn.Linear(k, k // 2),
                nn.BatchNorm1d(k // 2),  # 加入 BatchNorm
                nn.Linear(k // 2, 1),
            )
        elif data_name == 'Health_and_Personal_Care':
            self.freq_net_1 = nn.Sequential(
                nn.Linear(k * 2, k),
                nn.Linear(k, 1),
            )
            self.freq_net_2 = nn.Sequential(
                nn.Linear(k, k // 2),
                nn.Linear(k // 2, 1),
            )
        elif data_name == 'Office_Products':
            self.freq_net_1 = nn.Sequential(
                nn.Linear(k * 2, k),
                nn.LayerNorm(k),
                nn.GELU(),
                nn.Linear(k, 1),
            )
            self.freq_net_2 = nn.Sequential(
                nn.Linear(k, k // 2),
                nn.LayerNorm(k // 2),
                nn.GELU(),
                nn.Linear(k // 2, 1),
            )

    def topk_freq_features(self, fft_tensor, k):
        """
        fft_tensor: complex tensor [N, D]
        return: real tensor [N, 2k] （拼接实部+虚部）
        """
        # Compute amplitude spectrum for top-k selection
        amp = torch.abs(fft_tensor)  # [N, D]
        topk_vals, topk_idx = torch.topk(amp, k, dim=1)

        batch_idx = torch.arange(fft_tensor.size(0), device=fft_tensor.device).unsqueeze(1)
        topk_freqs = fft_tensor[batch_idx, topk_idx]  # complex: [N, k]

        # Separate real and imag parts
        topk_feat = torch.cat([topk_freqs.real, topk_freqs.imag], dim=-1)  # [N, 2k]
        return torch.tanh(topk_feat)
    def topk_low_freq(self, fft_tensor, k=None):
        """
        取低频前 K 个频率（按索引，从0开始）
        fft_tensor: complex tensor [N, D]
        return: real tensor [N, 2k] (实部+虚部)
        """
        lowk_freqs = fft_tensor[:, :k]  # 前 k 个频率
        topk_feat = torch.cat([lowk_freqs.real, lowk_freqs.imag], dim=-1)
        return torch.tanh(topk_feat)

    def topk_high_freq(self, fft_tensor, k=None):
        """
        取高频前 K 个频率（按索引，从末尾开始）
        fft_tensor: complex tensor [N, D]
        return: real tensor [N, 2k] (实部+虚部)
        """
        highk_freqs = fft_tensor[:, -k:]  # 后 k 个频率
        topk_feat = torch.cat([highk_freqs.real, highk_freqs.imag], dim=-1)
        return torch.tanh(topk_feat)
    def forward(self, res1, res2):
        # FFT: [N, D] -> [N, D] (complex)
        fft1 = torch.fft.fft(res1, dim=1)
        fft2 = torch.fft.fft(res2, dim=1)

        # Get top-k frequency components
        freq_feat1 = self.topk_freq_features(fft1, self.k)  # [N, 2k]
        freq_feat2 = self.topk_freq_features(fft2, self.k // 2)
        alph_1=torch.sigmoid(self.freq_net_1(freq_feat1))
        alph_2=torch.sigmoid(self.freq_net_2(freq_feat2))
        return alph_1, alph_2


class ODEFunc(nn.Module):

    def __init__(self, adj, data_name,top_k):
        super(ODEFunc, self).__init__()
        self.g = adj
        self.x0 = None
        self.freq_alpha_module = FrequencyDomainAlpha(top_k,data_name)
        self.name = data_name

    def forward(self, t, x):
        x0 = x
        x1 = torch.spmm(self.g, x)
        x2 = torch.spmm(self.g, x1)
        res1 = x1 - x0
        res2 = x2 - x1

        self.alph_1, self.alph_2 = self.freq_alpha_module(res1, res2)
        ax = self.alph_1 * torch.spmm(self.g, x)
        a2x = self.alph_2 * torch.spmm(self.g, ax)
        f = a2x - x
        return f
class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0, 1]), solver='euler'):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc
        self.solver = solver

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method=self.solver)[-1]
        return z