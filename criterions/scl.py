import torch
from torch import nn
from torch.nn import functional as F


class SCL(nn.Module):
    def __init__(self, metric, N, rho, alpha, s_init, single_s, temp):
        super(SCL, self).__init__()
        if metric == 'cosine':
            self.criterion = CosineLoss(N=N, rho=rho, alpha=alpha, temp=temp, s_init=s_init, single_s=single_s)
        elif metric == 'cauchy':
            self.criterion = CauchyLoss(N=N, rho=rho, alpha=alpha, temp=temp, s_init=s_init, single_s=single_s)
        elif metric == 'unit_gaussian':
            self.criterion = UnitGaussianLoss(N=N, rho=rho, alpha=alpha, temp=temp, s_init=s_init, single_s=single_s)
        elif metric == 'gaussian':
            self.criterion = GaussianLoss(N=N, rho=rho, alpha=alpha, temp=temp, s_init=s_init, single_s=single_s)
        else:
            raise ValueError('Unknown metric', metric)

    def forward(self, feats, feats_idx):
        return self.criterion(feats, feats_idx)


class SCLBase(nn.Module):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super(SCLBase, self).__init__()
        self.register_buffer("s_inv", torch.zeros(1 if single_s else N, ) + 1.0 / s_init)
        self.register_buffer("N", torch.zeros(1, ) + N)
        self.register_buffer("rho", torch.zeros(1, ) + rho)
        self.register_buffer("alpha", torch.zeros(1, ) + alpha)
        self.single_s = single_s
        self.temp = temp

        self.register_buffer("E_attr", torch.zeros(1, ) )
        self.register_buffer("E_rep", torch.zeros(1, )  )

    @torch.no_grad()
    def update_s(self, q_attr_1, q_attr_2, q_rep_1, q_rep_2, feats_idx):
        assert q_attr_1.size() == q_attr_2.size() and q_rep_1.size() == q_rep_2.size() and q_attr_1.size(0) == q_rep_2.size(0), "invalid shape"
        B = q_attr_1.size(0)
        # redundant but for consistency
        q_attr_1 = q_attr_1.detach()  # (B,)
        q_attr_2 = q_attr_2.detach()  # (B,)
        q_rep_1 = q_rep_1.detach()  # (B,)
        q_rep_2 = q_rep_2.detach()  # (B,)
        # Attraction
        E_attr_1 = q_attr_1  # (B,)
        E_attr_2 = q_attr_2  # (B,)
        # Repulsion
        E_rep_1 = q_rep_1  # (B,)
        E_rep_2 = q_rep_2  # (B,)
        if self.single_s:
            assert feats_idx == 0, "Error feats_idx not zero"
            # assert B == E_attr_1.shape, "Error B update_s"
            E_attr_1 = torch.sum(E_attr_1) / B  # (1,)
            E_attr_2 = torch.sum(E_attr_2) / B  # (1,)
            E_rep_1 = torch.sum(E_rep_1) / B  # (1,)
            E_rep_2 = torch.sum(E_rep_2) / B  # (1,)

        xi_div_omega_1 = self.alpha * E_attr_1 + (1.0 - self.alpha) * E_rep_1  # (B,) or (1,)
        xi_div_omega_2 = self.alpha * E_attr_2 + (1.0 - self.alpha) * E_rep_2  # (B,) or (1,)

        s_inv_1 = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_1  # (B,) + (B,)
        s_inv_2 = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(2) * xi_div_omega_2  # (B,) + (B,)

        self.s_inv[feats_idx] = (s_inv_1 + s_inv_2) / 2.0

        # Verbose mode
        self.E_attr = (1.0 - self.N**2 / (self.N**2 + 2*B * 1e5)) * self.E_attr + self.N**2 / (self.N**2 + 2*B * 1e5) * (torch.mean(E_attr_1) + torch.mean(E_attr_2)) / 2.0
        self.E_rep = (1.0 - self.N**2 / (self.N**2 + 2*B * 1e5)) * self.E_rep + self.N**2 / (self.N**2 + 2*B * 1e5) * (torch.mean(E_rep_1) + torch.mean(E_rep_2)) / 2.0


class CosineLoss(SCLBase):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super().__init__(N, rho, alpha, s_init, single_s, temp)

    def forward(self, feats, feats_idx):
        feats = F.normalize(feats, dim=1, p=2)
        B = feats.shape[0] // 2
        feats_1 = feats[:B]
        feats_2 = feats[B:]
        # Attraction
        q_attr_1 = torch.exp( torch.sum(feats_1 * feats_2, dim=1) / self.temp )  # (B,)
        q_attr_2 = torch.exp( torch.sum(feats_2 * feats_1, dim=1) / self.temp )  # (B,)
        attractive_forces_1 = - torch.log(q_attr_1)
        attractive_forces_2 = - torch.log(q_attr_2)
        # Repulsion
        q_rep_1 = torch.exp(torch.sum(feats_1 * torch.roll(feats_2, shifts=-1, dims=0), dim=1) / self.temp)  # (B,)
        q_rep_2 = torch.exp(torch.sum(feats_2 * torch.roll(feats_1, shifts=-1, dims=0), dim=1) / self.temp)  # (B,)

        if self.single_s:
            feats_idx = 0

        with torch.no_grad():
            Z_hat = self.s_inv[feats_idx] / self.N.pow(2)  # (B,)

        repulsive_forces_1 = q_rep_1 / Z_hat
        repulsive_forces_2 = q_rep_2 / Z_hat

        loss_1 = attractive_forces_1.mean() + repulsive_forces_1.mean()
        loss_2 = attractive_forces_2.mean() + repulsive_forces_2.mean()
        loss = (loss_1 + loss_2) / 2.0

        self.update_s(q_attr_1, q_attr_2, q_rep_1, q_rep_2,feats_idx)

        return loss


class CauchyLoss(SCLBase):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super().__init__(N, rho, alpha, s_init, single_s, temp)

    def forward(self, feats, feats_idx):
        B = feats.shape[0] // 2
        feats_1 = feats[:B]
        feats_2 = feats[B:]
        # Attraction
        q_attr_1 = 1.0 / (1.0 + F.pairwise_distance(feats_1, feats_2, p=2).pow(2))  # (B,)
        q_attr_2 = 1.0 / (1.0 + F.pairwise_distance(feats_2, feats_1, p=2).pow(2))  # (B,)
        attractive_forces_1 = - torch.log(q_attr_1)
        attractive_forces_2 = - torch.log(q_attr_2)
        # Repulsion
        q_rep_1 = 1.0 / (1.0 + F.pairwise_distance(feats_1, torch.roll(feats_2, shifts=-1, dims=0), p=2).pow(2))  # (B,)
        q_rep_2 = 1.0 / (1.0 + F.pairwise_distance(feats_2, torch.roll(feats_1, shifts=-1, dims=0), p=2).pow(2))  # (B,)

        if self.single_s:
            feats_idx = 0

        with torch.no_grad():
            Z_hat = self.s_inv[feats_idx] / self.N.pow(2)  # (B,)

        repulsive_forces_1 = q_rep_1 / Z_hat
        repulsive_forces_2 = q_rep_2 / Z_hat

        loss_1 = attractive_forces_1.mean() + repulsive_forces_1.mean()
        loss_2 = attractive_forces_2.mean() + repulsive_forces_2.mean()
        loss = (loss_1 + loss_2) / 2.0

        self.update_s(q_attr_1, q_attr_2, q_rep_1, q_rep_2,feats_idx)

        return loss


class UnitGaussianLoss(SCLBase):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super().__init__(N, rho, alpha, s_init, single_s, temp)

    def forward(self, feats, feats_idx):
        feats = F.normalize(feats, p=2, dim=1)
        B = feats.shape[0] // 2
        feats_1 = feats[:B]
        feats_2 = feats[B:]
        # Attraction
        q_attr_1 = torch.exp( -1.0 * F.pairwise_distance(feats_1, feats_2, p=2).pow(2) / self.temp )  # (B,)
        q_attr_2 = torch.exp( -1.0 * F.pairwise_distance(feats_2, feats_1, p=2).pow(2) / self.temp )  # (B,)
        attractive_forces_1 = - torch.log(q_attr_1)
        attractive_forces_2 = - torch.log(q_attr_2)
        # Repulsion
        q_rep_1 = torch.exp(-1.0 * F.pairwise_distance(feats_1, torch.roll(feats_2, shifts=-1, dims=0), p=2).pow(2) / self.temp)  # (B,)
        q_rep_2 = torch.exp(-1.0 * F.pairwise_distance(feats_2, torch.roll(feats_1, shifts=-1, dims=0), p=2).pow(2) / self.temp)  # (B,)

        if self.single_s:
            feats_idx = 0

        Z_hat = self.s_inv[feats_idx] / self.N.pow(2)  # (B,)

        repulsive_forces_1 = q_rep_1 / Z_hat
        repulsive_forces_2 = q_rep_2 / Z_hat

        loss_1 = attractive_forces_1.mean() + repulsive_forces_1.mean()
        loss_2 = attractive_forces_2.mean() + repulsive_forces_2.mean()
        loss = (loss_1 + loss_2) / 2.0

        self.update_s(q_attr_1, q_attr_2, q_rep_1, q_rep_2,feats_idx)

        return loss


class GaussianLoss(SCLBase):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super().__init__(N, rho, alpha, s_init, single_s, temp)

    def forward(self, feats, feats_idx):
        # feats = F.normalize(feats, p=2, dim=1)
        B = feats.shape[0] // 2
        feats_1 = feats[:B]
        feats_2 = feats[B:]
        # Attraction
        q_attr_1 = torch.exp( -1.0 * F.pairwise_distance(feats_1, feats_2, p=2).pow(2) / (2.0 * self.temp**2.0) )  # (B,)
        q_attr_2 = torch.exp( -1.0 * F.pairwise_distance(feats_2, feats_1, p=2).pow(2) / (2.0 * self.temp**2.0) )  # (B,)
        attractive_forces_1 = - torch.log(q_attr_1)
        attractive_forces_2 = - torch.log(q_attr_2)
        # Repulsion
        q_rep_1 = torch.exp(-1.0 * F.pairwise_distance(feats_1, torch.roll(feats_2, shifts=-1, dims=0), p=2).pow(2) / (2.0 * self.temp**2.0))  # (B,)
        q_rep_2 = torch.exp(-1.0 * F.pairwise_distance(feats_2, torch.roll(feats_1, shifts=-1, dims=0), p=2).pow(2) / (2.0 * self.temp**2.0))  # (B,)

        if self.single_s:
            feats_idx = 0

        Z_hat = self.s_inv[feats_idx] / self.N.pow(2)  # (B,)

        repulsive_forces_1 = q_rep_1 / Z_hat
        repulsive_forces_2 = q_rep_2 / Z_hat

        loss_1 = attractive_forces_1.mean() + repulsive_forces_1.mean()
        loss_2 = attractive_forces_2.mean() + repulsive_forces_2.mean()
        loss = (loss_1 + loss_2) / 2.0

        self.update_s(q_attr_1, q_attr_2, q_rep_1, q_rep_2,feats_idx)

        return loss
