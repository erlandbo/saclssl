import torch
from torch import nn
from torch.nn import functional as F
from criterions.sacl import SACLBase


class SACLBatchMix(nn.Module):
    def __init__(self, metric, N, rho, alpha, s_init, single_s, temp):
        super(SACLBatchMix, self).__init__()
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


class CosineLoss(SACLBase):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super().__init__(N, rho, alpha, s_init, single_s, temp)
        self.mu = 0.5

    def forward(self, feats, feats_idx):
        feats = F.normalize(feats, dim=1, p=2)
        B = feats.shape[0] // 2
        sim_matrix = torch.mm(feats, feats.T)
        self_mask = torch.eye(2 * B, device=feats.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(self_mask, float("-inf"))
        q = torch.exp(sim_matrix / self.temp)  # (2B,E),(E,2B) -> (2B,2B)
        # Attraction
        pos_mask = torch.roll(torch.eye(2 * B, device=feats.device, dtype=torch.bool), shifts=B, dims=1)
        q_attr = q[pos_mask]  # (2B,)
        q_attr_1 = q_attr[:B]  # (B,)
        q_attr_2 = q_attr[B:]  # (B,)
        attractive_forces_1 = - torch.log(q_attr_1)
        attractive_forces_2 = - torch.log(q_attr_2)
        # Repulsion
        q_rep = q.masked_fill(pos_mask, 0.0)  # (2B,2B)
        q_rep_1 = q_rep[:B]  # (B,2B)
        q_rep_2 = q_rep[B:]  # (B,2B)

        if self.single_s:
            feats_idx = 0

        with torch.no_grad():
            Zi_1 = torch.sum(q_rep_1.detach(), dim=1)  # (B,2B) -> (B,)
            Zi_2 = torch.sum(q_rep_2.detach(), dim=1)  # (B,2B) -> (B,)

            Z_hat = ( self.s_inv[feats_idx] / self.N.pow(2) ) * ( 2 * B - 2)  # (B,)

            Z_1 = self.mu * Z_hat + (1.0 - self.mu) * Zi_1
            Z_2 = self.mu * Z_hat + (1.0 - self.mu) * Zi_2

        repulsive_forces_1 = torch.sum(q_rep_1 / Z_1.detach().view(-1, 1), dim=1)
        repulsive_forces_2 = torch.sum(q_rep_2 / Z_2.detach().view(-1, 1), dim=1)

        loss_1 = attractive_forces_1.mean() + repulsive_forces_1.mean()
        loss_2 = attractive_forces_2.mean() + repulsive_forces_2.mean()
        loss = (loss_1 + loss_2) / 2.0

        self.update_s(q_attr_1, q_attr_2, q_rep_1, q_rep_2, feats_idx)

        #print(loss)

        return loss


class CauchyLoss(SACLBase):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super().__init__(N, rho, alpha, s_init, single_s, temp)
        self.mu = 0.5

    def forward(self, feats, feats_idx):
        B = feats.shape[0] // 2
        q = 1.0 / ( 1.0 + torch.cdist(feats, feats).pow(2) )   # (2B,E),(E,2B) -> (2B,2B)
        self_mask = torch.eye(2 * B, device=feats.device, dtype=torch.bool)
        q = q.masked_fill(self_mask, 0.0)
        # Attraction
        pos_mask = torch.roll(torch.eye(2 * B, device=feats.device, dtype=torch.bool), shifts=B, dims=1)
        q_attr = q[pos_mask]  # (2B,)
        q_attr_1 = q_attr[:B]  # (B,)
        q_attr_2 = q_attr[B:]  # (B,)
        attractive_forces_1 = - torch.log(q_attr_1)
        attractive_forces_2 = - torch.log(q_attr_2)
        # Repulsion
        q_rep = q.masked_fill(pos_mask, 0.0)  # (2B,2B)
        q_rep_1 = q_rep[:B]  # (B,2B)
        q_rep_2 = q_rep[B:]  # (B,2B)

        if self.single_s:
            feats_idx = 0

        with torch.no_grad():
            Zi_1 = torch.sum(q_rep_1.detach(), dim=1)  # (B,2B) -> (B,)
            Zi_2 = torch.sum(q_rep_2.detach(), dim=1)  # (B,2B) -> (B,)

            Z_hat = ( self.s_inv[feats_idx] / self.N.pow(2) ) * ( 2 * B - 2)  # (B,)

            Z_1 = self.mu * Z_hat + (1.0 - self.mu) * Zi_1
            Z_2 = self.mu * Z_hat + (1.0 - self.mu) * Zi_2

        repulsive_forces_1 = torch.sum(q_rep_1 / Z_1.detach().view(-1, 1), dim=1)
        repulsive_forces_2 = torch.sum(q_rep_2 / Z_2.detach().view(-1, 1), dim=1)

        loss_1 = attractive_forces_1.mean() + repulsive_forces_1.mean()
        loss_2 = attractive_forces_2.mean() + repulsive_forces_2.mean()
        loss = (loss_1 + loss_2) / 2.0

        self.update_s(q_attr_1, q_attr_2, q_rep_1, q_rep_2,feats_idx)

        return loss


class UnitGaussianLoss(SACLBase):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super().__init__(N, rho, alpha, s_init, single_s, temp)
        self.mu = 0.5

    def forward(self, feats, feats_idx):
        feats = F.normalize(feats, p=2, dim=1)
        B = feats.shape[0] // 2
        neg_pairwise_distances = -1.0 * torch.cdist(feats, feats, p=2).pow(2) / self.temp
        self_mask = torch.eye(2 * B, device=feats.device, dtype=torch.bool)
        neg_pairwise_distances = neg_pairwise_distances.masked_fill(self_mask, float("-inf"))
        q = torch.exp(neg_pairwise_distances)  # (2B,2B)
        # Attraction
        pos_mask = torch.roll(torch.eye(2 * B, device=feats.device, dtype=torch.bool), shifts=B, dims=1)
        q_attr = q[pos_mask]  # (2B,)
        q_attr_1 = q_attr[:B]  # (B,)
        q_attr_2 = q_attr[B:]  # (B,)
        attractive_forces_1 = - torch.log(q_attr_1)
        attractive_forces_2 = - torch.log(q_attr_2)
        # Repulsion
        q_rep = q.masked_fill(pos_mask, 0.0)  # (2B,2B)
        q_rep_1 = q_rep[:B]  # (B,2B)
        q_rep_2 = q_rep[B:]  # (B,2B)

        if self.single_s:
            feats_idx = 0

        with torch.no_grad():
            Zi_1 = torch.sum(q_rep_1.detach(), dim=1)  # (B,2B) -> (B,)
            Zi_2 = torch.sum(q_rep_2.detach(), dim=1)  # (B,2B) -> (B,)

            Z_hat = ( self.s_inv[feats_idx] / self.N.pow(2) ) * ( 2 * B - 2)  # (B,)

            Z_1 = self.mu * Z_hat + (1.0 - self.mu) * Zi_1
            Z_2 = self.mu * Z_hat + (1.0 - self.mu) * Zi_2

        repulsive_forces_1 = torch.sum(q_rep_1 / Z_1.detach().view(-1, 1), dim=1)
        repulsive_forces_2 = torch.sum(q_rep_2 / Z_2.detach().view(-1, 1), dim=1)

        loss_1 = attractive_forces_1.mean() + repulsive_forces_1.mean()
        loss_2 = attractive_forces_2.mean() + repulsive_forces_2.mean()
        loss = (loss_1 + loss_2) / 2.0

        self.update_s(q_attr_1, q_attr_2, q_rep_1, q_rep_2,feats_idx)

        return loss


class GaussianLoss(SACLBase):
    def __init__(self, N, rho, alpha, s_init, single_s, temp):
        super().__init__(N, rho, alpha, s_init, single_s, temp)
        self.mu = 0.5

    def forward(self, feats, feats_idx):
        # feats = F.normalize(feats, p=2, dim=1)
        B = feats.shape[0] // 2
        neg_pairwise_distances = -1.0 * torch.cdist(feats, feats, p=2).pow(2) / (2.0 * self.temp**2.0)
        self_mask = torch.eye(2 * B, device=feats.device, dtype=torch.bool)
        neg_pairwise_distances = neg_pairwise_distances.masked_fill(self_mask, float("-inf"))
        q = torch.exp(neg_pairwise_distances / self.temp)  # (2B,2B)
        # Attraction
        pos_mask = torch.roll(torch.eye(2 * B, device=feats.device, dtype=torch.bool), shifts=B, dims=1)
        q_attr = q[pos_mask]  # (2B,)
        q_attr_1 = q_attr[:B]  # (B,)
        q_attr_2 = q_attr[B:]  # (B,)
        attractive_forces_1 = - torch.log(q_attr_1)
        attractive_forces_2 = - torch.log(q_attr_2)
        # Repulsion
        q_rep = q.masked_fill(pos_mask, 0.0)  # (2B,2B)
        q_rep_1 = q_rep[:B]  # (B,2B)
        q_rep_2 = q_rep[B:]  # (B,2B)

        if self.single_s:
            feats_idx = 0

        with torch.no_grad():
            Zi_1 = torch.sum(q_rep_1.detach(), dim=1)  # (B,2B) -> (B,)
            Zi_2 = torch.sum(q_rep_2.detach(), dim=1)  # (B,2B) -> (B,)

            Z_hat = ( self.s_inv[feats_idx] / self.N.pow(2) ) * ( 2 * B - 2)  # (B,)

            Z_1 = self.mu * Z_hat + (1.0 - self.mu) * Zi_1
            Z_2 = self.mu * Z_hat + (1.0 - self.mu) * Zi_2

        repulsive_forces_1 = torch.sum(q_rep_1 / Z_1.detach().view(-1, 1), dim=1)
        repulsive_forces_2 = torch.sum(q_rep_2 / Z_2.detach().view(-1, 1), dim=1)

        loss_1 = attractive_forces_1.mean() + repulsive_forces_1.mean()
        loss_2 = attractive_forces_2.mean() + repulsive_forces_2.mean()
        loss = (loss_1 + loss_2) / 2.0

        self.update_s(q_attr_1, q_attr_2, q_rep_1, q_rep_2,feats_idx)

        return loss


# class CosineLossOld(SACLBatchMixBase):
#     def __init__(self, N, rho, alpha, s_init, Ns, temp):
#         super().__init__(N, rho, alpha, s_init, Ns, temp)
#
#     def forward(self, feats, feats_idx):
#         feats = F.normalize(feats, dim=1, p=2)
#         B = feats.shape[0] // 2
#         sim_matrix = torch.mm(feats, feats.T)
#         self_mask = torch.eye(2 * B, device=feats.device, dtype=torch.bool)
#         sim_matrix = sim_matrix.masked_fill(self_mask, float("-inf"))
#         q = torch.exp(sim_matrix / self.temp)  # (2B,E),(E,2B) -> (2B,2B)
#         # Attraction
#         pos_mask = torch.roll(torch.eye(2 * B, device=feats.device, dtype=torch.bool), shifts=B, dims=1)
#         q_attr = q[pos_mask]  # (2B,)
#         q_attr_1 = q_attr[:B]  # (B,)
#         q_attr_2 = q_attr[B:]  # (B,)
#         attractive_forces_1 = - torch.log(q_attr_1)
#         attractive_forces_2 = - torch.log(q_attr_2)
#         # Repulsion
#         q_rep = q.masked_fill(pos_mask, 0.0)  # (2B,2B)
#         q_rep_1 = q_rep[:B]  # (B,2B)
#         q_rep_2 = q_rep[B:]  # (B,2B)
#         Zi_1 = torch.sum(q_rep_1, dim=1)  # (B,2B) -> (B,)
#         Zi_2 = torch.sum(q_rep_2, dim=1)  # (B,2B) -> (B,)
#
#         if self.Ns:
#             Z_hat = ( self.s_inv[feats_idx] / self.N.pow(1) ) * ( 2 * B - 2)  # (B,)
#         else:
#             Z_hat = ( self.s_inv / self.N.pow(1) ) * ( 2 * B - 2)  # (B,)
#
#         grad_term = 1.0 / ( 1.0 - self.rho )
#         repulsive_forces_1 = torch.log( (1.0 - self.rho) * Zi_1 + self.rho * Z_hat ) * grad_term
#         repulsive_forces_2 = torch.log( (1.0 - self.rho) * Zi_2 + self.rho * Z_hat ) * grad_term
#
#         loss_1 = attractive_forces_1.mean() + repulsive_forces_1.mean()
#         loss_2 = attractive_forces_2.mean() + repulsive_forces_2.mean()
#         loss = (loss_1 + loss_2) / 2.0
#
#         self.update_s(q_attr_1, q_attr_2, q_rep_1, q_rep_2,feats_idx)
#
#         return loss


# class SACLBatchMixBase(nn.Module):
#     def __init__(self, N, rho, alpha, s_init, singel_s, temp):
#         super(SACLBatchMixBase, self).__init__()
#         self.register_buffer("s_inv", torch.zeros(1 if singel_s else N) + N**s_init)
#         self.register_buffer("N", torch.zeros(1, ) + N)
#         self.register_buffer("rho", torch.zeros(1, ) + rho)
#         self.register_buffer("alpha", torch.zeros(1, ) + alpha)
#         self.singel_s = singel_s
#         self.temp = temp
#
#     @torch.no_grad()
#     def update_s(self, q_attr_1, q_attr_2, q_rep_1, q_rep_2, feats_idx):
#         B = feats_idx.size(0)
#         # redundant but for consistency
#         q_attr_1 = q_attr_1.detach()  # (B,)
#         q_attr_2 = q_attr_2.detach()  # (B,)
#         q_rep_1 = q_rep_1.detach()  # (B,2B)
#         q_rep_2 = q_rep_2.detach()  # (B,2B)
#         # Attraction
#         E_attr_1 = q_attr_1  # (B,)
#         E_attr_2 = q_attr_2  # (B,)
#         # Repulsion
#         if self.singel_s:
#             assert feats_idx == 0, "featsidx not zero"
#             E_rep_1 = torch.sum(q_rep_1, dim=(0, 1)) / (B * (2.0 * B - 2.0))  # (1,)
#             E_rep_2 = torch.sum(q_rep_2, dim=(0, 1)) / (B * (2.0 * B - 2.0))  # (1,)
#         else:
#             E_rep_1 = torch.sum(q_rep_1, dim=1) / (2.0 * B - 2.0)  # (B,)
#             E_rep_2 = torch.sum(q_rep_2, dim=1) / (2.0 * B - 2.0)  # (B,)
#
#         xi_1 = self.alpha * E_attr_1 + (1.0 - self.alpha) * E_rep_1  # (B,) + (B,)
#         xi_2 = self.alpha * E_attr_2 + (1.0 - self.alpha) * E_rep_2  # (B,) + (B,)
#
#         s_inv_1 = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(1) * xi_1  # (B,) + (B,)
#         s_inv_2 = self.rho * self.s_inv[feats_idx] + (1.0 - self.rho) * self.N.pow(1) * xi_2  # (B,) + (B,)
#
#         self.s_inv[feats_idx] = (s_inv_1 + s_inv_2) / 2.0
