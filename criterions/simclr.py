import torch
import torch.nn.functional as F
from torch import nn


class SimCLRLoss(nn.Module):
    def __init__(self, temperature, metric):
        super().__init__()
        self.temperature = temperature
        self.metric = metric
        assert metric in ["cosine", "cauchy"], "Not implemented metric {}".format(self.metric)

    def forward(self, features, idx):
        if self.metric == "cosine":
            return self.ntxent_loss(hidden=features, hidden_norm=True, temperature=self.temperature)
        elif self.metric == "cauchy":
            return self.cauchy_loss(hidden=features)
        else:
            raise NotImplementedError("Not implemented metric {}".format(self.metric))

    def ntxent_loss(self,
            hidden,
            hidden_norm=True,
            temperature=1.0,
            distributed=False
    ):
        """Compute loss for model.
        From https://github.com/google-research/simclr/blob/master/objective.py
        Args:
          hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
          hidden_norm: whether or not to use normalization on the hidden vector.
          temperature: a `floating` number for temperature scaling.

        Returns:
          A loss scalar.
        """
        # Get (normalized) hidden1 and hidden2.

        LARGE_NUM = 1e9

        if hidden_norm:
            hidden = F.normalize(hidden, dim=1, p=2)
        B = hidden.shape[0] // 2
        hidden1 = hidden[:B]
        hidden2 = hidden[B:]

        # Gather hidden1/hidden2 across replicas and create local labels.
        if distributed:
            # TODO
            raise NotImplementedError("Distributed loss is not yet implemented.")
        else:
            hidden1_large = hidden1
            hidden2_large = hidden2
            # labels = F.one_hot(torch.arange(B, device=hidden1_large.device), B * 2,)
            labels = torch.arange(B, device=hidden1_large.device)
            masks = F.one_hot(torch.arange(B, device=hidden1_large.device), B)

        logits_aa = torch.matmul(hidden1, hidden1_large.T) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.T) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.T) / temperature
        logits_ba = torch.matmul(hidden2, hidden1_large.T) / temperature

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = (loss_a + loss_b) / 2.0

        return loss

    def cauchy_loss(self,
                    hidden,
                    distributed=False
                    ):
        """Compute loss for model.
        From https://github.com/google-research/simclr/blob/master/objective.py
        Args:
          hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).

        Returns:
          A loss scalar.
        """
        # Get (normalized) hidden1 and hidden2.

        LARGE_NUM = 1e9

        B = hidden.shape[0] // 2
        hidden1 = hidden[:B]
        hidden2 = hidden[B:]

        # Gather hidden1/hidden2 across replicas and create local labels.
        if distributed:
            # TODO
            raise NotImplementedError("Distributed loss is not yet implemented.")
        else:
            hidden1_large = hidden1
            hidden2_large = hidden2
            labels = F.one_hot(torch.arange(B, device=hidden1_large.device), B * 2)
            # labels = torch.arange(B, device=hidden1_large.device)
            masks = F.one_hot(torch.arange(B, device=hidden1_large.device), B)

        logits_aa = 1.0 / ( 1.0 + torch.cdist(hidden1, hidden1_large).pow(2) )
        logits_aa = logits_aa.masked_fill(masks.bool(), 0.0)
        logits_bb = 1.0 / ( 1.0 + torch.cdist(hidden2, hidden2_large).pow(2) )
        logits_bb = logits_bb.masked_fill(masks.bool(), 0.0)
        logits_ab = 1.0 / ( 1.0 + torch.cdist(hidden1, hidden2_large).pow(2) )
        logits_ba = 1.0 / ( 1.0 + torch.cdist(hidden2, hidden1_large).pow(2) )

        logits_a = torch.cat([logits_ab, logits_aa], dim=1)
        logits_b = torch.cat([logits_ba, logits_bb], dim=1)

        loss_a = - torch.log( logits_a[labels.bool()] / torch.sum(logits_a, dim=1) )
        loss_b = - torch.log( logits_b[labels.bool()] / torch.sum(logits_b, dim=1) )
        loss = ( loss_a.mean() + loss_b.mean() ) / 2.0
        return loss
