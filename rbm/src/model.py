"""Restricted Boltzmann Machine for collaborative filtering."""
import torch
import torch.nn as nn
from typing import Tuple


class RBM(nn.Module):
    """RBM with Gibbs sampling for recommendation inference."""

    def __init__(self, n_visible, n_hidden):
        """Initialize RBM with Xavier-initialized weights."""
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = nn.Parameter(torch.empty(n_hidden, n_visible))
        nn.init.xavier_uniform_(self.W)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden units given visible layer activation."""
        p_h = torch.sigmoid(torch.mm(v, self.W.t()) + self.h_bias)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct visible layer from hidden activation."""
        p_v = torch.sigmoid(torch.mm(h, self.W) + self.v_bias)
        return p_v, torch.bernoulli(p_v)
