"""
Restricted Boltzmann Machine (RBM) implementation for collaborative filtering.

This module implements an RBM neural network used for generating anime
recommendations. The model learns latent features from user-anime interaction
patterns and uses them to predict which anime a user might enjoy.

Architecture:
    - Visible layer: Binary units representing anime (liked/not liked)
    - Hidden layer: Latent features learned during training
    - Weights: Learned connections between visible and hidden layers

Reference:
    Salakhutdinov, R., Mnih, A., & Hinton, G. (2007).
    Restricted Boltzmann machines for collaborative filtering.
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn


class RBM(nn.Module):
    """
    Restricted Boltzmann Machine for recommendation systems.

    Uses contrastive divergence for training and Gibbs sampling for inference.
    The visible units represent anime items, and hidden units capture latent
    user preferences.

    Attributes:
        n_visible: Number of visible units (anime count)
        n_hidden: Number of hidden units (latent features)
        W: Weight matrix connecting visible and hidden layers
        v_bias: Bias for visible units
        h_bias: Bias for hidden units
    """

    def __init__(self, n_visible: int, n_hidden: int) -> None:
        """Initialize RBM with given dimensions."""
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = nn.Parameter(torch.empty(n_hidden, n_visible))
        nn.init.xavier_uniform_(self.W)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units given visible units.

        Args:
            v: Visible layer activation (batch_size, n_visible)

        Returns:
            Tuple of (probability, binary sample) for hidden units
        """
        p_h = torch.sigmoid(torch.mm(v, self.W.t()) + self.h_bias)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible units given hidden units (reconstruction).

        Args:
            h: Hidden layer activation (batch_size, n_hidden)

        Returns:
            Tuple of (probability, binary sample) for visible units
        """
        p_v = torch.sigmoid(torch.mm(h, self.W) + self.v_bias)
        return p_v, torch.bernoulli(p_v)
