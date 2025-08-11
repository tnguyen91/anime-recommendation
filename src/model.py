import torch
import torch.nn as nn
from typing import Tuple

class RBM(nn.Module):
    """
    Restricted Boltzmann Machine for collaborative filtering.
    
    A binary RBM with visible units representing user-item interactions
    and hidden units learning latent factors for recommendation.
    
    Args:
        n_visible (int): Number of visible units (items/anime)
        n_hidden (int): Number of hidden units (latent factors)
    """
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Weight matrix and biases
        self.W = nn.Parameter(torch.empty(n_hidden, n_visible))
        nn.init.xavier_uniform_(self.W)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_h(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden layer activations given visible layer input.
        
        Backward pass from visible to hidden layer using sigmoid activation.
        Used in the positive and negative phases of contrastive divergence.
        
        Args:
            v (torch.Tensor): Visible unit activations (batch_size, n_visible)
            
        Returns:
            tuple: (probabilities, binary_samples) for hidden units
        """
        p_h = torch.sigmoid(torch.mm(v, self.W.t()) + self.h_bias)
        return p_h, torch.bernoulli(p_h)

    def sample_v(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible units given hidden units.
        
        Forward pass from hidden to visible layer using sigmoid activation.
        Used in the negative phase of contrastive divergence training.
        
        Args:
            h (torch.Tensor): Hidden unit activations (batch_size, n_hidden)
            
        Returns:
            tuple: (probabilities, binary_samples) for visible units
        """
        p_v = torch.sigmoid(torch.mm(h, self.W) + self.v_bias)
        return p_v, torch.bernoulli(p_v)
