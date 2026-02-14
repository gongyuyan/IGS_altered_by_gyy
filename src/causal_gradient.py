"""
Causal Gradient-based Edge Pruning Module
Integrates Granger Causality concepts into IGS framework
Based on CI-GNN: A Granger Causality-Inspired Graph Neural Network

Reference: https://github.com/ZKZ-Brain/CI-GNN
Author: Adapted for IGS integration
"""
import torch
import numpy as np
from typing import Tuple, Optional
from scipy.spatial.distance import pdist, squareform

# Constants for numerical stability
EPS = 1e-8


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """
    Calculate pairwise Euclidean distances
    
    Args:
        x: Input tensor, should be 2D [batch_size, feature_dim]
    
    Returns:
        Pairwise distance matrix [batch_size, batch_size]
    """
    if x.dim() == 1:
        x = x.unsqueeze(1)
    
    instances_norm = torch.sum(x**2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def calculate_sigma(Z_numpy: torch.Tensor) -> float:
    """
    Calculate bandwidth parameter (sigma) using k-nearest neighbors heuristic
    Based on CI-GNN implementation
    
    Args:
        Z_numpy: Input tensor [batch_size, feature_dim]
    
    Returns:
        Bandwidth parameter (scalar)
    """
    if Z_numpy.dim() == 1:
        Z_numpy = Z_numpy.unsqueeze(1)
    
    Z_numpy = Z_numpy.cpu().detach().numpy()
    
    # Calculate Euclidean distance between all samples
    k = squareform(pdist(Z_numpy, 'euclidean'))
    
    # Use mean of k-nearest neighbor distances (k=10)
    sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))
    
    if sigma < 0.1:
        sigma = 0.1
    
    return sigma


def calculate_gram_mat(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Calculate Gram matrix using RBF (Radial Basis Function) kernel
    
    Args:
        x: Input tensor [batch_size, feature_dim]
        sigma: Bandwidth parameter
    
    Returns:
        Gram matrix [batch_size, batch_size]
    """
    dist = pairwise_distances(x)
    return torch.exp(-dist / sigma)


def reyi_entropy(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Calculate Rényi entropy using kernel Gram matrix
    Order alpha = 1.01
    
    Based on CI-GNN's entropy calculation
    
    Args:
        x: Input tensor [batch_size, feature_dim]
        sigma: Bandwidth parameter
    
    Returns:
        Rényi entropy (scalar)
    """
    alpha = 1.01
    
    # Calculate Gram matrix
    k = calculate_gram_mat(x, sigma)
    
    # Normalize by trace
    k = k / (torch.trace(k) + EPS)
    
    # Eigendecomposition
    eigv = torch.abs(torch.linalg.eigvalsh(k))
    
    # Rényi entropy
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow) + EPS)
    
    return entropy


def joint_entropy(x: torch.Tensor, y: torch.Tensor, 
                  s_x: float, s_y: float) -> torch.Tensor:
    """
    Calculate joint entropy using product of Gram matrices
    
    Args:
        x: First variable [batch_size, feature_dim_x]
        y: Second variable [batch_size, feature_dim_y]
        s_x: Bandwidth for x
        s_y: Bandwidth for y
    
    Returns:
        Joint entropy (scalar)
    """
    alpha = 1.01
    
    # Calculate Gram matrices
    kx = calculate_gram_mat(x, s_x)
    ky = calculate_gram_mat(y, s_y)
    
    # Element-wise product
    k = torch.mul(kx, ky)
    
    # Normalize by trace
    k = k / (torch.trace(k) + EPS)
    
    # Eigendecomposition
    eigv = torch.abs(torch.linalg.eigvalsh(k))
    
    # Entropy
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow) + EPS)
    
    return entropy


def joint_entropy3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor,
                   s_x: float, s_y: float, s_z: float) -> torch.Tensor:
    """
    Calculate three-way joint entropy
    
    Args:
        x, y, z: Variables [batch_size, feature_dim]
        s_x, s_y, s_z: Bandwidth parameters
    
    Returns:
        Three-way joint entropy (scalar)
    """
    alpha = 1.01
    
    # Calculate Gram matrices
    kx = calculate_gram_mat(x, s_x)
    ky = calculate_gram_mat(y, s_y)
    kz = calculate_gram_mat(z, s_z)
    
    # Element-wise products
    k = torch.mul(kx, ky)
    k = torch.mul(k, kz)
    
    # Normalize by trace
    k = k / (torch.trace(k) + EPS)
    
    # Eigendecomposition
    eigv = torch.abs(torch.linalg.eigvalsh(k))
    
    # Entropy
    eig_pow = eigv ** alpha
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eig_pow) + EPS)
    
    return entropy


def calculate_conditional_MI(x: torch.Tensor, y: torch.Tensor, 
                            z: torch.Tensor) -> torch.Tensor:
    """
    Calculate Conditional Mutual Information I(X; Y | Z)
    Core mechanism from CI-GNN for Granger causality detection
    
    Formula: I(X; Y | Z) = H(Y,Z) + H(X,Z) - H(Z) - H(X,Y,Z)
    
    This measures the dependence between X and Y after removing the effect of Z (confounders)
    
    Args:
        x: Causal factors (alpha) [batch_size, feature_dim]
        y: Target labels or representations [batch_size, num_classes]
        z: Confounding factors (beta) [batch_size, feature_dim]
    
    Returns:
        Conditional mutual information (scalar, non-negative)
    """
    try:
        # Calculate bandwidth parameters
        s_x = calculate_sigma(x) ** 2
        s_y = calculate_sigma(y) ** 2
        s_z = calculate_sigma(z) ** 2
        
        # Calculate entropies
        Hyz = joint_entropy(y, z, s_y, s_z)
        Hxz = joint_entropy(x, z, s_x, s_z)
        Hz = reyi_entropy(z, sigma=s_z)
        Hxyz = joint_entropy3(x, y, z, s_x, s_y, s_z)
        
        # Conditional MI formula
        cmi = Hyz + Hxz - Hz - Hxyz
        
        # Ensure non-negative (handle numerical errors)
        cmi = torch.clamp(cmi, min=EPS)
        
        return cmi
    except Exception as e:
        print(f"Warning: CMI calculation failed: {e}, returning small positive value")
        return torch.tensor(EPS, device=x.device)


def calculate_MI(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mutual Information I(X; Y)
    Measures total dependence between X and Y
    
    Formula: I(X; Y) = H(X) + H(Y) - H(X,Y)
    
    Args:
        x: First variable [batch_size, feature_dim_x]
        y: Second variable [batch_size, feature_dim_y]
    
    Returns:
        Mutual information (scalar, non-negative)
    """
    try:
        # Calculate bandwidth parameters
        s_x = calculate_sigma(x)
        s_y = calculate_sigma(y)
        
        # Calculate entropies
        Hx = reyi_entropy(x, s_x ** 2)
        Hy = reyi_entropy(y, s_y ** 2)
        Hxy = joint_entropy(x, y, s_x ** 2, s_y ** 2)
        
        # Mutual information formula
        mi = Hx + Hy - Hxy
        
        # Ensure non-negative
        mi = torch.clamp(mi, min=EPS)
        
        return mi
    except Exception as e:
        print(f"Warning: MI calculation failed: {e}, returning small positive value")
        return torch.tensor(EPS, device=x.device)


def separate_causal_confounding_factors(
    edge_mask: torch.Tensor,
    saliency_map: torch.Tensor,
    split_ratio: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Separate edge mask into causal (α) and confounding (β) factors
    Similar to CI-GNN's latent factor disentanglement
    
    High saliency edges are considered more causally relevant
    
    Args:
        edge_mask: Learned edge mask [num_nodes, num_nodes]
        saliency_map: Gradient-based saliency map [num_nodes, num_nodes]
        split_ratio: Ratio to split into causal vs confounding (default: 0.5)
    
    Returns:
        Tuple of (causal_mask, confounding_mask) both [num_nodes, num_nodes]
    """
    # Flatten for easy indexing
    flat_saliency = saliency_map.reshape(-1)
    
    # Get top-k elements by absolute saliency
    k = int(flat_saliency.numel() * split_ratio)
    k = max(k, 1)  # At least 1 element
    
    # Top-k elements are causal
    top_k_indices = torch.topk(flat_saliency.abs(), k)[1]
    
    # Create causal mask
    causal_mask = torch.zeros_like(edge_mask, dtype=torch.float32)
    causal_mask.reshape(-1)[top_k_indices] = 1.0
    
    # Confounding is the complement
    confounding_mask = 1.0 - causal_mask
    
    return causal_mask, confounding_mask


def compute_causal_effect_score(
    edge_mask: torch.Tensor,
    causal_mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute causal effect score based on mask overlap
    Higher score indicates more causal edges are retained
    
    Args:
        edge_mask: Current edge mask [num_nodes, num_nodes]
        causal_mask: Causal component mask [num_nodes, num_nodes]
    
    Returns:
        Scalar causal effect score in [0, 1]
    """
    # Overlap between learned mask and causal components
    causal_edges = edge_mask * causal_mask
    causal_strength = causal_edges.sum()
    
    # Normalize
    total = edge_mask.sum()
    if total > 0:
        causal_ratio = causal_strength / total
    else:
        causal_ratio = torch.tensor(0.0, device=edge_mask.device)
    
    return causal_ratio


def causal_effect_regularization(
    edge_mask: torch.Tensor,
    saliency_map: torch.Tensor,
    lambda_param: float = 0.05,
    split_ratio: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Add causal effect regularization to edge mask learning
    Ensures edges satisfy Granger causality criterion
    
    L_causal = λ * (confounding_strength - causal_strength)
    
    This encourages the mask to preserve causally-relevant edges
    while pruning confounding factors
    
    Args:
        edge_mask: Current edge mask being optimized [num_nodes, num_nodes]
        saliency_map: Gradient saliency map [num_nodes, num_nodes]
        lambda_param: Regularization weight (default: 0.05)
        split_ratio: Ratio to split causal vs confounding (default: 0.5)
    
    Returns:
        Tuple of (regularization_loss, causal_score)
    """
    # Separate causal and confounding components
    causal_mask, confounding_mask = separate_causal_confounding_factors(
        edge_mask, saliency_map, split_ratio=split_ratio
    )
    
    # Extract causal and confounding edges
    causal_edges = edge_mask * causal_mask
    confounding_edges = edge_mask * confounding_mask
    
    # Compute strengths
    causal_strength = causal_edges.sum()
    confounding_strength = confounding_edges.sum()
    
    # Regularization: maximize causal signal, minimize confounding
    reg_loss = lambda_param * (confounding_strength - causal_strength)
    
    # Compute causal effect score
    causal_score = compute_causal_effect_score(edge_mask, causal_mask)
    
    return reg_loss, causal_score


def weight_saliency_by_causality(
    saliency_map_0: torch.Tensor,
    saliency_map_1: torch.Tensor,
    causal_ratio: float = 0.7,
    split_ratio: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Weight saliency maps by their causal components
    Emphasizes causally-relevant edges in pruning decisions
    
    Args:
        saliency_map_0: Saliency map for class 0 [num_nodes, num_nodes]
        saliency_map_1: Saliency map for class 1 [num_nodes, num_nodes]
        causal_ratio: Weight for causal component (0.0 to 1.0)
        split_ratio: Ratio to split into causal vs confounding
    
    Returns:
        Tuple of (weighted_saliency_0, weighted_saliency_1)
    """
    # Separate components for both maps
    causal_mask_0, confounding_mask_0 = separate_causal_confounding_factors(
        saliency_map_0, saliency_map_0, split_ratio=split_ratio
    )
    causal_mask_1, confounding_mask_1 = separate_causal_confounding_factors(
        saliency_map_1, saliency_map_1, split_ratio=split_ratio
    )
    
    # Weight by causal ratio
    weighted_0 = (
        causal_ratio * saliency_map_0 * causal_mask_0 + 
        (1 - causal_ratio) * saliency_map_0 * confounding_mask_0
    )
    
    weighted_1 = (
        causal_ratio * saliency_map_1 * causal_mask_1 + 
        (1 - causal_ratio) * saliency_map_1 * confounding_mask_1
    )
    
    return weighted_0, weighted_1
