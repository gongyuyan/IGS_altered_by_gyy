"""
Extended Configuration for IGS with Granger Causality
Adds causal-related hyperparameters to IGS framework

Use this in place of or alongside the original configurations.py
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass 
class IGSConfig:
    """
    Arguments that IGS specifically have for training 
    """
    
    # Original IGS parameters
    mask_function: Optional[str] = field(
        default="Sigmoid",
        metadata={"help": "Probability function acting on the Mask"},
    )
    
    xavier_unif_init: Optional[bool] = field(
        default=False,
        metadata={"help": "using xavier uniform to init mask"},
    )
    
    load_from_previous: Optional[bool] = field(
        default=False, 
        metadata={"help": "load from previous iteration to init mask"}
    )
    
    load_from_Saliency: Optional[bool] = field(
        default=True,
        metadata={"help": "load with gradients"}
    )
    
    generate_individual_mask_all: Optional[bool] = field(
        default=True, 
        metadata={"help": "individual gradient mask for both class"} 
    )
    
    metaMask_Sum: Optional[bool] = field(
        default=True,
        metadata={"help": "take sum of of individual mask"} 
    )
    
    absolute_sum_unified_mask: Optional[bool] = field(
        default=True,
        metadata={"help": "take absolute sum of unified mask"}
    )
    
    add_indicator_matrix: Optional[bool] = field(
        default=True,
        metadata={"help": "avoid pruning same entires over iterations"}
    )
    
    sigmoid_after_mask: Optional[bool] = field(
        default=True,
        metadata={"help": "loss regularization using sigmoid"}
    )
    
    l1_after_mask: Optional[bool] = field(
        default=False,
        metadata={"help": "loss regularization using l1"}
    )
    
    use_original_edge_mask: Optional[bool] = field(
        default=False,
        metadata={"help": "original edge mask"}
    )
    
    use_symmetric_edge_mask: Optional[bool] = field(
        default=True,
        metadata={"help": "use symmetrized edge mask"}
    )
    
    save_model: Optional[bool] = field(
        default=True,
        metadata={"help": "whether saving model"}
    )
    
    # ============== NEW: Granger Causality Parameters ==============
    
    use_causal_effect_regularization: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Enable Granger causality-based regularization during edge mask learning. "
                   "This separates edges into causal and confounding factors."
        }
    )
    
    causal_lambda: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "Weight for causal effect regularization. "
                   "Controls the strength of the regularization term that encourages "
                   "preservation of causally-relevant edges."
        }
    )
    
    causal_factor_split_ratio: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Ratio (0.0-1.0) to split edge mask into causal vs confounding factors. "
                   "Higher saliency edges are considered causal. "
                   "E.g., 0.5 means top 50% by saliency are causal, bottom 50% are confounding."
        }
    )
    
    use_conditional_mi: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Use Conditional Mutual Information (CMI) to compute causal relationships. "
                   "CMI measures I(alpha; Y | beta), identifying edges with causal effects "
                   "after removing confounding influences."
        }
    )
    
    disentangle_latent_factors: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Learn disentangled latent representations (alpha, beta) similar to CI-GNN. "
                   "Alpha represents causally-relevant factors, beta represents confounders."
        }
    )
    
    causal_weighting_ratio: Optional[float] = field(
        default=0.7,
        metadata={
            "help": "Weight ratio for causal vs confounding components in saliency weighting. "
                   "Range: [0.0, 1.0]. Higher values emphasize causal edges more. "
                   "Default 0.7 means 70% weight on causal, 30% on confounding."
        }
    )
    
    enable_cmi_logging: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Enable logging of Conditional Mutual Information values during training. "
                   "Useful for monitoring causal factor learning."
        }
    )
    
    cmi_threshold: Optional[float] = field(
        default=0.01,
        metadata={
            "help": "Threshold for considering an edge as causally significant. "
                   "Edges with CMI below this threshold may be pruned more aggressively."
        }
    )
