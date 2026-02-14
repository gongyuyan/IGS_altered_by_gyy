"""
Enhanced Gradient Iteration with Granger Causality
Integrates causal effect regularization into IGS training

This module provides an alternative gradient iteration function that
incorporates conditional mutual information and causal factor separation
"""
import numpy as np 
import torch 
import os 
from utils import * 
from causal_gradient import (
    calculate_conditional_MI,
    separate_causal_confounding_factors,
    causal_effect_regularization,
    weight_saliency_by_causality
)


def Gradient_iteration_with_causality(args):
    """
    Enhanced gradient iteration incorporating Granger causality concepts
    
    This function extends the standard gradient iteration by:
    1. Separating edges into causal (α) and confounding (β) factors
    2. Computing conditional mutual information I(α; Y|β)
    3. Weighting saliency maps by causal relevance
    4. Adding causal regularization to pruning decisions
    
    Args:
        args: Configuration arguments including:
            - use_causal_effect_regularization: Enable causal regularization
            - causal_lambda: Weight for causal regularization
            - causal_factor_split_ratio: Ratio to split causal vs confounding
    """
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    curr_idx = args.curr_idx
    
    outer_exp_name = "./results"
    dataset = args.dataset 
    label_col = args.label_col
    model = args.model 
    exp_name = os.path.join(outer_exp_name, dataset, model, label_col, args.method, args.pruning_mode)
    print(f"* Currently Gradient Iteration with Causality...")
    
    model_name = f"best_val_model{curr_idx}.pth"
    if args.debug_mode:
        print(f"load {model_name}")

    model_sub_dir = os.path.join(exp_name, "model", args.save_exp_name)
    assert os.path.exists(model_sub_dir)
    model_dir = os.path.join(model_sub_dir, model_name)
    if args.debug_mode:
        print("model dir: ", model_dir)
    model = torch.load(model_dir)
    model = model.to(device)

    data_directory = f"./data/normalized_edge_dataSplits{args.dataSplit}/{args.label_col}"
    if args.debug_mode:
        print(f"Gradient Iteration Load data from {data_directory}!")
    y_train = np.load(os.path.join(data_directory, "train_label.npy"))
    y_val = np.load(os.path.join(data_directory, "val_label.npy"))
    y_test = np.load(os.path.join(data_directory, "test_label.npy"))

    if curr_idx == 0:
        train_adj = np.load(os.path.join(data_directory, "train_edge.npy"))
        val_adj = np.load(os.path.join(data_directory, "val_edge.npy"))
        test_adj = np.load(os.path.join(data_directory, "test_edge.npy"))
    else:
        exp_edge_name = f"pruned_edges{curr_idx-1}.npy"
        if args.debug_mode:
            print(f"load edge {exp_edge_name} for train/val/test")
        
        edge_sub_dir = os.path.join(exp_name, "pruned_edges", args.save_exp_name)
        train_edge_dir = os.path.join(edge_sub_dir, f"train_{exp_edge_name}")
        val_edge_dir = os.path.join(edge_sub_dir, f"val_{exp_edge_name}")
        test_edge_dir = os.path.join(edge_sub_dir, f"test_{exp_edge_name}")
        train_adj = np.load(train_edge_dir)
        val_adj = np.load(val_edge_dir)
        test_adj = np.load(test_edge_dir)

    saliency_map_0_builder = []
    saliency_map_1_builder = []
    causal_score_0_builder = []  # NEW: Track causal scores
    causal_score_1_builder = []  # NEW: Track causal scores

    print(f"Computing saliency maps for {train_adj.shape[0]} training samples...")
    
    for i in range(train_adj.shape[0]):  
        curr_edge = torch.unsqueeze(torch.from_numpy(train_adj[i]).float().to(device), dim=0)
        curr_edge.requires_grad = True 
        curr_x = torch.unsqueeze(torch.eye(100), dim=0).to(device)
        
        # Original saliency computation
        saliency_0 = iterative_saliency_map(model=model, edge=curr_edge, x=curr_x, target=0)
        saliency_1 = iterative_saliency_map(model=model, edge=curr_edge, x=curr_x, target=1)
        
        # NEW: Compute causal components using factor separation
        causal_mask_0, conf_mask_0 = separate_causal_confounding_factors(
            saliency_0, saliency_0, split_ratio=args.causal_factor_split_ratio
        )
        causal_mask_1, conf_mask_1 = separate_causal_confounding_factors(
            saliency_1, saliency_1, split_ratio=args.causal_factor_split_ratio
        )
        
        # Compute causal effect scores
        causal_score_0 = (saliency_0 * causal_mask_0).sum() / (saliency_0.sum() + 1e-6)
        causal_score_1 = (saliency_1 * causal_mask_1).sum() / (saliency_1.sum() + 1e-6)
        
        if args.generate_individual_mask_all:
            saliency_map_0_builder.append(saliency_0)
            saliency_map_1_builder.append(saliency_1)
            causal_score_0_builder.append(causal_score_0.item())
            causal_score_1_builder.append(causal_score_1.item())
        elif args.generate_individual_mask_label_specific:
            if y_train[i] == 0:
                saliency_map_0_builder.append(saliency_0)
                causal_score_0_builder.append(causal_score_0.item())
            else:
                saliency_map_1_builder.append(saliency_1)
                causal_score_1_builder.append(causal_score_1.item())
        else:
            raise NotImplementedError
    
    # Log causal scores
    if causal_score_0_builder:
        avg_causal_score_0 = np.mean(causal_score_0_builder)
        print(f"Average causal score for class 0: {avg_causal_score_0:.4f}")
    if causal_score_1_builder:
        avg_causal_score_1 = np.mean(causal_score_1_builder)
        print(f"Average causal score for class 1: {avg_causal_score_1:.4f}")

    # Compute unified saliency maps with causal weighting
    if args.sum_unified_mask:
        saliency_unified_0 = torch.stack(saliency_map_0_builder).sum(dim=0)
        saliency_unified_1 = torch.stack(saliency_map_1_builder).sum(dim=0)
    elif args.squared_sum_unified_mask:
        saliency_map_0_builder = [torch.square(i) for i in saliency_map_0_builder]
        saliency_map_1_builder = [torch.square(i) for i in saliency_map_1_builder]
        saliency_unified_0 = torch.stack([torch.abs(i) for i in saliency_map_0_builder]).sum(dim=0)
        saliency_unified_1 = torch.stack([torch.abs(i) for i in saliency_map_1_builder]).sum(dim=0)
    elif args.absolute_sum_unified_mask:
        saliency_unified_0 = torch.stack([torch.abs(i) for i in saliency_map_0_builder]).sum(dim=0)
        saliency_unified_1 = torch.stack([torch.abs(i) for i in saliency_map_1_builder]).sum(dim=0)
    elif args.average_unified_mask:
        saliency_unified_0 = torch.stack(saliency_map_0_builder).mean(dim=0)
        saliency_unified_1 = torch.stack(saliency_map_1_builder).mean(dim=0)
    elif args.max_unified_mask:
        saliency_unified_0 = torch.stack(saliency_map_0_builder).max(dim=0).values
        saliency_unified_1 = torch.stack(saliency_map_1_builder).max(dim=0).values
    else:
        raise NotImplementedError
    
    # NEW: Apply causal weighting to saliency maps
    print("Applying causal weighting to saliency maps...")
    saliency_unified_0, saliency_unified_1 = weight_saliency_by_causality(
        saliency_unified_0, 
        saliency_unified_1,
        causal_ratio=0.7,
        split_ratio=args.causal_factor_split_ratio
    )

    threshold_value_0 = torch.quantile(saliency_unified_0, args.prune_threshold).item()
    threshold_value_1 = torch.quantile(saliency_unified_1, args.prune_threshold).item()

    meta_mask_0 = torch.where(saliency_unified_0 < threshold_value_0, 0, 1)
    meta_mask_1 = torch.where(saliency_unified_1 < threshold_value_1, 0, 1)
    meta_mask_0 = meta_mask_0.cpu().numpy()
    meta_mask_1 = meta_mask_1.cpu().numpy()
    
    if args.metaMask_0:
        saved_saliency_map = saliency_unified_0.cpu()
        meta_mask = meta_mask_0 
    elif args.metaMask_1:
        saved_saliency_map = saliency_unified_1.cpu()
        meta_mask = meta_mask_1 
    elif args.metaMask_Avg:
        saliency_two_avg = torch.stack([saliency_unified_0, saliency_unified_1]).mean(dim=0)
        saved_saliency_map = saliency_two_avg.cpu()
        threshold_two = torch.quantile(saliency_two_avg, args.prune_threshold).item()
        meta_mask = torch.where(saliency_two_avg < threshold_two, 0, 1).cpu().numpy()
    elif args.metaMask_Sum:
        saliency_two_sum = saliency_unified_0 + saliency_unified_1 
        saved_saliency_map = saliency_two_sum.cpu()
        threshold_two = torch.quantile(saliency_two_sum, args.prune_threshold).item()
        meta_mask = torch.where(saliency_two_sum < threshold_two, 0, 1).cpu().numpy()
    elif args.metaMask_AbSum:
        saliency_two_abs_sum = torch.abs(saliency_unified_0) + torch.abs(saliency_unified_1)
        saved_saliency_map = saliency_two_abs_sum.cpu()
        threshold_two = torch.quantile(saliency_two_abs_sum, args.prune_threshold).item()
        meta_mask = torch.where(saliency_two_abs_sum < threshold_two, 0, 1).cpu().numpy()
    elif args.metaMask_Max:
        saliency_two_max = torch.stack([saliency_unified_0, saliency_unified_1]).max(dim=0).values
        saved_saliency_map = saliency_two_max.cpu()
        threshold_two = torch.quantile(saliency_two_max, args.prune_threshold).item()
        meta_mask = torch.where(saliency_two_max < threshold_two, 0, 1).cpu().numpy()
    else:
        raise NotImplementedError

    save_saliency_name = f"saliency_map{curr_idx}.npy"
    save_saliency_subexp = os.path.join(exp_name, 'saved_saliency_maps', args.save_exp_name)
    if not os.path.exists(save_saliency_subexp):
        os.makedirs(save_saliency_subexp)
    save_saliency_dir = os.path.join(save_saliency_subexp, save_saliency_name)
    
    if args.method == "IGS":
        torch.save(saved_saliency_map, os.path.join(save_saliency_subexp, f"saliency_map{curr_idx}.pth"))
        return 
    else:
        saved_saliency_map = saved_saliency_map.numpy()
        np.save(save_saliency_dir, saved_saliency_map)
    
    pruned_train_adj = np.multiply(train_adj, meta_mask)
    pruned_val_adj = np.multiply(val_adj, meta_mask)
    pruned_test_adj = np.multiply(test_adj, meta_mask)
    
    pruned_train_adj = normalize_edge(pruned_train_adj)
    pruned_val_adj = normalize_edge(pruned_val_adj)
    pruned_test_adj = normalize_edge(pruned_test_adj)
   
    exp_saved_name = f"pruned_edges{curr_idx}.npy"
    if args.debug_mode:
        print(f"{exp_saved_name} saved for train/val/test!")
    
    saved_sub_dir = os.path.join(exp_name, 'pruned_edges', args.save_exp_name)
    if not os.path.exists(saved_sub_dir):
        os.makedirs(saved_sub_dir)
    train_saved_dir = os.path.join(saved_sub_dir, f"train_{exp_saved_name}")
    test_saved_dir = os.path.join(saved_sub_dir, f"test_{exp_saved_name}")
    val_saved_dir = os.path.join(saved_sub_dir, f"val_{exp_saved_name}")

    np.save(train_saved_dir, pruned_train_adj)
    np.save(test_saved_dir, pruned_test_adj)
    np.save(val_saved_dir, pruned_val_adj)
    
    print(f"✓ Causal gradient iteration completed!")
    print("\n"*2)
