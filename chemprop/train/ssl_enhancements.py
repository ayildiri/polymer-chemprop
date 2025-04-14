# ssl_enhancements.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import math
from torch.optim.lr_scheduler import LambdaLR

def improve_model_initialization(model):
    """Improve model initialization for better convergence"""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) >= 2:
                # Use Kaiming initialization for convolutional/linear layers
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            else:
                # Use normal initialization for 1D weights
                nn.init.normal_(param, mean=0.0, std=0.01)
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    return model

def create_improved_scheduler(optimizer, epochs, warmup_epochs=None):
    """Create a more sophisticated learning rate scheduler with proper initialization"""
    # Create a warmup scheduler with cosine annealing
    if warmup_epochs is None:
        warmup_epochs = min(5, epochs // 10)
    
    # Important: Force set the initial learning rate
    for param_group in optimizer.param_groups:
        initial_lr = param_group['lr']  # Store the original learning rate
        param_group['initial_lr'] = initial_lr  # Make sure this is set for LambdaLR
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup, starting from a non-zero value (10% of lr)
            return 0.1 + 0.9 * (epoch / warmup_epochs)
        else:
            # Cosine decay
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
    
    return LambdaLR(optimizer, lr_lambda)

class ImprovedNodeEdgeMaskingTask:
    """Enhanced node and edge masking task with better loss weighting"""
    def __init__(self, model, mask_ratio=0.15, min_mask=2, edge_weight=1.5):
        self.model = model
        self.mask_ratio = mask_ratio
        self.min_mask = min_mask
        self.edge_weight = edge_weight
    
    def train_step(self, batch, optimizer, device):
        atom_feats = batch['atom_feats'].to(device)
        edge_src = batch['edge_src'].to(device)
        edge_dst = batch['edge_dst'].to(device)
        edge_feats = batch['edge_feats'].to(device)
        edge_weights = batch['edge_weights'].to(device) if batch['edge_weights'].numel() > 0 else torch.tensor([], device=device)
        b2rev = batch['b2rev'].to(device)
        node_to_graph = batch['node_to_graph'].to(device)
        batch_size = batch['batch_size']
        
        # Randomly select nodes and edges to mask
        mask_atom_indices = []
        mask_edge_indices = []
        
        for g_idx in range(batch_size):
            # Find nodes belonging to this graph
            node_indices = torch.nonzero(node_to_graph == g_idx, as_tuple=True)[0]
            # Find edges with source in this graph
            edge_indices = torch.nonzero(node_to_graph[edge_src] == g_idx, as_tuple=True)[0]
            
            if node_indices.numel() > 0:
                # Calculate number of nodes to mask (either min_mask or mask_ratio, whichever is greater)
                num_mask = max(self.min_mask, int(self.mask_ratio * node_indices.numel()))
                num_mask = min(num_mask, node_indices.numel())  # Don't exceed total nodes
                
                perm = torch.randperm(node_indices.numel())[:num_mask]
                sel_nodes = node_indices[perm]
                mask_atom_indices.extend(sel_nodes.tolist())
            
            if edge_indices.numel() > 0:
                # Increase edge masking slightly for better learning
                num_mask = max(self.min_mask, int((self.mask_ratio * 1.2) * edge_indices.numel() // 2))
                num_mask = min(num_mask, edge_indices.numel() // 2)
                
                perm_e = torch.randperm(edge_indices.numel())[:num_mask]
                sel_edges = edge_indices[perm_e]
                for ei in sel_edges:
                    ei = int(ei.item())
                    mask_edge_indices.append(ei)
                    rev_ei = int(b2rev[ei].item())
                    if rev_ei >= 0 and rev_ei not in mask_edge_indices:
                        mask_edge_indices.append(rev_ei)
        
        mask_atom_indices = list(set(mask_atom_indices))
        mask_edge_indices = list(set(mask_edge_indices))
        
        # Store original feature values
        orig_atom_feats = atom_feats.clone()
        orig_edge_feats = edge_feats.clone()
        
        # Mask features by zeroing them out
        if mask_atom_indices:
            atom_feats[mask_atom_indices] = 0.0
        if mask_edge_indices:
            edge_feats[mask_edge_indices] = 0.0
        
        # Forward pass
        pred_node, pred_edge, _, _, _, _ = self.model(atom_feats, edge_src, edge_dst, edge_feats, edge_weights, b2rev, node_to_graph)
        
        # Compute losses with edge task weighting
        loss_node = 0.0
        loss_edge = 0.0
        
        if mask_atom_indices:
            true_node_feats = orig_atom_feats[mask_atom_indices]
            pred_node_feats = pred_node[mask_atom_indices]
            loss_node = F.mse_loss(pred_node_feats, true_node_feats)
        
        if mask_edge_indices:
            true_edge_feats = orig_edge_feats[mask_edge_indices]
            pred_edge_feats = pred_edge[mask_edge_indices]
            loss_edge = F.mse_loss(pred_edge_feats, true_edge_feats) * self.edge_weight
        
        loss = loss_node + loss_edge
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'loss_node': loss_node.item() if mask_atom_indices else 0.0,
            'loss_edge': loss_edge.item() if mask_edge_indices else 0.0
        }
    
    def eval_step(self, batch, device):
        atom_feats = batch['atom_feats'].to(device)
        edge_src = batch['edge_src'].to(device)
        edge_dst = batch['edge_dst'].to(device)
        edge_feats = batch['edge_feats'].to(device)
        edge_weights = batch['edge_weights'].to(device) if batch['edge_weights'].numel() > 0 else torch.tensor([], device=device)
        b2rev = batch['b2rev'].to(device)
        node_to_graph = batch['node_to_graph'].to(device)
        batch_size = batch['batch_size']
        
        # Randomly select nodes and edges to mask
        mask_atom_indices = []
        mask_edge_indices = []
        
        for g_idx in range(batch_size):
            node_indices = torch.nonzero(node_to_graph == g_idx, as_tuple=True)[0]
            edge_indices = torch.nonzero(node_to_graph[edge_src] == g_idx, as_tuple=True)[0]
            
            if node_indices.numel() > 0:
                num_mask = max(self.min_mask, int(self.mask_ratio * node_indices.numel()))
                num_mask = min(num_mask, node_indices.numel())
                
                perm = torch.randperm(node_indices.numel())[:num_mask]
                sel_nodes = node_indices[perm]
                mask_atom_indices.extend(sel_nodes.tolist())
            
            if edge_indices.numel() > 0:
                num_mask = max(self.min_mask, int((self.mask_ratio * 1.2) * edge_indices.numel() // 2))
                num_mask = min(num_mask, edge_indices.numel() // 2)
                
                perm_e = torch.randperm(edge_indices.numel())[:num_mask]
                sel_edges = edge_indices[perm_e]
                for ei in sel_edges:
                    ei = int(ei.item())
                    mask_edge_indices.append(ei)
                    rev_ei = int(b2rev[ei].item())
                    if rev_ei >= 0 and rev_ei not in mask_edge_indices:
                        mask_edge_indices.append(rev_ei)
        
        mask_atom_indices = list(set(mask_atom_indices))
        mask_edge_indices = list(set(mask_edge_indices))
        
        orig_atom_feats = atom_feats.clone()
        orig_edge_feats = edge_feats.clone()
        
        if mask_atom_indices:
            atom_feats[mask_atom_indices] = 0.0
        if mask_edge_indices:
            edge_feats[mask_edge_indices] = 0.0
        
        with torch.no_grad():
            pred_node, pred_edge, _, graph_embeds, node_repr, edge_repr = self.model(atom_feats, edge_src, edge_dst, edge_feats, edge_weights, b2rev, node_to_graph)
        
        loss_node = 0.0
        loss_edge = 0.0
        
        if mask_atom_indices:
            true_node_feats = orig_atom_feats[mask_atom_indices]
            pred_node_feats = pred_node[mask_atom_indices]
            loss_node = F.mse_loss(pred_node_feats, true_node_feats).item()
        
        if mask_edge_indices:
            true_edge_feats = orig_edge_feats[mask_edge_indices]
            pred_edge_feats = pred_edge[mask_edge_indices]
            loss_edge = F.mse_loss(pred_edge_feats, true_edge_feats).item() * self.edge_weight
        
        loss = loss_node + loss_edge
        
        return {
            'loss': loss,
            'loss_node': loss_node,
            'loss_edge': loss_edge,
            'graph_embeds': graph_embeds,
            'node_repr': node_repr,
            'edge_repr': edge_repr
        }

class ImprovedGraphLevelTask:
    """Enhanced graph-level task with better regularization"""
    def __init__(self, model, graph_loss_weight=1.0):
        self.model = model
        self.graph_loss_weight = graph_loss_weight
        
        # Add auxiliary prediction head
        hidden_size = model.hidden_size
        self.aux_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(next(model.parameters()).device)
    
    def train_step(self, batch, optimizer, device):
        atom_feats = batch['atom_feats'].to(device)
        edge_src = batch['edge_src'].to(device)
        edge_dst = batch['edge_dst'].to(device)
        edge_feats = batch['edge_feats'].to(device)
        edge_weights = batch['edge_weights'].to(device) if batch['edge_weights'].numel() > 0 else torch.tensor([], device=device)
        b2rev = batch['b2rev'].to(device)
        node_to_graph = batch['node_to_graph'].to(device)
        mol_weights = batch['mol_weights'].to(device)
        
        # Forward pass
        _, _, pred_graph, graph_embeds, _, _ = self.model(atom_feats, edge_src, edge_dst, edge_feats, edge_weights, b2rev, node_to_graph)
        
        # Add auxiliary prediction
        aux_pred = self.aux_head(graph_embeds)
        
        # Better normalization with robust scaling
        mean_weight = torch.mean(mol_weights)
        std_weight = torch.std(mol_weights)
        if std_weight > 0:
            mol_weights_scaled = (mol_weights - mean_weight) / std_weight
        else:
            mol_weights_scaled = mol_weights - mean_weight
        
        # Compute losses with regularization
        main_loss = F.mse_loss(pred_graph, mol_weights_scaled)
        aux_loss = F.mse_loss(aux_pred.squeeze(-1), mol_weights_scaled)
        # Add small L2 regularization for graph embeddings
        embedding_reg = 0.001 * (graph_embeds ** 2).mean()
        
        loss = (main_loss + 0.5 * aux_loss + embedding_reg) * self.graph_loss_weight
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'loss_graph': main_loss.item()
        }
    
    def eval_step(self, batch, device):
        atom_feats = batch['atom_feats'].to(device)
        edge_src = batch['edge_src'].to(device)
        edge_dst = batch['edge_dst'].to(device)
        edge_feats = batch['edge_feats'].to(device)
        edge_weights = batch['edge_weights'].to(device) if batch['edge_weights'].numel() > 0 else torch.tensor([], device=device)
        b2rev = batch['b2rev'].to(device)
        node_to_graph = batch['node_to_graph'].to(device)
        mol_weights = batch['mol_weights'].to(device)
        
        # Better normalization
        mean_weight = torch.mean(mol_weights)
        std_weight = torch.std(mol_weights)
        if std_weight > 0:
            mol_weights_scaled = (mol_weights - mean_weight) / std_weight
        else:
            mol_weights_scaled = mol_weights - mean_weight
        
        with torch.no_grad():
            _, _, pred_graph, graph_embeds, node_repr, edge_repr = self.model(atom_feats, edge_src, edge_dst, edge_feats, edge_weights, b2rev, node_to_graph)
            aux_pred = self.aux_head(graph_embeds)
        
        main_loss = F.mse_loss(pred_graph, mol_weights_scaled).item()
        aux_loss = F.mse_loss(aux_pred.squeeze(-1), mol_weights_scaled).item()
        embedding_reg = 0.001 * (graph_embeds ** 2).mean().item()
        
        loss = (main_loss + 0.5 * aux_loss + embedding_reg) * self.graph_loss_weight
        
        return {
            'loss': loss,
            'loss_graph': main_loss,
            'graph_embeds': graph_embeds,
            'node_repr': node_repr,
            'edge_repr': edge_repr
        }

def apply_graph_augmentation(graph_batch, ratio=0.3):
    """Apply stochastic augmentation to the graph batch"""
    # In a real implementation, this would modify the actual graph structure
    # This is a simplified version for demonstration
    atom_feats = graph_batch['atom_feats']
    edge_weights = graph_batch['edge_weights']
    
    # Only modify a fraction of graphs in the batch
    if random.random() < ratio and edge_weights.numel() > 0:
        # Clone to avoid modifying the original
        edge_weights_aug = edge_weights.clone()
        
        # Only modify inter-monomer edges (those with weight < 1.0)
        mask = edge_weights_aug < 1.0
        if mask.sum() > 0:
            # Add small random perturbation
            perturb = torch.randn_like(edge_weights_aug) * 0.05
            edge_weights_aug[mask] = torch.clamp(edge_weights_aug[mask] + perturb[mask], 0.01, 0.99)
            graph_batch['edge_weights'] = edge_weights_aug
    
    return graph_batch

def improved_weight_transfer(model, checkpoint, transfer_strategy='c'):
    """Improved weight transfer with gradual unfreezing"""
    state_dict = checkpoint['model_state_dict']
    
    if transfer_strategy == 'a':
        # Strategy A: only message-passing layers
        keys_to_keep = [k for k in state_dict.keys() 
                      if any(name in k for name in ['W_initial', 'W_message', 'W_node'])]
    elif transfer_strategy == 'b':
        # Strategy B: message-passing + first two FC layers
        keys_to_keep = [k for k in state_dict.keys() 
                      if (any(name in k for name in ['W_initial', 'W_message', 'W_node']) or
                          k.startswith('graph_head.0') or k.startswith('graph_head.2'))]
    else:  # strategy == 'c'
        # Strategy C: all layers
        keys_to_keep = list(state_dict.keys())
    
    # Create filtered state dict
    filtered_state = {k: v for k, v in state_dict.items() if k in keys_to_keep}
    
    # Load state dict with filtered keys
    model.load_state_dict(filtered_state, strict=False)
    
    return model
