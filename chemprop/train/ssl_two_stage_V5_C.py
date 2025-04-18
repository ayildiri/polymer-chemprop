"""
Improved implementation of two-stage SSL pretraining for polymer property prediction.
It performs:
1. Node- and edge-level pretraining (masking atoms/bonds and predicting their features)
2. Graph-level pretraining (predicting ensemble polymer molecular weight)
3. Various weight transfer strategies for downstream fine-tuning

Key improvements:
- Improved sequential training between stages
- More careful weight transfer mechanism
- Adjusted hyperparameters
"""
import os
import math
import logging
import random
import argparse
import numpy as np
import pandas as pd
import pickle
import time
import json
from datetime import datetime
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
    from rdkit.Chem import Descriptors
except ImportError:
    Chem = None
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from chemprop.features.featurization import atom_features, bond_features, set_polymer
from ssl_enhancements import (ImprovedNodeEdgeMaskingTask, ImprovedGraphLevelTask, 
                            apply_graph_augmentation, improve_model_initialization,
                            create_improved_scheduler, improved_weight_transfer)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hardcoded SMILES column name
SMILES_COL = "poly_chemprop_input"

def one_hot_encoding(value, choices):
    """One-hot encoding with an extra 'unknown' category at end if value not in choices."""
    encoding = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1
    return encoding

def get_atom_features(atom):
    return atom_features(atom)

def get_bond_features(bond):
    return bond_features(bond)


class PolymerGraph:
    """Data structure for a polymer graph with precomputed features."""
    def __init__(self):
        self.n_atoms = 0
        self.n_edges = 0
        self.atom_features = []   # list of list[float]
        self.edge_index = []      # list of (src_idx, dest_idx)
        self.edge_features = []   # list of list[float]
        self.edge_weights = []    # list of float
        self.b2rev = []           # reverse edge mapping list
        self.mol_weight = 0.0     # pseudo-label (ensemble molecular weight)
        self.smiles = None  
        self.monomer_weights = [] # weights of each monomer
        self.monomer_types = []   # SMILES of each monomer

class PolymerDataset(Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        return self.graphs[idx]

def collate_graphs(batch_graphs):
    """Custom collate function to combine a list of PolymerGraph objects into a batch."""
    total_atoms = sum(g.n_atoms for g in batch_graphs)
    total_edges = sum(g.n_edges for g in batch_graphs)
    atom_feat_dim = len(batch_graphs[0].atom_features[0]) if batch_graphs[0].n_atoms > 0 else 0
    bond_feat_dim = len(batch_graphs[0].edge_features[0]) if batch_graphs[0].n_edges > 0 else 0
    combined_atom_feats = torch.zeros(total_atoms, atom_feat_dim, dtype=torch.float32)
    combined_edge_feats = torch.zeros(total_edges, bond_feat_dim, dtype=torch.float32)
    edge_src_list = []
    edge_dst_list = []
    b2rev_list = []
    node_to_graph = torch.zeros(total_atoms, dtype=torch.long)
    mol_weights = []
    atom_offset = 0
    edge_offset = 0
    for graph_idx, g in enumerate(batch_graphs):
        n = g.n_atoms
        e = g.n_edges
        if n > 0:
            combined_atom_feats[atom_offset:atom_offset+n] = torch.tensor(g.atom_features, dtype=torch.float32)
        if e > 0:
            combined_edge_feats[edge_offset:edge_offset+e] = torch.tensor(g.edge_features, dtype=torch.float32)
        for local_idx, (src, dst) in enumerate(g.edge_index):
            edge_src_list.append(src + atom_offset)
            edge_dst_list.append(dst + atom_offset)
        b2rev_list.extend([rev_idx + edge_offset if rev_idx is not None else None for rev_idx in g.b2rev])
        if n > 0:
            node_to_graph[atom_offset:atom_offset+n] = graph_idx
        mol_weights.append(g.mol_weight)
        atom_offset += n
        edge_offset += e
    edge_src = torch.tensor(edge_src_list, dtype=torch.long)
    edge_dst = torch.tensor(edge_dst_list, dtype=torch.long)
    edge_weights = torch.cat([torch.tensor(g.edge_weights, dtype=torch.float32) for g in batch_graphs], dim=0) if total_edges > 0 else torch.tensor([], dtype=torch.float32)
    # Convert b2rev None to int (self-loops won't be used, but ensure tensor dtype consistency)
    b2rev_tensor = []
    for val in b2rev_list:
        b2rev_tensor.append(-1 if val is None else val)
    b2rev = torch.tensor(b2rev_tensor, dtype=torch.long)
    mol_weights = torch.tensor(mol_weights, dtype=torch.float32)
    return {
        'atom_feats': combined_atom_feats,
        'edge_src': edge_src,
        'edge_dst': edge_dst,
        'edge_feats': combined_edge_feats,
        'edge_weights': edge_weights,
        'b2rev': b2rev,
        'node_to_graph': node_to_graph,
        'batch_size': len(batch_graphs),
        'mol_weights': mol_weights,
        'smiles': [g.smiles for g in batch_graphs]
    }

class SSLPretrainModel(nn.Module):
    """
    Model supporting both node/edge-level and graph-level SSL tasks.
    This implements the wD-MPNN (weighted Direct Message Passing Neural Network)
    architecture as described in the paper.
    """
    def __init__(self, atom_feat_dim, bond_feat_dim, hidden_size, depth, dropout):
        super(SSLPretrainModel, self).__init__()
        self.atom_feat_dim = atom_feat_dim
        self.bond_feat_dim = bond_feat_dim
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        
        # Initial message transformation (atom + bond -> hidden)
        self.W_initial = nn.Linear(atom_feat_dim + bond_feat_dim, hidden_size, bias=False)
        
        # Message passing update transformation (hidden -> hidden)
        self.W_message = nn.Linear(hidden_size, hidden_size)
        
        # Node update transformation
        self.W_node = nn.Linear(atom_feat_dim + hidden_size, hidden_size)
        
        # Prediction heads
        self.node_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, atom_feat_dim)
        )
        
        self.edge_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, bond_feat_dim)
        )
        
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, atom_feats, edge_src, edge_dst, edge_feats, edge_weights, b2rev, node_to_graph):
        # Initial directed edge hidden states
        src_atom_feats = atom_feats[edge_src]  # [E, atom_feat_dim]
        edge_input = torch.cat([src_atom_feats, edge_feats], dim=1)  # [E, atom_feat_dim + bond_feat_dim]
        hidden_edges = F.relu(self.W_initial(edge_input))
        hidden_edges = F.dropout(hidden_edges, p=self.dropout, training=self.training)
        
        E = hidden_edges.size(0)
        N = atom_feats.size(0)
        
        # Edge-centered message passing
        for t in range(self.depth):
            M = torch.zeros((N, self.hidden_size), device=hidden_edges.device)
            if E > 0:
                # Aggregate messages from neighbors (excluding reverse edge)
                M.index_add_(0, edge_dst, edge_weights.unsqueeze(1) * hidden_edges)
                
                # Get reverse edge hidden states
                rev_hidden = torch.zeros_like(hidden_edges)
                valid_rev_mask = (b2rev >= 0)
                rev_hidden[valid_rev_mask] = hidden_edges[b2rev[valid_rev_mask]]
                
                # Subtract reverse edge contribution to get directed message
                rev_weight = torch.zeros_like(edge_weights)
                rev_weight[valid_rev_mask] = edge_weights[b2rev[valid_rev_mask]]
                msg = M[edge_src] - rev_weight.unsqueeze(1) * rev_hidden
            else:
                msg = torch.zeros((0, self.hidden_size), device=hidden_edges.device)
            
            # Update edge hidden states
            hidden_edges = F.relu(self.W_message(msg))
            hidden_edges = F.dropout(hidden_edges, p=self.dropout, training=self.training)
        
        # Final node embeddings
        node_repr = torch.zeros((N, self.hidden_size), device=hidden_edges.device)
        if E > 0:
            # Aggregate messages to nodes
            node_repr.index_add_(0, edge_dst, edge_weights.unsqueeze(1) * hidden_edges)
            
            # Concatenate with original node features and update
            node_input = torch.cat([atom_feats, node_repr], dim=1)
            node_repr = F.relu(self.W_node(node_input))
            node_repr = F.dropout(node_repr, p=self.dropout, training=self.training)
        
        # Graph embeddings by sum pooling
        batch_size = int(node_to_graph.max().item()) + 1 if node_to_graph.numel() > 0 else 0
        graph_embeds = torch.zeros((batch_size, self.hidden_size), device=hidden_edges.device)
        if N > 0:
            graph_embeds.index_add_(0, node_to_graph, node_repr)
        
        # Predictions
        pred_node = self.node_head(node_repr)
        pred_edge = self.edge_head(hidden_edges)
        pred_graph = self.graph_head(graph_embeds).squeeze(-1)  # shape [batch_size]
        
        return pred_node, pred_edge, pred_graph, graph_embeds, node_repr, hidden_edges

def parse_polymer_smiles(polymer_smiles):
    """Parse extended polymer SMILES input into monomer parts, ratios, edges and Xn."""
    s = polymer_smiles.strip()
    parts = s.split('|')
    monomer_smiles_str = parts[0]
    monomer_list = monomer_smiles_str.split('.') if monomer_smiles_str else []
    ratios = []
    edges_str = ""
    if len(parts) > 1:
        num_monomers = len(monomer_list)
        ratio_parts = parts[1:1+num_monomers]
        try:
            ratios = [float(x) for x in ratio_parts]
        except:
            ratios = []
        if len(parts) > 1 + num_monomers:
            edges_str = parts[1+num_monomers]
    if not ratios or len(ratios) != len(monomer_list):
        if len(monomer_list) > 0:
            ratios = [1.0] * len(monomer_list)
    Xn = None
    if '~' in edges_str:
        try:
            edge_part, xn_part = edges_str.split('~', 1)
            Xn = float(xn_part)
        except:
            Xn = None
        edges_str = edge_part
    edges = []
    if edges_str:
        tokens = edges_str.split('<')
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            try:
                ij, w1, w2 = token.split(':')
                i_str, j_str = ij.split('-')
                i = int(i_str)
                j = int(j_str)
                w1 = float(w1)
                w2 = float(w2)
                edges.append((i, j, w1, w2))
            except:
                continue
    return monomer_list, ratios, edges, Xn

def build_polymer_graph(smiles):
    """Build PolymerGraph from an extended polymer SMILES string."""
    graph = PolymerGraph()
    graph.smiles = smiles
    monomers, ratios, edges_info, Xn = parse_polymer_smiles(smiles)
    if len(monomers) == 0:
        return None
    
    # Store monomer types and their ratios for later use
    graph.monomer_types = monomers
    # Normalize ratios to sum to 1 (as in the paper)
    total_ratio = sum(ratios)
    norm_ratios = [r/total_ratio for r in ratios] if total_ratio > 0 else ratios
    graph.monomer_weights = norm_ratios
    
    # Calculate ensemble molecular weight as weighted average of monomer weights
    # This follows the paper's approach more closely
    total_weight = 0.0
    for m_smiles, frac in zip(monomers, norm_ratios):
        mol_m = Chem.MolFromSmiles(m_smiles)
        if mol_m is None:
            logging.warning(f"RDKit failed to parse monomer: {m_smiles}")
            continue
        # Use RDKit's built-in molecular weight calculator for consistency
        weight = Descriptors.MolWt(mol_m)
        total_weight += frac * weight
    
    # Scale based on degree of polymerization if available (Xn)
    if Xn is not None and Xn > 1:
        # Apply logarithmic scaling as mentioned in paper
        scale_factor = 1.0 + math.log(Xn)
        total_weight *= scale_factor
    
    graph.mol_weight = total_weight
    
    # Build the graph structure
    ensemble_mol = Chem.MolFromSmiles('.'.join(monomers))
    if ensemble_mol is None:
        logging.error(f"Failed to parse polymer SMILES: {smiles}")
        return None
    
    dummy_index_map = {}
    for atom in ensemble_mol.GetAtoms():
        feat = get_atom_features(atom)
        graph.atom_features.append(feat)
        idx = atom.GetIdx()
        graph.n_atoms += 1
        if atom.GetSymbol() == '*' and atom.GetAtomMapNum() != 0:
            dummy_index_map[atom.GetAtomMapNum()] = idx
    
    # Get bond feature dimension from a sample bond if available or use default
    bond_feat_dim = 0
    if ensemble_mol.GetNumBonds() > 0:
        sample_bond = ensemble_mol.GetBondWithIdx(0)
        sample_bf = get_bond_features(sample_bond)
        bond_feat_dim = len(sample_bf)
    
    # Add edges within monomers
    for bond in ensemble_mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bf = get_bond_features(bond)
        
        e_index = graph.n_edges
        graph.edge_index.append((u, v))
        graph.edge_features.append(bf)
        graph.edge_weights.append(1.0)
        graph.b2rev.append(None)
        graph.n_edges += 1
        
        rev_index = graph.n_edges
        graph.edge_index.append((v, u))
        graph.edge_features.append(bf)
        graph.edge_weights.append(1.0)
        graph.b2rev.append(None)
        graph.n_edges += 1
        
        graph.b2rev[e_index] = rev_index
        graph.b2rev[rev_index] = e_index
    
    # Add stochastic edges between monomers
    weight_factor = 1.0
    if Xn is not None:
        try:
            weight_factor = 1.0 + math.log(Xn)
        except ValueError:
            weight_factor = 1.0
    
    # Add edges between monomers based on the edge info
    for u, v, w1, w2 in edges_info:
        if u in dummy_index_map and v in dummy_index_map:
            u = dummy_index_map[u]
            v = dummy_index_map[v]
            
            # Create empty bond features
            bf = [0.0] * bond_feat_dim if bond_feat_dim > 0 else []
            
            e_index = graph.n_edges
            graph.edge_index.append((u, v))
            graph.edge_features.append(bf)
            graph.edge_weights.append(w1 * weight_factor)
            graph.b2rev.append(None)
            graph.n_edges += 1
            
            rev_index = graph.n_edges
            graph.edge_index.append((v, u))
            graph.edge_features.append(bf)
            graph.edge_weights.append(w2 * weight_factor)
            graph.b2rev.append(None)
            graph.n_edges += 1
            
            graph.b2rev[e_index] = rev_index
            graph.b2rev[rev_index] = e_index
            
    return graph

class NodeEdgeMaskingTask:
    """
    Node and edge masking task for self-supervised learning.
    This corresponds to the node- and edge-level pretraining task from the paper.
    """
    def __init__(self, model, mask_ratio=0.15, min_mask=2):
        self.model = model
        self.mask_ratio = mask_ratio
        self.min_mask = min_mask
    
    def train_step(self, batch, optimizer, device):
        # Transfer batch to device
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
                # Similar for edges, but considering edge pairs (forward/reverse)
                num_mask = max(self.min_mask, int(self.mask_ratio * edge_indices.numel() // 2))
                num_mask = min(num_mask, edge_indices.numel() // 2)  # Don't exceed total edge pairs
                
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
        
        # Compute losses
        loss_node = 0.0
        loss_edge = 0.0
        
        if mask_atom_indices:
            true_node_feats = orig_atom_feats[mask_atom_indices]
            pred_node_feats = pred_node[mask_atom_indices]
            loss_node = F.mse_loss(pred_node_feats, true_node_feats)
        
        if mask_edge_indices:
            true_edge_feats = orig_edge_feats[mask_edge_indices]
            pred_edge_feats = pred_edge[mask_edge_indices]
            loss_edge = F.mse_loss(pred_edge_feats, true_edge_feats)
        
        loss = loss_node + loss_edge
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'loss_node': loss_node.item() if mask_atom_indices else 0.0,
            'loss_edge': loss_edge.item() if mask_edge_indices else 0.0
        }
    
    def eval_step(self, batch, device):
        # Similar to train_step but without optimization
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
                num_mask = max(self.min_mask, int(self.mask_ratio * edge_indices.numel() // 2))
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
            loss_edge = F.mse_loss(pred_edge_feats, true_edge_feats).item()
        
        loss = loss_node + loss_edge
        
        return {
            'loss': loss,
            'loss_node': loss_node,
            'loss_edge': loss_edge,
            'graph_embeds': graph_embeds,
            'node_repr': node_repr,
            'edge_repr': edge_repr
        }

class GraphLevelTask:
    """
    Graph-level task for self-supervised learning.
    This corresponds to the graph-level pretraining task from the paper,
    which predicts the ensemble polymer molecular weight.
    """
    def __init__(self, model, graph_loss_weight=1.0):
        self.model = model
        self.graph_loss_weight = graph_loss_weight
    
    def train_step(self, batch, optimizer, device):
        # Transfer batch to device
        atom_feats = batch['atom_feats'].to(device)
        edge_src = batch['edge_src'].to(device)
        edge_dst = batch['edge_dst'].to(device)
        edge_feats = batch['edge_feats'].to(device)
        edge_weights = batch['edge_weights'].to(device) if batch['edge_weights'].numel() > 0 else torch.tensor([], device=device)
        b2rev = batch['b2rev'].to(device)
        node_to_graph = batch['node_to_graph'].to(device)
        mol_weights = batch['mol_weights'].to(device)
        
        # Forward pass
        _, _, pred_graph, _, _, _ = self.model(atom_feats, edge_src, edge_dst, edge_feats, edge_weights, b2rev, node_to_graph)
        
        # Normalize predictions and targets to improve training
        # This scaling helps with convergence for molecular weight prediction
        if torch.std(mol_weights) > 0:
            mol_weights_scaled = (mol_weights - torch.mean(mol_weights)) / torch.std(mol_weights)
        else:
            mol_weights_scaled = mol_weights
        
        # Compute loss
        loss = F.mse_loss(pred_graph, mol_weights_scaled) * self.graph_loss_weight
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            'loss': loss.item(),
            'loss_graph': loss.item()
        }
    
  
    def eval_step(self, batch, device):
        # Similar to train_step but without optimization
        atom_feats = batch['atom_feats'].to(device)
        edge_src = batch['edge_src'].to(device)
        edge_dst = batch['edge_dst'].to(device)
        edge_feats = batch['edge_feats'].to(device)
        edge_weights = batch['edge_weights'].to(device) if batch['edge_weights'].numel() > 0 else torch.tensor([], device=device)
        b2rev = batch['b2rev'].to(device)
        node_to_graph = batch['node_to_graph'].to(device)
        mol_weights = batch['mol_weights'].to(device)
    
        # Normalize for consistency with training
        if torch.std(mol_weights) > 0:
            mol_weights_scaled = (mol_weights - torch.mean(mol_weights)) / torch.std(mol_weights)
        else:
            mol_weights_scaled = mol_weights
        
        with torch.no_grad():
            _, _, pred_graph, graph_embeds, node_repr, edge_repr = self.model(atom_feats, edge_src, edge_dst, edge_feats, edge_weights, b2rev, node_to_graph)
        
        loss = F.mse_loss(pred_graph, mol_weights_scaled).item() * self.graph_loss_weight
        
        return {
            'loss': loss,
            'loss_graph': loss,
            'graph_embeds': graph_embeds,
            'node_repr': node_repr,
            'edge_repr': edge_repr
        }

def run_pretraining_epoch(task, loader, device, optimizer=None, train=True, scheduler=None, use_enhanced_ssl=False):
    """Run one epoch of pretraining (either training or validation)."""
    epoch_losses = []
    task_losses = {}
    
    for batch in loader:
        # Apply graph augmentation if enhanced SSL is enabled and in training mode
        if train and use_enhanced_ssl and random.random() < 0.3:
            batch = apply_graph_augmentation(batch)
            
        if train:
            losses = task.train_step(batch, optimizer, device)
            # Apply per-batch learning rate scheduler if provided
            if scheduler is not None and hasattr(scheduler, 'step_batch'):
                scheduler.step_batch()
        else:
            losses = task.eval_step(batch, device)
        
        epoch_losses.append(losses['loss'])
        for key, value in losses.items():
            if key != 'loss' and key not in ['graph_embeds', 'node_repr', 'edge_repr']:
                if key not in task_losses:
                    task_losses[key] = []
                task_losses[key].append(value)
    
    avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    avg_task_losses = {k: float(np.mean(v)) if v else 0.0 for k, v in task_losses.items()}
    
    return avg_loss, avg_task_losses


# ----- Additional utility function to help with verifying split data -----

def analyze_pretrain_folds(pretrain_folds_file, total_count=None):
    """
    Analyze the format and content of a pretrain folds file.
    
    Args:
        pretrain_folds_file: Path to the pretrain folds file
        total_count: Total number of samples in the dataset (if known)
        
    Returns:
        A dictionary with information about the folds
    """
    with open(pretrain_folds_file, "rb") as f:
        data = pickle.load(f)
        
    if not isinstance(data, list):
        return {"error": "Not a list"}
    
    # Check if it's a list of integers (indices)
    if all(isinstance(x, int) for x in data):
        return {
            "format": "indices",
            "count": len(data),
            "percentage": len(data) / total_count if total_count else None,
            "min_index": min(data) if data else None,
            "max_index": max(data) if data else None
        }
    
    # Check if it's a list of 0s and 1s (folds)
    if total_count and len(data) == total_count and all(x is None or x in [0, 1] for x in data):
        pretrain_count = data.count(0)
        unused_count = data.count(1)
        none_count = data.count(None)
        
        return {
            "format": "folds",
            "total_length": len(data),
            "pretrain_count": pretrain_count,
            "unused_count": unused_count,
            "none_count": none_count,
            "pretrain_percentage": pretrain_count / total_count if total_count else None
        }
        
    return {"error": "Unknown format", "length": len(data)}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV file with poly_chemprop_input column.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the pretrained model.')
    parser.add_argument('--polymer', action='store_true', help='Use polymer-specific atom featurization.')
    parser.add_argument('--use_enhanced_ssl', action='store_true', help='Use enhanced SSL techniques for better performance')
    parser.add_argument('--pretrain_frac', type=float, default=1.0, help='Fraction of dataset to use for pretraining.')
    parser.add_argument('--val_frac', type=float, default=0.1, help='Fraction of data to use for validation.')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='Ratio of nodes/edges to mask in masking task.')
    parser.add_argument('--min_mask', type=int, default=2, help='Minimum number of nodes/edges to mask per graph.')
    parser.add_argument('--graph_loss_weight', type=float, default=0.01, help='Weight applied to the graph-level loss (default: 0.01)')
    parser.add_argument('--pretrain_folds_file', type=str, default=None, help='Optional path to a pickle file defining pretrain splits')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs for each pretraining stage.')
    parser.add_argument('--epochs_node_edge', type=int, default=None, help='Number of epochs for node-edge level pretraining (if None, uses --epochs value).')
    parser.add_argument('--epochs_graph', type=int, default=None, help='Number of epochs for graph level pretraining (if None, uses --epochs value).')
    parser.add_argument('--early_stop_patience', type=int, default=40, help='Patience for early stopping.')
    parser.add_argument('--scheduler_patience', type=int, default=10, help='Patience for LR scheduler.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--learning_rate_graph', type=float, default=None, help='Learning rate for graph-level task (defaults to --learning_rate).')
    parser.add_argument('--hidden_size', type=int, default=300, help='Hidden dimensionality for GNN.')
    parser.add_argument('--depth', type=int, default=5, help='Number of message passing steps.')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--save_graph_embeddings', action='store_true', help='Whether to save graph-level embeddings after SSL training.')
    parser.add_argument('--graph_embeddings_path', type=str, default=None, help='Path to save graph-level embeddings (as .npy).')
    parser.add_argument('--no_cuda', action='store_true', help='Disable GPU usage even if available.')
    parser.add_argument('--dataset_type', type=str, default='regression', help='Dataset type (for compatibility, not used).')
    parser.add_argument('--transfer_strategy', type=str, choices=['a', 'b', 'c'], default='c', 
                      help='Weight transfer strategy: a = only message-passing layers, b = message-passing + 2 FC layers, c = all layers')
    # For compatibility with chemprop scripts
    parser.add_argument('--ignore_columns', type=str, default=None, help='Columns to ignore (not used).')
    parser.add_argument('--features_path', type=str, default=None, help='Path to additional features (not used).')
    parser.add_argument('--atom_descriptors_path', type=str, default=None, help='Path to atom descriptors (not used).')
    parser.add_argument('--bond_features_path', type=str, default=None, help='Path to bond features (not used).')
    args = parser.parse_args()

    # Set polymer mode if requested
    if args.polymer:
        set_polymer(True)

    # Set random seeds for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Configure logging to file and console
    log_file = os.path.join(args.save_dir, 'pretraining.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    # Log the command-line arguments
    logging.info(f"Arguments: {vars(args)}")
    
    # Set second-stage learning rate if not specified
    if args.learning_rate_graph is None:
        args.learning_rate_graph = args.learning_rate

    # Load data
    df = pd.read_csv(args.data_path)
    if SMILES_COL not in df.columns:
        logging.error(f"Column '{SMILES_COL}' not found in data file.")
        return
        
    smiles_list = df[SMILES_COL].astype(str).tolist()
    total_data = len(smiles_list)
    
    # Handle data subsetting with more flexible format handling
    if args.pretrain_folds_file is not None:
        with open(args.pretrain_folds_file, "rb") as f:
            pretrain_data = pickle.load(f)
        
        # Check the format of the loaded data and handle appropriately
        if isinstance(pretrain_data, list):
            if all(isinstance(i, int) for i in pretrain_data):
                # Format 1: List of integer indices
                logging.info(f"✅ Detected index list format with {len(pretrain_data)} indices")
                pretrain_indices = pretrain_data
                smiles_list = [smiles_list[i] for i in pretrain_indices]
            elif len(pretrain_data) == total_data and all(x is None or x in [0, 1] for x in pretrain_data):
                # Format 2: List of 0/1 folds (0 = pretrain, 1 = unused)
                logging.info(f"✅ Detected folds format with {pretrain_data.count(0)} pretrain samples")
                pretrain_indices = [i for i, fold in enumerate(pretrain_data) if fold == 0]
                smiles_list = [smiles_list[i] for i in pretrain_indices]
            else:
                raise ValueError("Unrecognized format in pretrain_folds_file. Expected either a list of integer indices or a list of 0/1 folds.")
        else:
            raise ValueError("Expected pretrain_folds_file to contain a list.")
        
        logging.info(f"✅ Using {len(smiles_list)} samples from pretrain_folds_file for SSL pretraining.")
        logging.info(f"   This represents {len(smiles_list)/total_data:.1%} of the total dataset.")

    elif args.pretrain_frac < 1.0:
        subset_size = int(total_data * args.pretrain_frac)
        subset_size = max(subset_size, 1)
        smiles_list = random.sample(smiles_list, subset_size)
        logging.info(f"Subsampling dataset to {subset_size} entries out of {total_data}.")
    else:
        logging.info(f"Using full dataset of {total_data} entries.")

    # Build polymer graphs
    graphs = []
    for smi in smiles_list:
        if not isinstance(smi, str):
            continue
        graph = build_polymer_graph(smi)
        if graph is None:
            continue
        graphs.append(graph)
    logging.info(f"Built graph structures for {len(graphs)} polymers.")
    if len(graphs) == 0:
        logging.error("No valid polymer graphs could be constructed. Exiting.")
        return
        
    # Split data into train/val
    random.shuffle(graphs)
    val_count = int(len(graphs) * args.val_frac)
    val_graphs = graphs[:val_count]
    train_graphs = graphs[val_count:]
    logging.info(f"Training on {len(train_graphs)} samples, validating on {len(val_graphs)} samples.")
    
    # Create datasets and dataloaders
    train_dataset = PolymerDataset(train_graphs)
    val_dataset = PolymerDataset(val_graphs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)
    
    # Get feature dimensions
    atom_feat_dim = len(train_graphs[0].atom_features[0])
    bond_feat_dim = len(train_graphs[0].edge_features[0]) if train_graphs[0].n_edges > 0 else 0
    
    # Initialize model
    model = SSLPretrainModel(atom_feat_dim, bond_feat_dim, args.hidden_size, args.depth, args.dropout)
    if args.use_enhanced_ssl:
        model = improve_model_initialization(model)
    
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    model.to(device)
    
    # Log model statistics
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"🧠 Model has {total_params:,} parameters. Using device: {device}")
    
    # Save training configuration
    config = {
        'args': vars(args),
        'atom_feat_dim': atom_feat_dim,
        'bond_feat_dim': bond_feat_dim,
        'timestamp': datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    }
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Set epochs for each stage if not explicitly specified
    if args.epochs_node_edge is None:
        args.epochs_node_edge = args.epochs
    if args.epochs_graph is None:
        args.epochs_graph = args.epochs
        
    # STAGE 1: Node- and Edge-Level Pretraining
    logging.info("=" * 80)
    logging.info("🔍 STAGE 1: Node- and Edge-Level Pretraining")
    logging.info("=" * 80)
    
    # Use proper weight decay and learning rate as specified
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=args.learning_rate, 
                                weight_decay=args.weight_decay,
                                betas=(0.9, 0.999))
    # Ensure optimizer has correct initial learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.learning_rate
        logging.info(f"Initial learning rate set to: {param_group['lr']}")
    
    # Use both epoch-level LR scheduler and warmup
    if args.use_enhanced_ssl:
        scheduler = create_improved_scheduler(optimizer, args.epochs_node_edge, warmup_epochs=5)
    else:
        warmup_epochs = min(5, args.epochs_node_edge // 10)  # Warmup for 10% of epochs or 5, whichever is smaller
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=args.scheduler_patience, factor=0.5, verbose=True
        )
    
    # Create masking task
    if args.use_enhanced_ssl:
        node_edge_task = ImprovedNodeEdgeMaskingTask(model, 
                                                  mask_ratio=args.mask_ratio, 
                                                  min_mask=args.min_mask, 
                                                  edge_weight=1.5)
    else:
        node_edge_task = NodeEdgeMaskingTask(model, mask_ratio=args.mask_ratio, min_mask=args.min_mask)
    
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    lr_no_improve_epochs = 0
    node_edge_train_history = []
    node_edge_val_history = []
    best_stage1_model_path = os.path.join(args.save_dir, "best_stage1_model.pt")
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Log starting conditions    
    logging.info(f"Starting node/edge pretraining with {len(train_graphs)} graphs")
    logging.info(f"Masking ratio: {args.mask_ratio}, min masks: {args.min_mask}")
    logging.info(f"Initial learning rate: {initial_lr}")
    
    for epoch in range(1, args.epochs_node_edge + 1):
        # Define warmup epochs for both standard and enhanced methods
        warmup_epochs = min(5, args.epochs_node_edge // 10) if not args.use_enhanced_ssl else 5
        
        # Apply warmup scheduling if in warmup phase
        if epoch <= warmup_epochs:
            # Linear warmup
            lr_scale = min(1.0, epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = initial_lr * lr_scale
            logging.info(f"Warmup LR: {optimizer.param_groups[0]['lr']:.6e}")
        
        model.train()
        start_time = time.time()
        train_loss, train_task_losses = run_pretraining_epoch(
            node_edge_task, train_loader, device, optimizer, train=True
        )
        train_time = time.time() - start_time
        
        model.eval()
        start_time = time.time()
        val_loss, val_task_losses = run_pretraining_epoch(
            node_edge_task, val_loader, device, train=False
        )
        val_time = time.time() - start_time
        
        # Update learning rate scheduler (after warmup)
        old_lr = optimizer.param_groups[0]['lr']
        if epoch > warmup_epochs:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()  # Don't pass epoch parameter to step()
        new_lr = optimizer.param_groups[0]['lr']  # Make sure this comes AFTER the scheduler.step()
        
        # Log learning rate changes
        if new_lr < old_lr:
            logging.info(f"🔻 Learning rate reduced: {old_lr:.6e} → {new_lr:.6e}")
            lr_no_improve_epochs = 0
        else:
            lr_no_improve_epochs += 1
        
        # Log results
        logging.info(
            f"Epoch {epoch}/{args.epochs_node_edge} | "
            f"Train Loss: {train_loss:.4f} (node: {train_task_losses.get('loss_node', 0):.4f}, "
            f"edge: {train_task_losses.get('loss_edge', 0):.4f}) | "
            f"Val Loss: {val_loss:.4f} (node: {val_task_losses.get('loss_node', 0):.4f}, "
            f"edge: {val_task_losses.get('loss_edge', 0):.4f}) | "
            f"LR: {new_lr:.6e} | "
            f"Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s"
        )
        
        # Record history
        node_edge_train_history.append({
            'epoch': epoch,
            'loss': train_loss,
            'lr': new_lr,
            **train_task_losses
        })
        node_edge_val_history.append({
            'epoch': epoch,
            'loss': val_loss,
            'lr': new_lr,
            **val_task_losses
        })
        
        # Save loss to CSV
        loss_csv_path = os.path.join(args.save_dir, 'stage1_losses.csv')
        if epoch == 1:
            # Create new file with header
            with open(loss_csv_path, 'w') as f:
                f.write('epoch,train_loss,val_loss,train_loss_node,train_loss_edge,val_loss_node,val_loss_edge,learning_rate\n')
        
        # Append current epoch losses
        with open(loss_csv_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_task_losses.get('loss_node', 0):.6f},"
                    f"{train_task_losses.get('loss_edge', 0):.6f},{val_task_losses.get('loss_node', 0):.6f},"
                    f"{val_task_losses.get('loss_edge', 0):.6f},{new_lr:.8e}\n")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, best_stage1_model_path)
            logging.info(f"✅ Saved best stage 1 model to {best_stage1_model_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"🕰️ Early stopping patience: {epochs_no_improve}/{args.early_stop_patience}")
            
        # Early stopping
        if epochs_no_improve >= args.early_stop_patience:
            logging.info(f"⏹️ Early stopping triggered after {epochs_no_improve} epochs without improvement")
            break
    
    # Save training history
    history = {
        'node_edge_train': node_edge_train_history,
        'node_edge_val': node_edge_val_history
    }
    with open(os.path.join(args.save_dir, 'stage1_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Load best model from Stage 1
    logging.info(f"Loading best stage 1 model from epoch {best_epoch}...")
    checkpoint = torch.load(best_stage1_model_path)
    if args.use_enhanced_ssl:
        model = improved_weight_transfer(model, checkpoint, args.transfer_strategy)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # ✅ UNFREEZE MODEL FOR STRATEGY C
    if args.transfer_strategy == 'c':
        for name, param in model.named_parameters():
            if not param.requires_grad:
                param.requires_grad = True
                logging.debug(f"🔓 Unfroze parameter: {name}")
        
    # STAGE 2: Graph-Level Pretraining
    logging.info("\n" + "=" * 80)
    logging.info("🔍 STAGE 2: Graph-Level Pretraining")
    logging.info("=" * 80)
    
    # Save first stage model for potential direct use
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model.state_dict(),
        'config': config,
        'stage': 'node_edge'
    }, os.path.join(args.save_dir, "stage1_final_model.pt"))
    
    # Initialize new optimizer for stage 2 with possibly different learning rate
    optimizer = torch.optim.AdamW(model.parameters(), 
                                 lr=args.learning_rate_graph,
                                 weight_decay=args.weight_decay,
                                 betas=(0.9, 0.999))
    # Ensure optimizer has correct initial learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.learning_rate
        logging.info(f"Initial learning rate set to: {param_group['lr']}")
        
    # Create scheduler
    if args.use_enhanced_ssl:
        scheduler = create_improved_scheduler(optimizer, args.epochs_graph, warmup_epochs=5)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=args.scheduler_patience, factor=0.5, verbose=True
        )
      
    # Configure proper graph loss weight (important parameter from paper)
    if args.use_enhanced_ssl:
        graph_task = ImprovedGraphLevelTask(model, args.graph_loss_weight)
    else:
        graph_task = GraphLevelTask(model, args.graph_loss_weight)
    
    # Reset tracking variables for stage 2
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0
    lr_no_improve_epochs = 0
    graph_train_history = []
    graph_val_history = []
    best_stage2_model_path = os.path.join(args.save_dir, "best_stage2_model.pt")
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Warmup for graph task
    warmup_epochs = min(5, args.epochs_graph // 10)
    
    # Log graph-task specific info
    logging.info(f"Graph loss weight: {args.graph_loss_weight}")
    logging.info(f"Starting learning rate: {initial_lr}")
    
    for epoch in range(1, args.epochs_graph + 1):
        # Apply warmup scheduling if in warmup phase
        if epoch <= warmup_epochs:
            # Linear warmup
            lr_scale = min(1.0, epoch / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = initial_lr * lr_scale
            logging.info(f"Warmup LR: {optimizer.param_groups[0]['lr']:.6e}")
        
        model.train()
        start_time = time.time()
        train_loss, train_task_losses = run_pretraining_epoch(
            graph_task, train_loader, device, optimizer, train=True
        )
        train_time = time.time() - start_time
        
        model.eval()
        start_time = time.time()
        val_loss, val_task_losses = run_pretraining_epoch(
            graph_task, val_loader, device, train=False
        )
        val_time = time.time() - start_time
        
        # Update learning rate scheduler (after warmup)
        old_lr = optimizer.param_groups[0]['lr']
        if epoch > warmup_epochs:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()  # Don't pass epoch parameter to step()
        new_lr = optimizer.param_groups[0]['lr']  # Make sure this comes AFTER the scheduler.step()
        
        # Log learning rate changes
        if new_lr < old_lr:
            logging.info(f"🔻 Learning rate reduced: {old_lr:.6e} → {new_lr:.6e}")
            lr_no_improve_epochs = 0
        else:
            lr_no_improve_epochs += 1
        
        # Log results
        logging.info(
            f"Epoch {epoch}/{args.epochs_graph} | "
            f"Train Loss: {train_loss:.4f} (graph: {train_task_losses.get('loss_graph', 0):.4f}) | "
            f"Val Loss: {val_loss:.4f} (graph: {val_task_losses.get('loss_graph', 0):.4f}) | "
            f"LR: {new_lr:.6e} | "
            f"Train Time: {train_time:.2f}s, Val Time: {val_time:.2f}s"
        )
        
        # Record history
        graph_train_history.append({
            'epoch': epoch,
            'loss': train_loss,
            'lr': new_lr,
            **train_task_losses
        })
        graph_val_history.append({
            'epoch': epoch,
            'loss': val_loss,
            'lr': new_lr,
            **val_task_losses
        })
        
        # Save loss to CSV
        loss_csv_path = os.path.join(args.save_dir, 'stage2_losses.csv')
        if epoch == 1:
            # Create new file with header
            with open(loss_csv_path, 'w') as f:
                f.write('epoch,train_loss,val_loss,train_loss_graph,val_loss_graph,learning_rate\n')
        
        # Append current epoch losses
        with open(loss_csv_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                    f"{train_task_losses.get('loss_graph', 0):.6f},{val_task_losses.get('loss_graph', 0):.6f},{new_lr:.8e}\n")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, best_stage2_model_path)
            logging.info(f"✅ Saved best stage 2 model to {best_stage2_model_path}")
            
            # Save graph embeddings
            if args.save_graph_embeddings:
                all_graph_embeddings = []
                all_smiles = []
                
                model.eval()
                with torch.no_grad():
                    for batch in val_loader:
                        result = graph_task.eval_step(batch, device)
                        all_graph_embeddings.append(result['graph_embeds'].cpu())
                        all_smiles.extend(batch['smiles'])
                
                embeddings = torch.cat(all_graph_embeddings, dim=0).numpy()
                
                # Save raw embeddings
                if args.graph_embeddings_path:
                    np.save(args.graph_embeddings_path, embeddings)
                else:
                    np.save(os.path.join(args.save_dir, 'graph_embeddings.npy'), embeddings)
                
                # Save as CSV with SMILES
                df_embeddings = pd.DataFrame(embeddings)
                df_embeddings.columns = [f'emb_{i}' for i in range(embeddings.shape[1])]
                df_embeddings['smiles'] = all_smiles
                
                embeddings_csv_path = os.path.join(args.save_dir, 'graph_embeddings.csv')
                df_embeddings.to_csv(embeddings_csv_path, index=False)
                logging.info(f"📊 Saved graph embeddings for {len(df_embeddings)} molecules")
        else:
            epochs_no_improve += 1
            logging.info(f"🕰️ Early stopping patience: {epochs_no_improve}/{args.early_stop_patience}")
            
        # Early stopping
        if epochs_no_improve >= args.early_stop_patience:
            logging.info(f"⏹️ Early stopping triggered after {epochs_no_improve} epochs without improvement")
            break
    
    # Update history with stage 2 results
    history['graph_train'] = graph_train_history
    history['graph_val'] = graph_val_history
    with open(os.path.join(args.save_dir, 'full_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Load best model from Stage 2
    logging.info(f"Loading best stage 2 model from epoch {best_epoch}...")
    checkpoint = torch.load(best_stage2_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save the final model with the appropriate transfer strategy
    # Depending on transfer_strategy, we'll save different parts of the model
    final_model_state = {}
    transfer_strategy = args.transfer_strategy
    
    if args.use_enhanced_ssl:
        model = improved_weight_transfer(model, {'model_state_dict': model.state_dict()}, transfer_strategy)
        final_model_state = model.state_dict()
        logging.info(f"📦 Saved model with enhanced transfer strategy {transfer_strategy}")
    else:
        if transfer_strategy == 'a':
            # Strategy A: only message-passing layers
            final_model_state = {
                k: v for k, v in model.state_dict().items() 
                if any(name in k for name in ['W_initial', 'W_message', 'W_node'])
            }
            logging.info("📦 Saved model with transfer strategy A: only message-passing layers")
        
        elif transfer_strategy == 'b':
            # Strategy B: message-passing + first two FC layers
            final_model_state = {
                k: v for k, v in model.state_dict().items() 
                if (any(name in k for name in ['W_initial', 'W_message', 'W_node']) or
                    k.startswith('graph_head.0') or k.startswith('graph_head.2'))
            }
            logging.info("📦 Saved model with transfer strategy B: message-passing + two FC layers")
        
        elif transfer_strategy == 'c':
            # Strategy C: all layers
            final_model_state = model.state_dict()
            logging.info("📦 Saved model with transfer strategy C: all layers")
    
    # Save the final model
    final_model_path = os.path.join(args.save_dir, f"final_model_strategy_{transfer_strategy}.pt")
    torch.save({
        'model_state_dict': final_model_state,
        'config': config,
        'transfer_strategy': transfer_strategy,
        'stage': 'combined',
        'args': vars(args)
    }, final_model_path)
    logging.info(f"✅ Saved final model to {final_model_path}")
    
    # Also save a copy as model.pt for compatibility with chemprop
    standard_model_path = os.path.join(args.save_dir, "model.pt")
    torch.save({
        'model_state_dict': final_model_state,  # Use filtered state dict for compatibility
        'config': config,
        'transfer_strategy': transfer_strategy,
        'stage': 'combined',
        'args': vars(args)
    }, standard_model_path)
    logging.info(f"✅ Saved standard model to {standard_model_path}")
    
    # Save the full model state for reference/debugging
    full_model_path = os.path.join(args.save_dir, "full_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'transfer_strategy': transfer_strategy,
        'stage': 'combined',
        'args': vars(args)
    }, full_model_path)
    logging.info(f"✅ Saved full model to {full_model_path}")
    
    # Create visualization of learning curves
    try:
        import matplotlib.pyplot as plt
        
        # Plot Stage 1 (Node-Edge) Learning Curves
        if os.path.exists(os.path.join(args.save_dir, 'stage1_losses.csv')):
            stage1_df = pd.read_csv(os.path.join(args.save_dir, 'stage1_losses.csv'))
            plt.figure(figsize=(15, 10))
            
            # Total loss
            plt.subplot(2, 3, 1)
            plt.plot(stage1_df['epoch'], stage1_df['train_loss'], label='Train Loss')
            plt.plot(stage1_df['epoch'], stage1_df['val_loss'], label='Validation Loss')
            plt.title('Stage 1: Total Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Node loss
            plt.subplot(2, 3, 2)
            plt.plot(stage1_df['epoch'], stage1_df['train_loss_node'], label='Train Node Loss')
            plt.plot(stage1_df['epoch'], stage1_df['val_loss_node'], label='Validation Node Loss')
            plt.title('Stage 1: Node Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Edge loss
            plt.subplot(2, 3, 3)
            plt.plot(stage1_df['epoch'], stage1_df['train_loss_edge'], label='Train Edge Loss')
            plt.plot(stage1_df['epoch'], stage1_df['val_loss_edge'], label='Validation Edge Loss')
            plt.title('Stage 1: Edge Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Learning rate
            plt.subplot(2, 3, 4)
            plt.plot(stage1_df['epoch'], stage1_df['learning_rate'])
            plt.title('Stage 1: Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, 'stage1_learning_curves.png'))
            plt.close()
            logging.info(f"📊 Saved Stage 1 learning curves visualization")
        
        # Plot Stage 2 (Graph) Learning Curves
        if os.path.exists(os.path.join(args.save_dir, 'stage2_losses.csv')):
            stage2_df = pd.read_csv(os.path.join(args.save_dir, 'stage2_losses.csv'))
            plt.figure(figsize=(15, 6))
            
            # Total loss
            plt.subplot(1, 3, 1)
            plt.plot(stage2_df['epoch'], stage2_df['train_loss'], label='Train Loss')
            plt.plot(stage2_df['epoch'], stage2_df['val_loss'], label='Validation Loss')
            plt.title('Stage 2: Total Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Graph loss
            plt.subplot(1, 3, 2)
            plt.plot(stage2_df['epoch'], stage2_df['train_loss_graph'], label='Train Graph Loss')
            plt.plot(stage2_df['epoch'], stage2_df['val_loss_graph'], label='Validation Graph Loss')
            plt.title('Stage 2: Graph Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Learning rate
            plt.subplot(1, 3, 3)
            plt.plot(stage2_df['epoch'], stage2_df['learning_rate'])
            plt.title('Stage 2: Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.yscale('log')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, 'stage2_learning_curves.png'))
            plt.close()
            logging.info(f"📊 Saved Stage 2 learning curves visualization")
    except Exception as e:
        logging.warning(f"Could not generate learning curve visualizations: {e}")
    
    logging.info("\n" + "=" * 80)
    logging.info("✨ Self-supervised pretraining completed successfully!")
    logging.info("=" * 80)

if __name__ == "__main__":
    if Chem is None:
        logging.error("RDKit is not installed. Please install RDKit to run this script.")
    else:
        main()
