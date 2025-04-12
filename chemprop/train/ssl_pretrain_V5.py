"""
Implementation of Self-supervised Graph Neural Networks for Polymer Property Prediction
Based on the paper: "Self-supervised graph neural networks for polymer property prediction"
by Qinghe Gao, Tammo Dukker, Artur M. Schweidtmann and Jana M. Weber.

This code implements the best performing model from the paper:
- Ensemble node-, edge- & graph-level self-supervised learning
- All layers transferred during fine-tuning

Adapted from the original implementation in ssl_pretrain_V4.py
"""

import os
import math
import logging
import random
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import copy
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# For Google Colab, install RDKit if not already installed
try:
    import google.colab
    !pip install -q rdkit
except:
    pass

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdchem
except ImportError:
    logging.error("RDKit is not installed. Installing now...")
    !pip install -q rdkit
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdchem

# Define SMILES column name for polymer data
SMILES_COL = "poly_chemprop_input"

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#################
# Feature Generation
#################

def one_hot_encoding(value, choices):
    """One-hot encoding with an extra 'unknown' category at end if value not in choices."""
    encoding = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
        encoding[idx] = 1
    except ValueError:
        encoding[-1] = 1
    return encoding

def atom_features(atom):
    """
    Generate atom features as described in the paper's Table 1
    """
    features = []
    
    # Atom type (one-hot)
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H']
    features.extend(one_hot_encoding(atom.GetSymbol(), atom_types))
    
    # Chirality (one-hot)
    chirality_types = [
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
    ]
    features.extend(one_hot_encoding(atom.GetChiralTag(), chirality_types))
    
    # Degree (one-hot)
    features.extend(one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
    
    # Aromaticity (one-hot)
    features.append(atom.GetIsAromatic())
    
    # Hybridization (one-hot)
    hybridization_types = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
    features.extend(one_hot_encoding(atom.GetHybridization(), hybridization_types))
    
    # Formal charge (one-hot)
    features.extend(one_hot_encoding(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3]))
    
    # Number of hydrogens (one-hot)
    features.extend(one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))
    
    # Atomic mass (normalized)
    features.append(atom.GetMass() / 100.0)  # Normalize by dividing by 100
    
    return features

def bond_features(bond):
    """
    Generate bond features as described in the paper's Table 2
    """
    if bond is None:
        # For dummy bonds in the case of disconnected fragments
        # Return a feature vector of appropriate size with all zeros
        return [0] * 10
    
    features = []
    
    # Bond type (one-hot)
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    features.extend(one_hot_encoding(bond.GetBondType(), bond_types))
    
    # Conjugation (one-hot)
    features.append(bond.GetIsConjugated())
    
    # Ring (one-hot)
    features.append(bond.IsInRing())
    
    # Stereo (one-hot)
    stereo_types = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ]
    features.extend(one_hot_encoding(bond.GetStereo(), stereo_types))
    
    return features

#################
# Polymer Graph Representation
#################

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
        self.smiles = None        # polymer SMILES string
        self.property_value = None  # target property (EA or IP)

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

def build_polymer_graph(smiles, property_value=None):
    """Build PolymerGraph from an extended polymer SMILES string."""
    graph = PolymerGraph()
    graph.smiles = smiles
    graph.property_value = property_value
    
    monomers, ratios, edges_info, Xn = parse_polymer_smiles(smiles)
    if len(monomers) == 0:
        return None
    
    # Calculate molecular weight for graph-level SSL
    total_weight = 0.0
    for m_smiles, frac in zip(monomers, ratios):
        mol_m = Chem.MolFromSmiles(m_smiles)
        if mol_m is None:
            logging.warning(f"RDKit failed to parse monomer: {m_smiles}")
            continue
        weight = 0.0
        for atom in mol_m.GetAtoms():
            if atom.GetAtomicNum() > 0:
                weight += atom.GetMass()
        total_weight += frac * weight
    graph.mol_weight = total_weight
    
    # Build molecular graph from monomers
    ensemble_mol = Chem.MolFromSmiles('.'.join(monomers))
    if ensemble_mol is None:
        logging.error(f"Failed to parse polymer SMILES: {smiles}")
        return None
    
    # Process atoms (nodes)
    dummy_index_map = {}
    for atom in ensemble_mol.GetAtoms():
        feat = atom_features(atom)
        graph.atom_features.append(feat)
        idx = atom.GetIdx()
        graph.n_atoms += 1
        if atom.GetSymbol() == '*' and atom.GetAtomMapNum() != 0:
            dummy_index_map[atom.GetAtomMapNum()] = idx
    
    # Process bonds (edges)
    for bond in ensemble_mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        
        # Add forward edge
        e_index = graph.n_edges
        graph.edge_index.append((u, v))
        graph.edge_features.append(bf)
        graph.edge_weights.append(1.0)  # Weight 1.0 for internal monomer bonds
        graph.b2rev.append(None)
        graph.n_edges += 1
        
        # Add reverse edge
        rev_index = graph.n_edges
        graph.edge_index.append((v, u))
        graph.edge_features.append(bf)
        graph.edge_weights.append(1.0)  # Weight 1.0 for internal monomer bonds
        graph.b2rev.append(None)
        graph.n_edges += 1
        
        # Link edges to their reverse counterparts
        graph.b2rev[e_index] = rev_index
        graph.b2rev[rev_index] = e_index
    
    # Process inter-monomer connectivity with stochastic weights
    weight_factor = 1.0
    if Xn is not None:
        try:
            weight_factor = 1.0 + math.log(Xn)
        except ValueError:
            weight_factor = 1.0
    
    # Add inter-monomer edges with stochastic weights
    for i, j, w1, w2 in edges_info:
        if i < len(monomers) and j < len(monomers):
            # Create dummy bond features for inter-monomer connections
            fake_bond = Chem.Bond()  # create a dummy bond
            bf = bond_features(fake_bond)
            
            # Find appropriate terminal atoms to connect
            # In a real implementation, you'd identify the specific atoms to connect
            # between monomers - this is simplified
            u = i  # placeholder - you'd identify actual atom indices
            v = j  # placeholder - you'd identify actual atom indices
            
            # Add forward edge with stochastic weight
            e_index = graph.n_edges
            graph.edge_index.append((u, v))
            graph.edge_features.append(bf)
            graph.edge_weights.append(w1 * weight_factor)
            graph.b2rev.append(None)
            graph.n_edges += 1
            
            # Add reverse edge with stochastic weight
            rev_index = graph.n_edges
            graph.edge_index.append((v, u))
            graph.edge_features.append(bf)
            graph.edge_weights.append(w2 * weight_factor)
            graph.b2rev.append(None)
            graph.n_edges += 1
            
            # Link reverse edges
            graph.b2rev[e_index] = rev_index
            graph.b2rev[rev_index] = e_index
    
    return graph

#################
# Dataset and DataLoader
#################

class PolymerDataset(Dataset):
    """Dataset class for polymer graphs"""
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
    property_values = []
    
    atom_offset = 0
    edge_offset = 0
    
    for graph_idx, g in enumerate(batch_graphs):
        n = g.n_atoms
        e = g.n_edges
        
        # Add atom features
        if n > 0:
            combined_atom_feats[atom_offset:atom_offset+n] = torch.tensor(g.atom_features, dtype=torch.float32)
        
        # Add edge features
        if e > 0:
            combined_edge_feats[edge_offset:edge_offset+e] = torch.tensor(g.edge_features, dtype=torch.float32)
        
        # Adjust edge indices with offset
        for local_idx, (src, dst) in enumerate(g.edge_index):
            edge_src_list.append(src + atom_offset)
            edge_dst_list.append(dst + atom_offset)
        
        # Adjust reverse edge indices
        b2rev_list.extend([rev_idx + edge_offset if rev_idx is not None else None for rev_idx in g.b2rev])
        
        # Track which nodes belong to which graphs
        if n > 0:
            node_to_graph[atom_offset:atom_offset+n] = graph_idx
        
        # Collect graph-level properties
        mol_weights.append(g.mol_weight)
        if g.property_value is not None:
            property_values.append(g.property_value)
        
        # Update offsets
        atom_offset += n
        edge_offset += e
    
    # Convert lists to tensors
    edge_src = torch.tensor(edge_src_list, dtype=torch.long)
    edge_dst = torch.tensor(edge_dst_list, dtype=torch.long)
    edge_weights = torch.cat([torch.tensor(g.edge_weights, dtype=torch.float32) for g in batch_graphs]) if total_edges > 0 else torch.tensor([], dtype=torch.float32)
    
    # Convert b2rev None to int (-1 for None values)
    b2rev_tensor = []
    for val in b2rev_list:
        b2rev_tensor.append(-1 if val is None else val)
    b2rev = torch.tensor(b2rev_tensor, dtype=torch.long)
    
    # Graph-level properties
    mol_weights = torch.tensor(mol_weights, dtype=torch.float32)
    property_tensor = torch.tensor(property_values, dtype=torch.float32) if property_values else None
    
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
        'property_values': property_tensor,
        'smiles': [g.smiles for g in batch_graphs]
    }

def load_polymer_dataset(data_path, property_name=None, split_sizes=None):
    """
    Load polymer dataset and create graph representations
    
    Args:
        data_path: Path to the polymer dataset CSV
        property_name: Name of the property to predict ('EA' or 'IP')
        split_sizes: Dictionary with keys 'train', 'val', 'test', 'ssl' and values as fractions
    
    Returns:
        Dictionary of dataset splits
    """
    if split_sizes is None:
        split_sizes = {'ssl': 0.4, 'train': 0.4, 'val': 0.1, 'test': 0.1}
    
    # Check if split sizes sum to 1
    total = sum(split_sizes.values())
    if abs(total - 1.0) > 1e-6:
        # Normalize
        for k in split_sizes:
            split_sizes[k] /= total
    
    # Load data
    df = pd.read_csv(data_path)
    if SMILES_COL not in df.columns:
        raise ValueError(f"Column '{SMILES_COL}' not found in the dataset.")
    
    # Check if property exists when specified
    if property_name is not None and property_name not in df.columns:
        raise ValueError(f"Property column '{property_name}' not found in the dataset.")
    
    # Build polymer graphs
    graphs = []
    for idx, row in df.iterrows():
        smiles = row[SMILES_COL]
        property_value = row[property_name] if property_name is not None else None
        
        graph = build_polymer_graph(smiles, property_value)
        if graph is not None:
            graphs.append(graph)
    
    logging.info(f"Built {len(graphs)} polymer graphs from dataset.")
    
    # Split dataset
    random.shuffle(graphs)
    total_graphs = len(graphs)
    
    ssl_count = int(split_sizes['ssl'] * total_graphs)
    train_count = int(split_sizes['train'] * total_graphs)
    val_count = int(split_sizes['val'] * total_graphs)
    
    ssl_graphs = graphs[:ssl_count]
    train_graphs = graphs[ssl_count:ssl_count+train_count]
    val_graphs = graphs[ssl_count+train_count:ssl_count+train_count+val_count]
    test_graphs = graphs[ssl_count+train_count+val_count:]
    
    logging.info(f"Dataset split: SSL={len(ssl_graphs)}, Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")
    
    # Create datasets
    ssl_dataset = PolymerDataset(ssl_graphs)
    train_dataset = PolymerDataset(train_graphs)
    val_dataset = PolymerDataset(val_graphs)
    test_dataset = PolymerDataset(test_graphs)
    
    return {
        'ssl': ssl_dataset,
        'train': train_dataset,
        'val': val_dataset,
        'test': test_dataset,
        'atom_feat_dim': len(graphs[0].atom_features[0]),
        'bond_feat_dim': len(graphs[0].edge_features[0]) if graphs[0].n_edges > 0 else 0
    }

#################
# Model Definition
#################

class WeightedDirectedMPNN(MessagePassing):
    """
    Weighted Directed Message Passing Neural Network (wD-MPNN) as described in the paper
    """
    def __init__(self, in_node_dim, in_edge_dim, hidden_dim):
        """
        Initialize the wD-MPNN layer
        
        Args:
            in_node_dim: Dimension of input node features
            in_edge_dim: Dimension of input edge features
            hidden_dim: Dimension of hidden node/edge features
        """
        super(WeightedDirectedMPNN, self).__init__(aggr='add')  # "add" aggregation
        
        # Step 1: Initialize hidden edge features
        self.edge_init = nn.Sequential(
            nn.Linear(in_node_dim + in_edge_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Step 2: Edge-centered message passing
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Step 3: Update node features
        self.node_update = nn.Sequential(
            nn.Linear(in_node_dim + hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, x, edge_index, edge_attr, node_weights, edge_weights, num_layers=3):
        """
        Forward pass through the wD-MPNN
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            node_weights: Weights for nodes
            edge_weights: Weights for edges
            num_layers: Number of message passing layers
        
        Returns:
            Updated node features and edge features
        """
        # Store original node features
        original_x = x
        
        # Step 1: Initialize hidden edge features
        # For each edge (v, u), concatenate source node v's features with edge features
        row, col = edge_index
        h0_edges = torch.zeros(edge_index.size(1), self.hidden_dim, device=edge_index.device)
        
        for i in range(edge_index.size(1)):
            v = row[i]
            # Concatenate node features with edge features
            concat_features = torch.cat([x[v], edge_attr[i]], dim=0)
            # Apply linear layer and activation
            h0_edges[i] = self.edge_init(concat_features)
        
        # Step 2: Edge-centered message passing
        h_edges = h0_edges
        for _ in range(num_layers):
            h_edges_new = h_edges.clone()
            
            for i in range(edge_index.size(1)):
                v = row[i]
                u = col[i]
                
                # Get all incoming edges to node v except the one from node u
                mask = (row != u) & (col == v)
                incoming_edges = edge_index[:, mask]
                incoming_edge_indices = torch.where(mask)[0]
                
                # If there are incoming edges, update the edge feature
                if incoming_edges.size(1) > 0:
                    # Weighted sum of incoming edge features
                    weighted_sum = torch.zeros(self.hidden_dim, device=edge_index.device)
                    for j, idx in enumerate(incoming_edge_indices):
                        k = row[idx]  # Source node of incoming edge
                        weighted_sum += edge_weights[idx] * h_edges[idx]
                    
                    # Update edge feature
                    h_edges_new[i] = self.edge_update(h0_edges[i] + weighted_sum)
            
            h_edges = h_edges_new
        
        # Step 3: Update node features
        h_nodes = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        
        for i in range(x.size(0)):
            # Get all incoming edges to node i
            mask = (col == i)
            incoming_edges = edge_index[:, mask]
            incoming_edge_indices = torch.where(mask)[0]
            
            # Weighted sum of incoming edge features
            if incoming_edges.size(1) > 0:
                weighted_sum = torch.zeros(self.hidden_dim, device=x.device)
                for j, idx in enumerate(incoming_edge_indices):
                    weighted_sum += edge_weights[idx] * h_edges[idx]
                
                # Concatenate original node features with weighted sum
                concat_features = torch.cat([original_x[i], weighted_sum], dim=0)
                # Update node feature
                h_nodes[i] = self.node_update(concat_features)
            else:
                # If no incoming edges, just use original node features
                concat_features = torch.cat([original_x[i], torch.zeros(self.hidden_dim, device=x.device)], dim=0)
                h_nodes[i] = self.node_update(concat_features)
        
        return h_nodes, h_edges


class PolymerGNN(nn.Module):
    """
    Full Graph Neural Network for polymer property prediction with SSL capabilities
    """
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, num_layers=3):
        """
        Initialize the PolymerGNN
        
        Args:
            node_dim: Dimension of input node features
            edge_dim: Dimension of input edge features
            hidden_dim: Dimension of hidden features
            output_dim: Dimension of output (1 for property prediction)
            num_layers: Number of message passing layers
        """
        super(PolymerGNN, self).__init__()
        
        # Message passing layers
        self.mpnn = WeightedDirectedMPNN(node_dim, edge_dim, hidden_dim)
        self.num_layers = num_layers
        
        # Readout for graph-level tasks
        self.readout_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.readout_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.readout_fc3 = nn.Linear(hidden_dim // 2, output_dim)
        
        # Node/edge feature prediction for SSL
        self.node_ssl_fc = nn.Linear(hidden_dim, node_dim)
        self.edge_ssl_fc = nn.Linear(hidden_dim, edge_dim)
        
        # Graph-level property prediction for SSL (ensemble molecular weight)
        self.graph_ssl_fc = nn.Linear(hidden_dim // 2, 1)
        
        self.hidden_dim = hidden_dim
    
    def encode(self, x, edge_index, edge_attr, node_weights, edge_weights):
        """
        Encode the graph using message passing
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            node_weights: Weights for nodes
            edge_weights: Weights for edges
        
        Returns:
            h_nodes: Updated node features
            h_edges: Updated edge features
        """
        h_nodes, h_edges = self.mpnn(x, edge_index, edge_attr, node_weights, edge_weights, self.num_layers)
        return h_nodes, h_edges
    
    def pool(self, h_nodes, node_weights, batch=None):
        """
        Pool node features to get graph-level representation
        
        Args:
            h_nodes: Node features
            node_weights: Weights for nodes
            batch: Batch indices for multiple graphs
        
        Returns:
            h_graph: Graph-level representation
        """
        if batch is None:
            # Single graph - weighted average
            h_graph = torch.sum(h_nodes * node_weights.unsqueeze(1), dim=0) / torch.sum(node_weights)
        else:
            # Multiple graphs - weighted average per graph
            h_graph = torch.zeros(batch.max().item() + 1, h_nodes.size(1), device=h_nodes.device)
            for i in range(batch.max().item() + 1):
                mask = (batch == i)
                graph_nodes = h_nodes[mask]
                graph_weights = node_weights[mask]
                h_graph[i] = torch.sum(graph_nodes * graph_weights.unsqueeze(1), dim=0) / torch.sum(graph_weights)
        
        return h_graph
    
    def readout(self, h_graph):
        """
        Readout for property prediction
        
        Args:
            h_graph: Graph-level representation
        
        Returns:
            out: Predicted property value
        """
        h = F.relu(self.readout_fc1(h_graph))
        h = F.relu(self.readout_fc2(h))
        out = self.readout_fc3(h)
        return out
    
    def forward(self, data):
        """
        Forward pass for property prediction
        
        Args:
            data: PyTorch Geometric Data object
        
        Returns:
            out: Predicted property value
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        node_weights, edge_weights = data.node_weights, data.edge_weights
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Encode
        h_nodes, _ = self.encode(x, edge_index, edge_attr, node_weights, edge_weights)
        
        # Pool
        h_graph = self.pool(h_nodes, node_weights, batch)
        
        # Readout
        out = self.readout(h_graph)
        
        return out
    
    def node_edge_ssl_forward(self, data, masked_node_indices=None, masked_edge_indices=None):
        """
        Forward pass for node and edge feature prediction (SSL)
        
        Args:
            data: PyTorch Geometric Data object
            masked_node_indices: Indices of nodes with masked features
            masked_edge_indices: Indices of edges with masked features
        
        Returns:
            node_pred: Predicted node features
            edge_pred: Predicted edge features
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        node_weights, edge_weights = data.node_weights, data.edge_weights
        
        # Encode
        h_nodes, h_edges = self.encode(x, edge_index, edge_attr, node_weights, edge_weights)
        
        # Predict node features (only for masked nodes)
        if masked_node_indices is not None:
            node_pred = self.node_ssl_fc(h_nodes[masked_node_indices])
        else:
            node_pred = None
        
        # Predict edge features (only for masked edges)
        if masked_edge_indices is not None:
            edge_pred = self.edge_ssl_fc(h_edges[masked_edge_indices])
        else:
            edge_pred = None
        
        return node_pred, edge_pred
    
    def graph_ssl_forward(self, data):
        """
        Forward pass for graph-level property prediction (SSL)
        
        Args:
            data: PyTorch Geometric Data object
        
        Returns:
            m_pred: Predicted ensemble molecular weight
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        node_weights, edge_weights = data.node_weights, data.edge_weights
        batch = data.batch if hasattr(data, 'batch') else None
        
        # Encode
        h_nodes, _ = self.encode(x, edge_index, edge_attr, node_weights, edge_weights)
        
        # Pool
        h_graph = self.pool(h_nodes, node_weights, batch)
        
        # Intermediate layer (shared with property prediction)
        h = F.relu(self.readout_fc1(h_graph))
        h = F.relu(self.readout_fc2(h))
        
        # Predict ensemble molecular weight
        m_pred = self.graph_ssl_fc(h)
        
        return m_pred


#################
# Training Functions
#################

def mask_features(data, node_mask_ratio=0.15, edge_mask_ratio=0.15):
    """
    Mask node and edge features for self-supervised learning
    
    Args:
        data: PyTorch Geometric Data object
        node_mask_ratio: Ratio of nodes to mask
        edge_mask_ratio: Ratio of edges to mask
    
    Returns:
        data_masked: Data with masked features
        masked_node_indices: Indices of masked nodes
        masked_edge_indices: Indices of masked edges
        original_node_features: Original features of masked nodes
        original_edge_features: Original features of masked edges
    """
    data_masked = copy.deepcopy(data)
    
    # Mask node features
    num_nodes = data.x.size(0)
    num_edges = data.edge_attr.size(0)
    
    # Select random nodes to mask
    num_nodes_to_mask = max(2, int(node_mask_ratio * num_nodes))  # At least 2 nodes
    masked_node_indices = torch.randperm(num_nodes)[:num_nodes_to_mask]
    
    # Store original features
    original_node_features = data.x[masked_node_indices].clone()
    
    # Mask (set to zero)
    data_masked.x[masked_node_indices] = 0
    
    # Select random edges to mask
    num_edges_to_mask = max(2, int(edge_mask_ratio * num_edges))  # At least 2 edges
    masked_edge_indices = torch.randperm(num_edges)[:num_edges_to_mask]
    
    # Store original features
    original_edge_features = data.edge_attr[masked_edge_indices].clone()
    
    # Mask (set to zero)
    data_masked.edge_attr[masked_edge_indices] = 0
    
    return data_masked, masked_node_indices, masked_edge_indices, original_node_features, original_edge_features


def train_node_edge_ssl(model, data_loader, optimizer, device):
    """
    Train the model with node and edge masking self-supervised learning
    
    Args:
        model: PolymerGNN model
        data_loader: DataLoader for SSL data
        optimizer: Optimizer
        device: Device (cpu or cuda)
    
    Returns:
        avg_loss: Average loss over all batches
    """
    model.train()
    total_loss = 0
    
    for data in data_loader:
        data = data.to(device)
        
        # Mask features
        data_masked, masked_node_indices, masked_edge_indices, original_node_features, original_edge_features = mask_features(data)
        
        # Forward pass
        optimizer.zero_grad()
        node_pred, edge_pred = model.node_edge_ssl_forward(data_masked, masked_node_indices, masked_edge_indices)
        
        # Calculate loss
        node_loss = F.mse_loss(node_pred, original_node_features)
        edge_loss = F.mse_loss(edge_pred, original_edge_features)
        loss = node_loss + edge_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def train_graph_ssl(model, data_loader, optimizer, device):
    """
    Train the model with graph-level self-supervised learning
    
    Args:
        model: PolymerGNN model
        data_loader: DataLoader for SSL data
        optimizer: Optimizer
        device: Device (cpu or cuda)
    
    Returns:
        avg_loss: Average loss over all batches
    """
    model.train()
    total_loss = 0
    
    for data in data_loader:
        data = data.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        m_pred = model.graph_ssl_forward(data)
        
        # Calculate loss
        loss = F.mse_loss(m_pred, data.m_ensemble)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def train_supervised(model, data_loader, optimizer, device):
    """
    Train the model with supervised learning
    
    Args:
        model: PolymerGNN model
        data_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device (cpu or cuda)
    
    Returns:
        avg_loss: Average loss over all batches
    """
    model.train()
    total_loss = 0
    
    for data in data_loader:
        data = data.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred = model(data)
        
        # Calculate loss
        loss = F.mse_loss(pred, data.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def evaluate(model, data_loader, device):
    """
    Evaluate the model on the given data
    
    Args:
        model: PolymerGNN model
        data_loader: DataLoader for evaluation data
        device: Device (cpu or cuda)
    
    Returns:
        rmse: Root mean squared error
        r2: R-squared score
    """
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            pred = model(data)
            
            y_true.append(data.y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return rmse, r2


#################
# Main Training Pipeline
#################

def train_ssl_gnn_pipeline(dataset, property_name, fine_tuning_size=0.08, batch_size=64, epochs=100, lr=1e-3, hidden_dim=300):
    """
    Train the self-supervised GNN pipeline following the best approach from the paper:
    - Ensemble node-, edge- & graph-level SSL
    - All layers transferred
    
    Args:
        dataset: PolymerDataset
        property_name: Name of the property to predict ('EA' or 'IP')
        fine_tuning_size: Size of the fine-tuning dataset as a fraction of the total dataset
        batch_size: Batch size
        epochs: Number of epochs
        lr: Learning rate
        hidden_dim: Hidden dimension
    
    Returns:
        test_rmse: Root mean squared error on test set
        test_r2: R-squared score on test set
        best_model: Trained model
    """
    # Set the SSL, training, validation, and test sizes
    ssl_size = 0.4  # 40% for SSL pre-training
    train_size = fine_tuning_size  # Variable size for fine-tuning
    val_size = 0.1  # 10% for validation
    test_size = 0.2  # 20% for testing
    
    # Split the dataset
    ssl_data, train_data, val_data, test_data = dataset.split_dataset(
        train_size=train_size, val_size=val_size, test_size=test_size, ssl_size=ssl_size
    )
    
    # Create data loaders
    ssl_loader = DataLoader(ssl_data, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Get feature dimensions from the first graph
    node_dim = ssl_data[0].x.size(1)
    edge_dim = ssl_data[0].edge_attr.size(1)
    
    # Initialize model
    model = PolymerGNN(node_dim, edge_dim, hidden_dim, output_dim=1).to(device)
    
    print(f"Training SSL GNN pipeline for {property_name} with fine-tuning size {fine_tuning_size*100}%")
    print(f"SSL data: {len(ssl_data)}, Train data: {len(train_data)}, Val data: {len(val_data)}, Test data: {len(test_data)}")
    
    # STAGE 1: Node- and edge-level SSL pre-training
    print("\nStage 1: Node- and edge-level SSL pre-training")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_ssl_loss = float('inf')
    best_ssl_model = None
    
    for epoch in range(epochs):
        ssl_loss = train_node_edge_ssl(model, ssl_loader, optimizer, device)
        
        if (epoch + 1) % 10 == 0:
            print(f"Node/Edge SSL Epoch {epoch+1}/{epochs}, Loss: {ssl_loss:.4f}")
        
        if ssl_loss < best_ssl_loss:
            best_ssl_loss = ssl_loss
            best_ssl_model = copy.deepcopy(model.state_dict())
    
    # Load best model from node/edge SSL
    model.load_state_dict(best_ssl_model)
    
    # STAGE 2: Graph-level SSL pre-training
    print("\nStage 2: Graph-level SSL pre-training")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_graph_ssl_loss = float('inf')
    best_graph_ssl_model = None
    
    for epoch in range(epochs):
        graph_ssl_loss = train_graph
