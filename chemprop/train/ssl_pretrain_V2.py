
# ssl_pretrain_V2.py
"""
Self-supervised pretraining script for polymer-chemprop.
It loads polymer ensemble SMILES from a CSV (with 'poly_chemprop_input' column), 
performs graph-based self-supervised learning (masking atoms/bonds and predicting their features, plus predicting polymer molecular weight).
"""
import os
import math
import logging
import random
import argparse
import numpy as np
import pandas as pd
try:
    from rdkit import Chem
    from rdkit.Chem import rdchem
except ImportError:
    Chem = None
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
    """Compute atom feature vector (based on Chemprop defaults) for a given RDKit atom."""
    # Atomic number one-hot (allow up to 100)
    atom_num = atom.GetAtomicNum()
    atom_num_feature = one_hot_encoding(atom_num, list(range(1, 101)))
    # Degree one-hot (degree of atom in graph, up to 5)
    degree_feature = one_hot_encoding(atom.GetTotalDegree(), list(range(6)))
    # Formal charge one-hot (allow -3 to +3)
    formal_charge_feature = one_hot_encoding(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3])
    # Chirality one-hot
    chiral_tag = int(atom.GetChiralTag())
    chiral_feature = one_hot_encoding(chiral_tag, [0, 1, 2, 3])
    # Number of Hs one-hot (explicit + implicit total Hs)
    num_H = int(atom.GetTotalNumHs())
    num_H_feature = one_hot_encoding(num_H, list(range(5)))
    # Hybridization one-hot
    hybrid = int(atom.GetHybridization())
    hybrid_feature = one_hot_encoding(hybrid, [0, 1, 2, 3, 4, 5])
    # Aromaticity
    aromatic_feature = [1 if atom.GetIsAromatic() else 0]
    # Atomic mass (scaled)
    mass_feature = [atom.GetMass() * 0.01]
    features = atom_num_feature + degree_feature + formal_charge_feature + chiral_feature + num_H_feature + hybrid_feature + aromatic_feature + mass_feature
    return features

def get_bond_features(bond):
    """Compute bond feature vector for an RDKit bond (Chemprop style)."""
    bt = bond.GetBondType()
    bond_type_features = [
        bt == rdchem.BondType.SINGLE,
        bt == rdchem.BondType.DOUBLE,
        bt == rdchem.BondType.TRIPLE,
        bt == rdchem.BondType.AROMATIC
    ]
    conjugated = bond.GetIsConjugated() if bt is not None else False
    in_ring = bond.IsInRing() if bt is not None else False
    conj_ring_features = [1 if conjugated else 0, 1 if in_ring else 0]
    stereo = int(bond.GetStereo())
    stereo_feature = [0]*6
    if 0 <= stereo < 6:
        stereo_feature[stereo] = 1
    features = bond_type_features + conj_ring_features + stereo_feature
    features = [1 if x else 0 for x in features]
    return features

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
        'mol_weights': mol_weights
    }

class SSLPretrainModel(nn.Module):
    def __init__(self, atom_feat_dim, bond_feat_dim, hidden_size, depth, dropout):
        super(SSLPretrainModel, self).__init__()
        self.atom_feat_dim = atom_feat_dim
        self.bond_feat_dim = bond_feat_dim
        self.hidden_size = hidden_size
        self.depth = depth
        # Initial message transformation (atom + bond -> hidden)
        self.W_initial = nn.Linear(atom_feat_dim + bond_feat_dim, hidden_size)
        # Message passing update transformation (hidden -> hidden)
        self.W_message = nn.Linear(hidden_size, hidden_size)
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
        E = hidden_edges.size(0)
        N = atom_feats.size(0)
        for t in range(self.depth):
            M = torch.zeros((N, self.hidden_size), device=hidden_edges.device)
            if E > 0:
                M.index_add_(0, edge_dst, edge_weights.unsqueeze(1) * hidden_edges)
                rev_hidden = hidden_edges[b2rev]  # hidden state of reverse edge for each edge
                rev_weight = edge_weights[b2rev].unsqueeze(1)
                msg = M[edge_src] - rev_weight * rev_hidden
            else:
                msg = torch.zeros((0, self.hidden_size), device=hidden_edges.device)
            hidden_edges = F.relu(self.W_message(msg))
        # Final node embeddings
        node_repr = torch.zeros((N, self.hidden_size), device=hidden_edges.device)
        if E > 0:
            node_repr.index_add_(0, edge_dst, edge_weights.unsqueeze(1) * hidden_edges)
        # Graph embeddings by sum pooling
        batch_size = int(node_to_graph.max().item()) + 1 if node_to_graph.numel() > 0 else 0
        graph_embeds = torch.zeros((batch_size, self.hidden_size), device=hidden_edges.device)
        if N > 0:
            graph_embeds.index_add_(0, node_to_graph, node_repr)
        # Predictions
        pred_node = self.node_head(node_repr)
        pred_edge = self.edge_head(hidden_edges)
        pred_graph = self.graph_head(graph_embeds).squeeze(-1)  # shape [batch_size]
        return pred_node, pred_edge, pred_graph

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
    monomers, ratios, edges_info, Xn = parse_polymer_smiles(smiles)
    if len(monomers) == 0:
        return None
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
    weight_factor = 1.0
    if Xn is not None:
        try:
            weight_factor = 1.0 + math.log(Xn)
        except ValueError:
            weight_factor = 1.0
    for (i, j, w1, w2) in edges_info:
        if i in dummy_index_map and j in dummy_index_map:
            u = dummy_index_map[i]
            v = dummy_index_map[j]
            bf = [1,0,0,0, 0,0] + [1,0,0,0,0,0]  # 12-length feature vector for polymer bond
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV file with poly_chemprop_input column.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the pretrained model.')
    parser.add_argument('--pretrain_frac', type=float, default=1.0, help='Fraction of dataset to use for pretraining.')
    parser.add_argument('--val_frac', type=float, default=0.1, help='Fraction of data to use for validation.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--hidden_size', type=int, default=300, help='Hidden dimensionality for GNN.')
    parser.add_argument('--depth', type=int, default=3, help='Number of message passing steps.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability (not used in message passing).')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    parser.add_argument('--no_cuda', action='store_true', help='Disable GPU usage even if available.')
    parser.add_argument('--dataset_type', type=str, default='regression', help='Dataset type (for compatibility, not used).')
    parser.add_argument('--ignore_columns', type=str, default=None, help='Columns to ignore (not used).')
    parser.add_argument('--features_path', type=str, default=None, help='Path to additional features (not used).')
    parser.add_argument('--atom_descriptors_path', type=str, default=None, help='Path to atom descriptors (not used).')
    parser.add_argument('--bond_features_path', type=str, default=None, help='Path to bond features (not used).')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    df = pd.read_csv(args.data_path)
    if SMILES_COL not in df.columns:
        logging.error(f"Column '{SMILES_COL}' not found in data file.")
        return
    smiles_list = df[SMILES_COL].astype(str).tolist()
    total_data = len(smiles_list)
    if args.pretrain_frac < 1.0:
        subset_size = int(total_data * args.pretrain_frac)
        subset_size = max(subset_size, 1)
        smiles_list = random.sample(smiles_list, subset_size)
        logging.info(f"Subsampling dataset to {subset_size} entries out of {total_data}.")
    else:
        logging.info(f"Using full dataset of {total_data} entries.")
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
    random.shuffle(graphs)
    val_count = int(len(graphs) * args.val_frac)
    val_graphs = graphs[:val_count]
    train_graphs = graphs[val_count:]
    logging.info(f"Training on {len(train_graphs)} samples, validating on {len(val_graphs)} samples.")
    train_dataset = PolymerDataset(train_graphs)
    val_dataset = PolymerDataset(val_graphs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)
    atom_feat_dim = len(train_graphs[0].atom_features[0])
    bond_feat_dim = len(train_graphs[0].edge_features[0]) if train_graphs[0].n_edges > 0 else 0
    model = SSLPretrainModel(atom_feat_dim, bond_feat_dim, args.hidden_size, args.depth, args.dropout)
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs+1):
        model.train()
        train_losses = []
        for batch in train_loader:
            atom_feats = batch['atom_feats'].to(device)
            edge_src = batch['edge_src'].to(device)
            edge_dst = batch['edge_dst'].to(device)
            edge_feats = batch['edge_feats'].to(device)
            edge_weights = batch['edge_weights'].to(device) if batch['edge_weights'].numel() > 0 else torch.tensor([], device=device)
            b2rev = batch['b2rev'].to(device)
            node_to_graph = batch['node_to_graph'].to(device)
            batch_size = batch['batch_size']
            mask_atom_indices = []
            mask_edge_indices = []
            for g_idx in range(batch_size):
                node_indices = torch.nonzero(node_to_graph == g_idx, as_tuple=True)[0]
                edge_indices = torch.nonzero(node_to_graph[edge_src] == g_idx, as_tuple=True)[0]
                if node_indices.numel() > 0:
                    perm = torch.randperm(node_indices.numel())[:2]
                    sel_nodes = node_indices[perm]
                    mask_atom_indices.extend(sel_nodes.tolist())
                if edge_indices.numel() > 0:
                    perm_e = torch.randperm(edge_indices.numel())[:2]
                    sel_edges = edge_indices[perm_e]
                    for ei in sel_edges:
                        ei = int(ei.item())
                        mask_edge_indices.append(ei)
                        rev_ei = int(b2rev[ei].item())
                        if rev_ei not in mask_edge_indices:
                            mask_edge_indices.append(rev_ei)
            mask_atom_indices = list(set(mask_atom_indices))
            mask_edge_indices = list(set(mask_edge_indices))
            if mask_atom_indices:
                atom_feats[mask_atom_indices] = 0.0
            if mask_edge_indices:
                edge_feats[mask_edge_indices] = 0.0
            pred_node, pred_edge, pred_graph = model(atom_feats, edge_src, edge_dst, edge_feats, edge_weights, b2rev, node_to_graph)
            loss_node = 0.0
            loss_edge = 0.0
            if mask_atom_indices:
                true_node_feats = batch['atom_feats'].to(device)[mask_atom_indices]
                pred_node_feats = pred_node[mask_atom_indices]
                loss_node = F.mse_loss(pred_node_feats, true_node_feats)
            if mask_edge_indices:
                true_edge_feats = batch['edge_feats'].to(device)[mask_edge_indices]
                pred_edge_feats = pred_edge[mask_edge_indices]
                loss_edge = F.mse_loss(pred_edge_feats, true_edge_feats)
            true_graph_vals = batch['mol_weights'].to(device)
            pred_graph_vals = pred_graph
            loss_graph = F.mse_loss(pred_graph_vals, true_graph_vals)
            loss = loss_node + loss_edge + loss_graph
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                atom_feats = batch['atom_feats'].to(device)
                edge_src = batch['edge_src'].to(device)
                edge_dst = batch['edge_dst'].to(device)
                edge_feats = batch['edge_feats'].to(device)
                edge_weights = batch['edge_weights'].to(device) if batch['edge_weights'].numel() > 0 else torch.tensor([], device=device)
                b2rev = batch['b2rev'].to(device)
                node_to_graph = batch['node_to_graph'].to(device)
                mask_atom_indices = []
                mask_edge_indices = []
                batch_size = batch['batch_size']
                for g_idx in range(batch_size):
                    node_indices = torch.nonzero(node_to_graph == g_idx, as_tuple=True)[0]
                    edge_indices = torch.nonzero(node_to_graph[edge_src] == g_idx, as_tuple=True)[0]
                    if node_indices.numel() > 0:
                        perm = torch.randperm(node_indices.numel())[:2]
                        sel_nodes = node_indices[perm]
                        mask_atom_indices.extend(sel_nodes.tolist())
                    if edge_indices.numel() > 0:
                        perm_e = torch.randperm(edge_indices.numel())[:2]
                        sel_edges = edge_indices[perm_e]
                        for ei in sel_edges:
                            ei = int(ei.item())
                            mask_edge_indices.append(ei)
                            rev_ei = int(b2rev[ei].item())
                            if rev_ei not in mask_edge_indices:
                                mask_edge_indices.append(rev_ei)
                mask_atom_indices = list(set(mask_atom_indices))
                mask_edge_indices = list(set(mask_edge_indices))
                if mask_atom_indices:
                    atom_feats[mask_atom_indices] = 0.0
                if mask_edge_indices:
                    edge_feats[mask_edge_indices] = 0.0
                pred_node, pred_edge, pred_graph = model(atom_feats, edge_src, edge_dst, edge_feats, edge_weights, b2rev, node_to_graph)
                loss_node = 0.0
                loss_edge = 0.0
                if mask_atom_indices:
                    true_node_feats = batch['atom_feats'].to(device)[mask_atom_indices]
                    pred_node_feats = pred_node[mask_atom_indices]
                    loss_node = F.mse_loss(pred_node_feats, true_node_feats)
                if mask_edge_indices:
                    true_edge_feats = batch['edge_feats'].to(device)[mask_edge_indices]
                    pred_edge_feats = pred_edge[mask_edge_indices]
                    loss_edge = F.mse_loss(pred_edge_feats, true_edge_feats)
                true_graph_vals = batch['mol_weights'].to(device)
                pred_graph_vals = pred_graph
                loss_graph = F.mse_loss(pred_graph_vals, true_graph_vals)
                loss = loss_node + loss_edge + loss_graph
                val_losses.append(loss.item())
        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        logging.info(f"Epoch {epoch}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "model.pt")
    torch.save(model.state_dict(), save_path)
    logging.info(f"Saved trained model to {save_path}")

if __name__ == "__main__":
    if Chem is None:
        logging.error("RDKit is not installed. Please install RDKit to run this script.")
    else:
        main()
