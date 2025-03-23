import os
import argparse
import random
from typing import List
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

import torch
import torch.nn as nn
import torch.optim as optim

# ✅ Correct imports from Chemprop
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.features import BatchMolGraph, MolGraph
from chemprop.models.mpn import MPN  # ✅ wD-MPNN encoder
from chemprop.args import TrainArgs


class SSLPretrainModel(nn.Module):
    """Chemprop-based GNN with separate heads for node, edge, and graph tasks."""
    def __init__(self, hidden_size: int, atom_feat_size: int, bond_feat_size: int):
        super(SSLPretrainModel, self).__init__()

        # ✅ Create TrainArgs and set relevant fields
        args = TrainArgs()
        args.hidden_size = hidden_size
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.atom_messages = False                  # or True if desired
        args.number_of_molecules = 1
        args.mpn_shared = True
        args.depth = 3                              # as in the paper
        args.dropout = 0.0                          # default dropout
        args.aggregation = 'mean'                   # or 'sum', etc.

        # ✅ Chemprop message passing encoder (MPN)
        self.encoder = MPN(args=args)

        # 🔁 Prediction heads
        self.node_head = nn.Linear(hidden_size, atom_feat_size)         # node feature reconstruction
        self.edge_head = nn.Linear(hidden_size, bond_feat_size)         # bond feature reconstruction

        # 📐 Graph head: a two-layer MLP for ensemble molecular weight prediction
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, batch_graph: BatchMolGraph):
        # Perform message passing to get final atom embeddings.
        # Chemprop's MessagePassing returns a molecule-level embedding by default, 
        # so we use internal methods to get atom-level embeddings.
        atom_hiddens = self.encoder.encoder[0](batch_graph)
        # batch_graph.atom_scope is a list of (start_index, num_atoms) for each molecule
        graph_embeddings = []
        for (start, size) in batch_graph.a_scope:
            if size == 0:
                mol_embedding = torch.zeros(atom_hiddens.size(1), device=atom_hiddens.device)
            else:
                mol_embedding = atom_hiddens[start:start+size].sum(dim=0)
            graph_embeddings.append(mol_embedding)
        
        graph_embeddings = torch.stack(graph_embeddings, dim=0)  # (num_molecules, hidden_size)

        node_pred = self.node_head(atom_hiddens)  # shape (total_atoms, atom_feat_size)
        # Predict bond features: for each bond, combine the two endpoint atom embeddings.
        # We average the embeddings of the two atoms connected by each bond.
        E = len(batch_graph.f_bonds)
        bond_preds = []
        visited = set()
        
        bond_to_mol = []
        for mol_idx, (start, length) in enumerate(batch_graph.b_scope):
            bond_to_mol.extend([mol_idx] * length)
        bond_to_mol = torch.tensor(bond_to_mol, device=atom_hiddens.device)
        
        for e in range(E):
            rev_e = batch_graph.b2revb[e].item()
            if rev_e in visited:
                continue
            visited.add(e)
            visited.add(rev_e)
        
            mol_idx = bond_to_mol[e].item()
        
            a1_local = batch_graph.b2a[e].item()
            a2_local = batch_graph.b2a[rev_e].item()
        
            a1_start, _ = batch_graph.a_scope[mol_idx]
            a2_start, _ = batch_graph.a_scope[mol_idx]
        
            a1_idx = a1_start + a1_local
            a2_idx = a2_start + a2_local
        
            if a1_idx >= atom_hiddens.size(0) or a2_idx >= atom_hiddens.size(0):
                print(f"⚠️  Skipping edge ({e}) due to out-of-bounds atom indices: a1={a1_idx}, a2={a2_idx}")
                continue
        
            h1 = atom_hiddens[a1_idx]
            h2 = atom_hiddens[a2_idx]
            bond_emb = 0.5 * (h1 + h2)
            bond_preds.append(self.edge_head(bond_emb))



        rev_indices = batch_graph.rev_edge_index  # mapping from edge index to reverse edge index
        bond_preds: List[torch.Tensor] = []
        visited = set()
        for e in range(E):
            rev_e = rev_indices[e].item()
            if rev_e in visited:
                continue  # skip reverse edge once we've handled the pair
            visited.add(e); visited.add(rev_e)
            # endpoints of edge e:
            a1_idx, a2_idx = batch_graph.edge_index[:, e]
            # endpoints of reverse (should be a2 -> a1)
            # a1_idx_rev, a2_idx_rev = batch_graph.edge_index[:, rev_e]  # a2_idx_rev == a1_idx
            # Get atom embeddings
            h1 = atom_hiddens[a1_idx]; h2 = atom_hiddens[a2_idx]
            bond_emb = 0.5 * (h1 + h2)
            bond_preds.append(self.edge_head(bond_emb))  # predict original bond features
        edge_pred = torch.stack(bond_preds, dim=0) if bond_preds else torch.empty(0, batch_graph.num_edge_features)
        # Predict graph-level property (ensemble molecular weight) for each molecule
        graph_pred = self.graph_head(graph_embeddings)  # shape (batch_size, 1)
        return node_pred, edge_pred, graph_pred

def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return None

def compute_ensemble_weight(poly_input: str):
    try:
        parts = poly_input.split('|')
        monomer_smiles = parts[0].split('.')
        monomer_weights = parts[1:-1]

        if len(monomer_smiles) < 2 or len(monomer_weights) < 2:
            print(f"[Warning] Incomplete polymer entry: {poly_input}")
            return 0.0

        w1 = safe_float(monomer_weights[0])
        w2 = safe_float(monomer_weights[1])
        if w1 is None or w2 is None:
            print(f"[Warning] Non-numeric weights in: {poly_input}")
            return 0.0

        mol1 = Chem.MolFromSmiles(monomer_smiles[0])
        mol2 = Chem.MolFromSmiles(monomer_smiles[1])

        if mol1 is None or mol2 is None:
            print(f"[Warning] Invalid monomer SMILES in: {poly_input}")
            return 0.0

        mw1 = Descriptors.ExactMolWt(mol1)
        mw2 = Descriptors.ExactMolWt(mol2)

        M_ensemble = w1 * mw1 + w2 * mw2
        return M_ensemble

    except Exception as e:
        print(f"[Error] Failed to compute ensemble weight for: {poly_input} — {e}")
        return 0.0

def load_polymer_data(path: str, polymer_mode: bool) -> List[str]:
    """Load polymer SMILES data from a file (CSV or TXT). Returns list of polymer strings."""
    data = []
    with open(path, 'r') as f:
        header = f.readline().strip().split(',')

        # ✅ Detect which column contains SMILES
        if polymer_mode:
            if "poly_chemprop_input" in header:
                smiles_col = header.index("poly_chemprop_input")
            elif "polymer" in header:
                smiles_col = header.index("polymer")
            elif "smiles" in header:
                smiles_col = header.index("smiles")
            else:
                smiles_col = 0
        else:
            smiles_col = 0

        # 🧠 Handle edge case: file has no header or starts with a SMILES
        first_line = None
        if len(header) == 1 and any(ch in header[0] for ch in ['.', '=', '*', '|']):
            f.seek(0)
            header = None
        else:
            if header and any(ch in header[0] for ch in ['[', '.', '=', '*', '|']):
                first_line = header
                header = None

        if first_line:
            data.append(','.join(first_line))

        for line in f:
            if not line.strip():
                continue
            data.append(line.strip())

    # ✅ Extract SMILES column if CSV
    if header is not None:
        polymer_list = []
        for line in data:
            cols = line.split(',')
            if len(cols) > smiles_col:
                polymer_list.append(cols[smiles_col])
        return polymer_list
    else:
        return data

   
def save_smiles_splits(save_dir: str, train_data: List[str], val_data: List[str], test_data: List[str]):
    """Save the SMILES splits to files in the save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "smiles_train.csv"), 'w') as f:
        for s in train_data:
            f.write(s + "\n")
    with open(os.path.join(save_dir, "smiles_val.csv"), 'w') as f:
        for s in val_data:
            f.write(s + "\n")
    if test_data is not None:
        with open(os.path.join(save_dir, "smiles_test.csv"), 'w') as f:
            for s in test_data:
                f.write(s + "\n")
                
def move_batch_to_device(batch_graph, device):
    batch_graph.f_atoms = batch_graph.f_atoms.to(device)
    batch_graph.f_bonds = batch_graph.f_bonds.to(device)
    batch_graph.a2b = batch_graph.a2b.to(device)
    batch_graph.b2a = batch_graph.b2a.to(device)
    batch_graph.b2revb = batch_graph.b2revb.to(device)

    # Optional: move if these exist in your fork (skip if not)
    if hasattr(batch_graph, "a2a") and batch_graph.a2a is not None:
        batch_graph.a2a = batch_graph.a2a.to(device)

    if hasattr(batch_graph, "features_batch") and batch_graph.features_batch is not None:
        batch_graph.features_batch = batch_graph.features_batch.to(device)

    
    
def main():
    parser = argparse.ArgumentParser(description="Self-supervised pretraining (Phase 1) for polymer GNN")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the unlabeled polymer data (CSV or TXT).")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the pretrained model and splits.")
    parser.add_argument('--polymer', action='store_true', help="Flag indicating polymer SMILES with connection point notation are used.")
    parser.add_argument('--epochs', type=int, default=100, help="Number of pretraining epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of DataLoader workers for loading data.")
    parser.add_argument('--hidden_size', type=int, default=300, help="Hidden size for GNN message passing layers.")
    parser.add_argument('--save_smiles_splits', action='store_true', help="If set, save the train/val (and test) SMILES splits to CSV files.")
    args = parser.parse_args()

    # Load polymer data
    polymers = load_polymer_data(args.data_path, polymer_mode=args.polymer)
    # Split data into train/val (and test if desired). Here we do an 80/10/10 split randomly.
    random.shuffle(polymers)
    n = len(polymers)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_polymers = polymers[:n_train]
    val_polymers = polymers[n_train:n_train+n_val]
    test_polymers = polymers[n_train+n_val:]  # remaining 10%
    if args.save_smiles_splits:
        save_smiles_splits(args.save_dir, train_polymers, val_polymers, test_polymers if test_polymers else [])

    # Create model. Determine feature dimensions from a sample polymer:
    sample_poly = train_polymers[0] if train_polymers else polymers[0]
    # Use Chemprop's MolGraph to get feature lengths
    sample_graph = MolGraph(sample_poly.split('|')[0])  # this will parse the SMILES up to '|' if present
    atom_feat_size = len(sample_graph.f_atoms[0])  # length of atom feature vector
    bond_feat_size = len(sample_graph.f_bonds[0])  # length of bond feature vector
    model = SSLPretrainModel(hidden_size=args.hidden_size, atom_feat_size=atom_feat_size, bond_feat_size=bond_feat_size)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_polymers)
        # Manual batching (since dataset is small/medium, for simplicity)
        for i in range(0, len(train_polymers), args.batch_size):
            batch_smiles = train_polymers[i : i + args.batch_size]
            # Prepare batch MolGraph objects, mask random nodes/edges and record original features
            mol_graphs = []
            original_node_feats = []  # will store original features of masked nodes
            pred_node_indices = []    # indices of masked nodes in the BatchMolGraph
            original_edge_feats = []
            pred_edge_indices = []    # indices of masked edges (directed) in BatchMolGraph
            # Also gather graph-level targets for batch
            graph_targets = []
            for smi in batch_smiles:
                # Compute graph-level target (molecular weight)
                if args.polymer:
                    mw = compute_ensemble_weight(smi)
                else:
                    mw = 0.0  # (If not polymer mode, no graph pseudo-label; but in polymer mode we always compute)
                graph_targets.append(mw)
                # Create MolGraph for the polymer (Chemprop will parse the SMILES into atoms/bonds)
                cleaned_smi = smi.split('|')[0] if args.polymer else smi
                mg = MolGraph(cleaned_smi)
                n_atoms = mg.n_atoms
                n_bonds = mg.n_bonds  # number of *directed* bonds
                # Randomly choose 2 atom indices to mask (if available)
                node_indices = random.sample(range(n_atoms), min(2, n_atoms))
                for idx in node_indices:
                    original_node_feats.append(mg.f_atoms[idx][:])  # copy original features
                    mg.f_atoms[idx] = [0.0] * len(mg.f_atoms[idx])   # mask by zeroing out
                    # We will determine the global index in batch later once we batch graphs.
                    pred_node_indices.append((len(mol_graphs), idx))
                # Randomly choose 2 bonds (undirected) to mask, if available
                bond_indices = []
                if n_bonds > 0:
                    # Each actual bond corresponds to two directed edges (pairs in f_bonds list)
                    # We select up to 2 *pairs*:
                    bond_indices = random.sample(range(n_bonds // 2), min(2, n_bonds // 2))
                for b_idx in bond_indices:
                    # directed indices for the two directions of this bond:
                    dir1 = 2 * b_idx
                    dir2 = dir1 + 1
                    original_edge_feats.append(mg.f_bonds[dir1][:])  # original bond feature (same for both dirs)
                    # Mask both directions:
                    mg.f_bonds[dir1] = [0.0] * len(mg.f_bonds[dir1])
                    mg.f_bonds[dir2] = [0.0] * len(mg.f_bonds[dir2])
                    pred_edge_indices.append((len(mol_graphs), dir1))
                    # We record only one of the directed indices (dir1) for loss, as edge_head prediction uses combined embed.
                mol_graphs.append(mg)
            # Batch the MolGraphs into a BatchMolGraph for model input
            batch_graph = BatchMolGraph(mol_graphs)
            move_batch_to_device(batch_graph, model.encoder.device)
            # Forward pass
            node_preds, edge_preds, graph_preds = model(batch_graph)
            # Compute losses
            node_loss = 0.0; edge_loss = 0.0; graph_loss = 0.0
            # Node loss: gather predictions for masked nodes and compare to originals
            for (mol_idx, atom_idx), orig_feat in zip(pred_node_indices, original_node_feats):
                # Find the global index of this atom in the batched graph
                # BatchMolGraph stores a mapping of molecule index to atom indices range
                global_atom_index = batch_graph.atom_start_indices[mol_idx] + atom_idx
                orig = torch.tensor(orig_feat, dtype=torch.float32, device=node_preds.device)
                pred = node_preds[global_atom_index]
                node_loss += torch.mean((pred - orig)**2)  # MSE for that node (averaged over feature components)
            if len(pred_node_indices) > 0:
                node_loss = node_loss / len(pred_node_indices)  # average over masked nodes
            # Edge loss: predictions for masked bonds vs original
            # `edge_preds` list corresponds to each masked bond (undirected) in the batch in the order we appended.
            for pred_idx, orig_feat in enumerate(original_edge_feats):
                orig = torch.tensor(orig_feat, dtype=torch.float32, device=edge_preds.device)
                pred = edge_preds[pred_idx]
                edge_loss += torch.mean((pred - orig)**2)
            if len(original_edge_feats) > 0:
                edge_loss = edge_loss / len(original_edge_feats)
            # Graph loss: MSE between predicted and target molecular weight for each molecule
            target_tensor = torch.tensor(graph_targets, dtype=torch.float32, device=graph_preds.device).unsqueeze(1)
            graph_loss = torch.mean((graph_preds - target_tensor)**2)
            # Combine losses
            total_loss = node_loss + edge_loss + graph_loss
            # Backpropagate and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}/{args.epochs} completed.")

    # Save the trained model (encoder + heads)
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "ssl_pretrained_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Pretrained model saved to {model_path}")

if __name__ == "__main__":
    main()
