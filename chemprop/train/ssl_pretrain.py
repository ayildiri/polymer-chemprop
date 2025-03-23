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

# Chemprop imports
from chemprop.data import MoleculeDatapoint, MoleculeDataset
from chemprop.features import BatchMolGraph, MolGraph
from chemprop.models.mpn import MPN
from chemprop.args import TrainArgs

class SSLPretrainModel(nn.Module):
    """Chemprop-based GNN with separate heads for node, edge, and graph tasks."""
    def __init__(self, hidden_size: int, atom_feat_size: int, bond_feat_size: int):
        super(SSLPretrainModel, self).__init__()
        # Configure Chemprop arguments for the MPN encoder
        args = TrainArgs()
        args.hidden_size = hidden_size
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.atom_messages = False        # use bond-based message passing (default Chemprop behavior)
        args.number_of_molecules = 1      # single-molecule input (polymer treated as one graph)
        args.mpn_shared = True
        args.depth = 3                    # number of message-passing steps (as in the paper)
        args.dropout = 0.0                # no dropout for pretraining unless specified
        args.aggregation = 'mean'         # how to aggregate node embeddings for graph embedding
        # Initialize Chemprop MPN encoder
        self.encoder = MPN(args=args)
        # SSL prediction heads
        self.node_head = nn.Linear(hidden_size, atom_feat_size)        # for masked atom feature reconstruction
        self.edge_head = nn.Linear(hidden_size, bond_feat_size)        # for masked bond feature reconstruction
        self.graph_head = nn.Sequential(                               # for graph-level property prediction
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, batch_graph: BatchMolGraph):
        # Run message passing to obtain final atom embeddings (shape: total_atoms_in_batch x hidden_size)
        atom_hiddens = self.encoder.encoder[0](batch_graph)  # accessing the first (and only) MPN encoder
        # Compute graph embeddings by aggregating atom embeddings for each molecule in the batch
        graph_embeddings = []
        for start, size in batch_graph.a_scope:  # a_scope gives (start_index, num_atoms) for each molecule
            if size == 0:
                # No atoms (edge case), use a zero vector
                graph_embeddings.append(torch.zeros(atom_hiddens.size(1), device=atom_hiddens.device))
            else:
                # Sum (or mean) aggregation of atom embeddings for this molecule
                mol_emb = atom_hiddens[start : start + size].sum(dim=0)
                graph_embeddings.append(mol_emb)
        graph_embeddings = torch.stack(graph_embeddings, dim=0)  # shape: (batch_size, hidden_size)

        # Node-level predictions: predict masked atom features for every atom (unmasked atoms' predictions will be ignored in loss)
        node_preds = self.node_head(atom_hiddens)  # shape: (total_atoms, atom_feat_size)

        # Edge-level predictions: predict masked bond features.
        # We compute one prediction per actual bond (undirected). For each bond, use the two endpoint atom embeddings.
        bond_preds_list = []
        visited = set()
        # Map each directed bond index to its prediction index for ease of loss mapping
        bond_pred_index_map = {}
        pred_count = 0
        for e in range(len(batch_graph.f_bonds)):
            rev_e = batch_graph.b2revb[e].item()
            if e in visited:
                continue  # this bond (or its reverse) already handled
            visited.add(e)
            visited.add(rev_e)
            # Get the two atom indices (global) for this bond's endpoints
            a1_idx = batch_graph.b2a[e].item()
            a2_idx = batch_graph.b2a[rev_e].item()
            # Map directed indices to the index in bond_preds_list
            bond_pred_index_map[e] = pred_count
            bond_pred_index_map[rev_e] = pred_count
            pred_count += 1
            # Average the two atom hidden states to represent the bond (undirected)
            h_avg = 0.5 * (atom_hiddens[a1_idx] + atom_hiddens[a2_idx])
            bond_preds_list.append(self.edge_head(h_avg))
        # Stack bond predictions into a tensor
        if bond_preds_list:
            edge_preds = torch.stack(bond_preds_list, dim=0)  # shape: (num_actual_bonds, bond_feat_size)
        else:
            # If no bonds in batch (unlikely, unless molecules are single atoms), create an empty tensor
            edge_preds = torch.empty(0, batch_graph.f_bonds.size(1), device=atom_hiddens.device)

        # Graph-level prediction: one output per molecule
        graph_preds = self.graph_head(graph_embeddings)  # shape: (batch_size, 1)
        return node_preds, edge_preds, graph_preds, bond_pred_index_map

def safe_float(x):
    try:
        return float(x)
    except ValueError:
        return None

def compute_ensemble_weight(poly_input: str) -> float:
    """Compute ensemble molecular weight from a polymer string with stoichiometry weights."""
    try:
        parts = poly_input.split('|')
        monomer_smiles = parts[0].split('.')  # SMILES for each monomer (separated by '.')
        monomer_weights = parts[1:-1]         # expected to be two numbers (fractions for each monomer)
        if len(monomer_smiles) < 2 or len(monomer_weights) < 2:
            print(f"[Warning] Incomplete polymer entry: {poly_input}")
            return 0.0
        w1, w2 = safe_float(monomer_weights[0]), safe_float(monomer_weights[1])
        if w1 is None or w2 is None:
            print(f"[Warning] Non-numeric weights in polymer entry: {poly_input}")
            return 0.0
        # Calculate molecular weight of each monomer
        mol1 = Chem.MolFromSmiles(monomer_smiles[0])
        mol2 = Chem.MolFromSmiles(monomer_smiles[1])
        if mol1 is None or mol2 is None:
            print(f"[Warning] Invalid monomer SMILES in: {poly_input}")
            return 0.0
        mw1 = Descriptors.ExactMolWt(mol1)
        mw2 = Descriptors.ExactMolWt(mol2)
        # Ensemble (average) molecular weight = w1 * MW1 + w2 * MW2
        return w1 * mw1 + w2 * mw2
    except Exception as e:
        print(f"[Error] Failed to compute ensemble weight for: {poly_input} — {e}")
        return 0.0

def load_polymer_data(path: str, polymer_mode: bool) -> List[str]:
    """Load polymer SMILES strings from a CSV or TXT file. Returns a list of SMILES (with polymer syntax if applicable)."""
    data = []
    with open(path, 'r') as f:
        # Attempt to read header (for CSVs). If no header, handle accordingly.
        header = f.readline().strip().split(',')
        if polymer_mode:
            # Identify the column that contains the polymer representation
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
        # If the first line is actually data (no header present or header looks like a SMILES), include it
        first_line = None
        if len(header) == 1 and any(ch in header[0] for ch in ['.', '=', '*', '|', '[', ']']):
            # The first line might be a SMILES (polymer or molecule) rather than a header
            first_line = header
            header = None
        elif header and any(ch in header[0] for ch in ['[', '.', '=', '*', '|']):
            # The header line itself looks like SMILES data (in case file has no header)
            first_line = header
            header = None
        if first_line:
            data.append(','.join(first_line))
        # Read the rest of the lines
        for line in f:
            if not line.strip():
                continue
            data.append(line.strip())
    # Extract the SMILES/polymer strings from each line if CSV
    if header is not None:
        polymer_list = []
        for line in data:
            cols = line.split(',')
            if len(cols) > smiles_col:
                polymer_list.append(cols[smiles_col])
        return polymer_list
    else:
        # Data already consists of SMILES strings
        return data

def save_smiles_splits(save_dir: str, train_data: List[str], val_data: List[str], test_data: List[str]):
    """Save the train/val/test splits of SMILES strings to CSV files in the save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "smiles_train.csv"), 'w') as f:
        for s in train_data:
            f.write(s + "\n")
    with open(os.path.join(save_dir, "smiles_val.csv"), 'w') as f:
        for s in val_data:
            f.write(s + "\n")
    if test_data:
        with open(os.path.join(save_dir, "smiles_test.csv"), 'w') as f:
            for s in test_data:
                f.write(s + "\n")

def move_batch_to_device(batch_graph: BatchMolGraph, device: torch.device):
    """Move all torch tensors inside a BatchMolGraph to the specified device (in-place)."""
    batch_graph.f_atoms = batch_graph.f_atoms.to(device)
    batch_graph.f_bonds = batch_graph.f_bonds.to(device)
    batch_graph.a2b = batch_graph.a2b.to(device)
    batch_graph.b2a = batch_graph.b2a.to(device)
    batch_graph.b2revb = batch_graph.b2revb.to(device)
    # The BatchMolGraph in Chemprop might have additional attributes; move them if present
    if hasattr(batch_graph, "a2a") and batch_graph.a2a is not None:
        batch_graph.a2a = batch_graph.a2a.to(device)
    if hasattr(batch_graph, "features_batch") and batch_graph.features_batch is not None:
        batch_graph.features_batch = batch_graph.features_batch.to(device)

def main():
    parser = argparse.ArgumentParser(description="Self-supervised pretraining (Phase 1) for polymer GNN")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the unlabeled polymer data (CSV or TXT).")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the pretrained model and splits.")
    parser.add_argument('--polymer', action='store_true', help="Flag indicating input uses polymer notation (with '|' metadata).")
    parser.add_argument('--epochs', type=int, default=100, help="Number of pretraining epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    parser.add_argument('--num_workers', type=int, default=0, help="Number of DataLoader workers for loading data (if used).")
    parser.add_argument('--hidden_size', type=int, default=300, help="Hidden size for GNN message passing layers.")
    parser.add_argument('--ssl_frac', type=float, default=1.0, help="Fraction of data to use for SSL pretraining (between 0 and 1).")
    parser.add_argument('--save_smiles_splits', action='store_true',
                        help="If set, save the train/val/test SMILES splits to files in save_dir.")
    args = parser.parse_args()

    # Load polymer data from file
    polymers = load_polymer_data(args.data_path, polymer_mode=args.polymer)
    if args.ssl_frac < 1.0:
        # Subsample the data if a fraction is specified
        random.shuffle(polymers)
        subset_size = int(len(polymers) * args.ssl_frac)
        polymers = polymers[:subset_size]
        print(f"Using {subset_size} samples ({args.ssl_frac*100:.0f}%) out of {len(polymers)} total data for SSL pretraining.")
    else:
        # Shuffle all data
        random.shuffle(polymers)

    # Split data into train/val/test (80/10/10 split by default)
    n = len(polymers)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_polymers = polymers[:n_train]
    val_polymers = polymers[n_train : n_train + n_val]
    test_polymers = polymers[n_train + n_val :]
    if args.save_smiles_splits:
        # Save the splits for reference if needed
        save_smiles_splits(args.save_dir, train_polymers, val_polymers, test_polymers)

    # Determine feature dimensions by inspecting one sample molecule/polymer
    sample_poly = train_polymers[0] if train_polymers else polymers[0]
    # If polymer notation is used, strip at '|' to get the actual SMILES part for feature extraction
    sample_smiles = sample_poly.split('|')[0] if args.polymer else sample_poly
    sample_graph = MolGraph(sample_smiles)
    atom_feat_size = len(sample_graph.f_atoms[0]) if sample_graph.n_atoms > 0 else 0
    bond_feat_size = len(sample_graph.f_bonds[0]) if sample_graph.n_bonds > 0 else 0

    # Initialize model and optimizer
    model = SSLPretrainModel(hidden_size=args.hidden_size, atom_feat_size=atom_feat_size, bond_feat_size=bond_feat_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    # Optionally set a manual seed for reproducibility (commented out by default)
    # random.seed(0); np.random.seed(0); torch.manual_seed(0)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_polymers)
        total_loss_value = 0.0
        for i in range(0, len(train_polymers), args.batch_size):
            batch_smiles = train_polymers[i : i + args.batch_size]
            if not batch_smiles:
                continue

            # Prepare MolGraph objects for each molecule/polymer in the batch
            mol_graphs = []
            # Lists to store original features of masked atoms/bonds and their global indices
            masked_atom_orig_feats = []
            masked_atom_indices_global = []
            masked_bond_orig_feats = []
            masked_bond_indices_pred = []  # we will store index corresponding to edge_preds row

            for mol_idx, smi in enumerate(batch_smiles):
                # Compute graph-level target (ensemble molecular weight for polymers)
                # If not polymer mode, we can skip or use 0.0 as a placeholder target.
                mw_target = compute_ensemble_weight(smi) if args.polymer else 0.0

                # Parse the SMILES to a MolGraph (Chemprop will handle polymer SMARTS with wildcard atoms if present)
                smiles_only = smi.split('|')[0] if args.polymer else smi
                mg = MolGraph(smiles_only)

                # Mask random atoms
                n_atoms = mg.n_atoms
                if n_atoms > 0:
                    # Choose up to 2 atoms to mask (or fewer if molecule is tiny)
                    num_mask_atoms = min(2, n_atoms)
                    atom_indices = random.sample(range(n_atoms), num_mask_atoms)
                else:
                    atom_indices = []
                for atom_idx in atom_indices:
                    # Save original atom feature and replace with zeros (mask)
                    masked_atom_orig_feats.append(mg.f_atoms[atom_idx][:])  # copy feature list
                    mg.f_atoms[atom_idx] = [0.0] * len(mg.f_atoms[atom_idx])
                    # We cannot determine global index yet (we will do after BatchMolGraph is created),
                    # so store (molecule_index_in_batch, atom_idx)
                    masked_atom_indices_global.append((mol_idx, atom_idx))

                # Mask random bonds (actual bonds, affecting both bond directions)
                n_bonds_dir = mg.n_bonds  # directed bonds count (2 * actual bonds)
                if n_bonds_dir > 1:
                    # Number of actual bonds = n_bonds_dir // 2
                    actual_bonds = n_bonds_dir // 2
                    num_mask_bonds = min(2, actual_bonds)
                    bond_indices = random.sample(range(actual_bonds), num_mask_bonds)
                else:
                    bond_indices = []
                for b_idx in bond_indices:
                    # Determine directed indices for this bond
                    dir1 = 2 * b_idx
                    dir2 = dir1 + 1
                    # Save original bond feature (for one direction; both dir1 and dir2 share the same features)
                    masked_bond_orig_feats.append(mg.f_bonds[dir1][:])
                    # Mask both directions of this bond
                    mg.f_bonds[dir1] = [0.0] * len(mg.f_bonds[dir1])
                    mg.f_bonds[dir2] = [0.0] * len(mg.f_bonds[dir2])
                    # Store (molecule_index_in_batch, directed_index) for loss calculation.
                    # We'll map this to the prediction index after forwarding.
                    masked_bond_indices_pred.append((mol_idx, dir1))

                mol_graphs.append(mg)
                # Attach graph-level target to keep track (we will use graph_targets list below)
                # (We could store in parallel list; see below outside loop.)

            # Prepare BatchMolGraph for the batch
            batch_graph = BatchMolGraph(mol_graphs)
            # Move batch to device (especially important if using GPU)
            move_batch_to_device(batch_graph, device)

            # Forward pass to get predictions and the bond index map
            node_preds, edge_preds, graph_preds, bond_pred_index_map = model(batch_graph)

            # Compute losses for this batch
            node_loss = torch.tensor(0.0, device=device)
            edge_loss = torch.tensor(0.0, device=device)
            graph_loss = torch.tensor(0.0, device=device)

            # Calculate node-level reconstruction loss (MSE)
            if masked_atom_indices_global:
                for (mol_idx, atom_idx), orig_feat in zip(masked_atom_indices_global, masked_atom_orig_feats):
                    # Convert local atom index to global index using a_scope
                    atom_offset = batch_graph.a_scope[mol_idx][0]
                    global_atom_idx = atom_offset + atom_idx
                    if global_atom_idx >= node_preds.size(0):
                        # This should not happen if indexing is correct, but guard just in case
                        print(f"Skipping masked atom {global_atom_idx} (out of range)")
                        continue
                    pred_feat = node_preds[global_atom_idx]              # predicted feature vector
                    target_feat = torch.tensor(orig_feat, dtype=torch.float32, device=device)
                    node_loss += torch.mean((pred_feat - target_feat) ** 2)
                node_loss = node_loss / len(masked_atom_indices_global)  # average loss per masked atom

            # Calculate edge-level reconstruction loss (MSE)
            if masked_bond_indices_pred:
                for (mol_idx, dir_idx), orig_feat in zip(masked_bond_indices_pred, masked_bond_orig_feats):
                    # Convert local directed bond index to global directed index using b_scope
                    bond_offset = batch_graph.b_scope[mol_idx][0]
                    global_dir_idx = bond_offset + dir_idx
                    # Map the global directed bond index to the index in edge_preds (using bond_pred_index_map)
                    if global_dir_idx not in bond_pred_index_map:
                        # In case something went wrong and this bond wasn't predicted (should not happen)
                        print(f"Skipping masked bond {global_dir_idx}: no prediction available")
                        continue
                    pred_idx = bond_pred_index_map[global_dir_idx]
                    if pred_idx >= edge_preds.size(0):
                        print(f"Skipping masked bond index {pred_idx} (out of range for predictions)")
                        continue
                    pred_feat = edge_preds[pred_idx]
                    target_feat = torch.tensor(orig_feat, dtype=torch.float32, device=device)
                    edge_loss += torch.mean((pred_feat - target_feat) ** 2)
                edge_loss = edge_loss / len(masked_bond_indices_pred)  # average loss per masked bond

            # Calculate graph-level loss (MSE for ensemble molecular weight or other property)
            # Prepare target tensor for graph labels (ensemble molecular weight or 0.0 if not polymer)
            graph_targets = []
            for smi in batch_smiles:
                if args.polymer:
                    graph_targets.append(compute_ensemble_weight(smi))
                else:
                    graph_targets.append(0.0)
            target_tensor = torch.tensor(graph_targets, dtype=torch.float32, device=device).unsqueeze(1)
            graph_loss = torch.mean((graph_preds - target_tensor) ** 2)

            # Combine losses (all SSL tasks)
            total_loss = node_loss + edge_loss + graph_loss

            # Backpropagation and optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_value += total_loss.item()
        print(f"Epoch {epoch}/{args.epochs} - Training loss: {total_loss_value:.4f}")

    # Save the trained model (encoder + heads)
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "ssl_pretrained_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Pretrained model saved to {model_path}")

if __name__ == "__main__":
    main()
