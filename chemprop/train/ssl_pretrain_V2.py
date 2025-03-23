import os
import csv
import math
import random
import logging
from typing import List, Tuple

import torch
import torch.nn as nn

try:
    from chemprop.data.molgraph import MolGraph, BatchMolGraph
    from chemprop.features import get_atom_fdim, get_bond_fdim
except ImportError:
    raise ImportError("Could not import chemprop (polymer-chemprop). Please install the ColeyGroup polymer-chemprop fork.")

from rdkit import Chem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Argument parsing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='Path to CSV data file containing polymer SMILES.')
parser.add_argument('--polymer', action='store_true', help='Flag indicating the input SMILES are polymer extended format.')
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save model checkpoints and outputs.')
parser.add_argument('--pretrain_frac', type=float, default=1.0, help='Fraction of data to use for SSL pretraining (remainder held out).')
parser.add_argument('--epochs', type=int, default=30, help='Number of pretraining epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for pretraining.')
parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers (0 uses main process).')
parser.add_argument('--save_smiles_splits', action='store_true', help='If set, save CSV files of the pretrain and hold-out SMILES splits.')
args = parser.parse_args()

# Ensure save directory exists
os.makedirs(args.save_dir, exist_ok=True)

# Set random seed for reproducibility of data split
random.seed(0)

# Load data
logger.info(f"Loading data from {args.data_path}")
with open(args.data_path, 'r') as f:
    reader = csv.reader(f)
    data_lines = list(reader)
# Remove header if present
if data_lines and "smiles" in data_lines[0][0].lower():
    data_lines = data_lines[1:]
smiles_all = [line[0] for line in data_lines]

# Split into pretrain and hold-out sets
total_count = len(smiles_all)
pretrain_count = math.floor(total_count * args.pretrain_frac)
if pretrain_count < 1:
    raise ValueError("pretrain_frac is too low, no data selected for pretraining.")
indices = list(range(total_count)); random.shuffle(indices)
pretrain_idx_set = set(indices[:pretrain_count])
holdout_idx_set = set(indices[pretrain_count:])
pretrain_smiles = [smiles_all[i] for i in pretrain_idx_set]
holdout_smiles = [smiles_all[i] for i in holdout_idx_set]
logger.info(f"Total entries: {total_count}. Using {len(pretrain_smiles)} for pretraining ({args.pretrain_frac*100:.1f}%). Hold-out: {len(holdout_smiles)}.")

# Save SMILES splits if requested
if args.save_smiles_splits:
    pretrain_file = os.path.join(args.save_dir, "poly_pretrain_smiles.csv")
    holdout_file = os.path.join(args.save_dir, "poly_holdout_smiles.csv")
    with open(pretrain_file, 'w', newline='') as pf:
        writer = csv.writer(pf); writer.writerow(["smiles"])
        for s in pretrain_smiles: writer.writerow([s])
    with open(holdout_file, 'w', newline='') as hf:
        writer = csv.writer(hf); writer.writerow(["smiles"])
        for s in holdout_smiles: writer.writerow([s])
    logger.info(f"Saved split SMILES to {pretrain_file} (pretrain) and {holdout_file} (hold-out).")

# Build dataset of MolGraphs for pretraining
dataset = []
bad_entries = []
logger.info("Parsing polymer structures and featurizing molecules...")
for smiles in pretrain_smiles:
    orig_smiles = smiles
    # Polymer parsing
    if args.polymer:
        # Remove degree of polymerization (Xn) if present
        if '~' in smiles:
            parts = smiles.split('~')
            smiles = parts[0]
            try:
                Xn_val = float(parts[1])
                logger.info(f"Ignoring degree of polymerization (~{Xn_val}) in input: {orig_smiles}")
            except:
                logger.info(f"Ignoring degree of polymerization in input: {orig_smiles}")
        # Split into monomers and adjacency
        sections = smiles.split('|')
        monomer_section = sections[0] if sections else smiles
        monomer_smiles_list = monomer_section.split('.') if monomer_section else []
        # Adjacency info (starts with '<' in sections)
        adjacency_str = ''
        for sec in sections[1:]:
            if sec.startswith('<'):
                adjacency_str = sec
                break
        if not adjacency_str and len(sections) > 1:
            # If adjacency is concatenated to last fraction part
            last = sections[-1]
            if '<' in last:
                idx = last.index('<')
                adjacency_str = last[idx:]
        # Build RDKit Mol for combined monomers
        monomer_mols = []
        parse_fail = False
        for mono in monomer_smiles_list:
            mol = Chem.MolFromSmiles(mono)
            if mol is None:
                logger.warning(f"Malformed monomer SMILES '{mono}' in polymer: {orig_smiles}. Skipping.")
                parse_fail = True
                break
            monomer_mols.append(mol)
        if parse_fail or not monomer_mols:
            bad_entries.append(orig_smiles); continue
        combined_mol = monomer_mols[0]
        for m in monomer_mols[1:]:
            combined_mol = Chem.CombineMols(combined_mol, m)
        directed_weights = {}  # {(id_from, id_to): weight}
        if adjacency_str:
            # Parse adjacency entries
            adj_entries = adjacency_str.split('<')
            adj_entries = [e for e in adj_entries if e]
            rw_mol = Chem.RWMol(combined_mol)
            for entry in adj_entries:
                try:
                    ids, w1, w2 = entry.split(':')
                    a_id, b_id = ids.split('-')
                    a_id, b_id = int(a_id), int(b_id)
                    w1, w2 = float(w1), float(w2)
                except:
                    continue
                directed_weights[(a_id, b_id)] = w1
                directed_weights[(b_id, a_id)] = w2
                # Find atoms by map number and add bond
                atom_a = None; atom_b = None
                for atom in rw_mol.GetAtoms():
                    if atom.GetAtomMapNum() == a_id: atom_a = atom
                    if atom.GetAtomMapNum() == b_id: atom_b = atom
                if atom_a is None or atom_b is None:
                    continue
                try:
                    rw_mol.AddBond(atom_a.GetIdx(), atom_b.GetIdx(), Chem.BondType.SINGLE)
                except Exception as e:
                    logger.warning(f"Failed to add bond between [{a_id}] and [{b_id}] for {orig_smiles}: {e}")
            combined_mol = rw_mol.GetMol()
        # Remove mapping numbers and sanitize
        for atom in combined_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        try:
            Chem.SanitizeMol(combined_mol)
        except Chem.KekulizeException:
            try:
                Chem.SanitizeMol(combined_mol, Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except Exception as e:
                logger.warning(f"Sanitization failed for polymer: {orig_smiles} (error: {e}). Skipping.")
                bad_entries.append(orig_smiles); continue
        mol = combined_mol
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Skipping malformed SMILES: {orig_smiles}")
            bad_entries.append(orig_smiles); continue
        directed_weights = {}
    # Skip if no real atoms (should have been filtered earlier)
    if all(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        logger.warning(f"No real atoms in polymer (only wildcards): {orig_smiles}. Skipping.")
        bad_entries.append(orig_smiles); continue
    # Create MolGraph
    try:
        mol_graph = MolGraph(mol)
    except Exception as e:
        logger.warning(f"Failed to featurize molecule: {orig_smiles}. Error: {e}")
        bad_entries.append(orig_smiles); continue
    # Collect atomic numbers and mapping (if polymer)
    atom_nums = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    atom_map_nums = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    # Count monomers (for graph-level task)
    monomer_count = len(monomer_smiles_list) if args.polymer else 1
    # Collect bond types (for edge-level labels)
    bond_types = [bond.GetBondType() for bond in mol.GetBonds()]
    dataset.append((mol_graph, atom_map_nums, directed_weights, atom_nums, monomer_count, bond_types))

if bad_entries:
    logger.warning(f"Skipped {len(bad_entries)} malformed entries.")
    bad_path = os.path.join(args.save_dir, "skipped_smiles.csv")
    with open(bad_path, 'w', newline='') as bf:
        writer = csv.writer(bf); writer.writerow(["skipped_smiles"])
        for s in bad_entries: writer.writerow([s])
    logger.info(f"List of skipped SMILES saved to {bad_path}.")

if len(dataset) == 0:
    raise RuntimeError("No valid data for pretraining after filtering. Aborting.")

# Determine feature dimensions from chemprop (should include dummy atom handling)
atom_fdim = get_atom_fdim() if 'get_atom_fdim' in globals() else dataset[0][0].atom_features.shape[1]
bond_fdim = get_bond_fdim() if 'get_bond_fdim' in globals() else dataset[0][0].bond_features.shape[1]
hidden_size = 300  # Hidden size for wD-MPNN encoder (using 300 per Coley et al.)

# Prepare classification label mappings
# Node (atom type) classes:
unique_atom_nums = set()
for _, _, _, atom_nums, _, _ in dataset:
    for num in atom_nums:
        if num != 0: unique_atom_nums.add(num)
unique_atom_nums = sorted(unique_atom_nums)
element_to_class = {elem: idx for idx, elem in enumerate(unique_atom_nums)}
node_classes = len(unique_atom_nums)
logger.info(f"Node (atom) classes: {node_classes} types (excluding dummy atoms).")

# Edge (bond type) classes:
bond_type_set = set()
for _, _, _, _, _, bond_types in dataset:
    bond_type_set.update(bond_types)
# Define mapping for bond types
bond_type_to_class = {}
for bt in bond_type_set:
    bond_type_to_class[bt] = len(bond_type_to_class)
edge_classes = len(bond_type_to_class)
logger.info(f"Edge (bond) classes: {edge_classes} types.")

# Graph (monomer count) classes:
monomer_counts = [mc for *_, mc, _ in dataset]
max_count = max(monomer_counts)
if max_count < 2:
    graph_classes = 1
elif max_count < 3:
    graph_classes = 2
else:
    graph_classes = 3
logger.info(f"Graph (monomer count) classes: {graph_classes} categories (1, 2, >=3 monomers).")

# Define model components
class EncoderWDMPNN(nn.Module):
    def __init__(self, atom_dim, bond_dim, hidden_dim):
        super(EncoderWDMPNN, self).__init__()
        self.W_i = nn.Linear(atom_dim + bond_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(atom_dim + hidden_dim, hidden_dim, bias=True)
        self.act = nn.ReLU()
    def forward(self, batch_graph: BatchMolGraph, weight: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        V = batch_graph.V          # atom features (N_atoms x atom_fdim)
        E = batch_graph.E          # bond features (N_bonds x bond_fdim)
        edge_index = batch_graph.edge_index      # (2, N_bonds)
        rev_edge_index = batch_graph.rev_edge_index  # (N_bonds,)
        atom_batch = batch_graph.batch           # (N_atoms,)
        N_atoms = V.size(0)
        N_bonds = E.size(0)
        if weight is None:
            weight = torch.ones(N_bonds, device=V.device)
        if N_bonds == 0:
            # No bonds: just transform atom features directly
            h_atom = self.act(self.W_o(torch.cat([V, torch.zeros((N_atoms, self.W_h.in_features), device=V.device)], dim=1)))
            return h_atom, atom_batch, torch.zeros((0, ), device=V.device)
        src = edge_index[0]
        dest = edge_index[1]
        # Initial bond hidden states h^0 (for each directed bond)
        inp = torch.cat([V[src], E], dim=1)    # (N_bonds, atom_dim + bond_dim)
        h0 = self.act(self.W_i(inp))          # (N_bonds, hidden_dim)
        h = h0.clone()
        # Message passing
        T = 3
        for t in range(T):
            # Sum incoming messages to each atom (apply weights)
            m_to_atom = torch.zeros((N_atoms, h.size(1)), device=h.device)
            m_to_atom.index_add_(0, dest, h * weight.unsqueeze(1))
            # Compute message for each bond's source atom, excluding reverse
            m_to_src = m_to_atom[src]
            h_rev = h[rev_edge_index]
            weight_rev = weight[rev_edge_index]
            rev_contrib = weight_rev.unsqueeze(1) * h_rev
            m = m_to_src - rev_contrib
            # Update bond hidden states
            h = self.act(self.W_h(m) + h0)
        # Final atom embeddings
        m_to_atom_final = torch.zeros((N_atoms, h.size(1)), device=h.device)
        m_to_atom_final.index_add_(0, dest, h)
        atom_in = torch.cat([V, m_to_atom_final], dim=1)
        h_atom = self.act(self.W_o(atom_in))
        return h_atom, atom_batch, h

class SSLModel(nn.Module):
    def __init__(self, atom_dim, bond_dim, hidden_dim, node_classes, edge_classes, graph_classes):
        super(SSLModel, self).__init__()
        self.encoder = EncoderWDMPNN(atom_dim, bond_dim, hidden_dim)
        # Heads
        self.node_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, node_classes))
        self.edge_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, edge_classes))
        self.graph_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, graph_classes))
    def forward(self, batch_graph: BatchMolGraph, weight: torch.Tensor = None):
        # Encoder forward pass
        h_atom, atom_batch, h_bond = self.encoder(batch_graph, weight)
        # Graph (molecular) embeddings by summing atom embeddings per molecule
        batch_size = int(atom_batch.max().item()) + 1 if atom_batch.numel() > 0 else 0
        if batch_size == 0:
            return h_atom, atom_batch, torch.empty(0), h_bond
        mol_vecs = torch.zeros((batch_size, h_atom.size(1)), device=h_atom.device)
        mol_vecs.index_add_(0, atom_batch, h_atom)
        return h_atom, atom_batch, mol_vecs, h_bond

# Initialize model and optimizer
model = SSLModel(atom_fdim, bond_fdim, hidden_size, node_classes, edge_classes, graph_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
logger.info("Starting SSL pretraining...")
for epoch in range(1, args.epochs + 1):
    model.train()
    random.shuffle(dataset)
    total_node_loss = 0.0
    total_edge_loss = 0.0
    total_graph_loss = 0.0
    batch_count = 0
    for i in range(0, len(dataset), args.batch_size):
        batch_entries = dataset[i: i + args.batch_size]
        batch_size = len(batch_entries)
        # Prepare BatchMolGraph
        mol_graphs = [entry[0] for entry in batch_entries]
        batch_graph = BatchMolGraph(mol_graphs)
        batch_graph = batch_graph.to(device)
        # Build weight tensor for directed bonds in this batch
        N_bonds = batch_graph.edge_index.size(1)
        weight = torch.ones(N_bonds, device=device)
        if args.polymer:
            # Assign weights for polymer edges
            bond_scopes = batch_graph.b_scope if hasattr(batch_graph, 'b_scope') else None
            atom_scopes = batch_graph.a_scope if hasattr(batch_graph, 'a_scope') else None
            # If scopes are available:
            if bond_scopes and atom_scopes:
                for mol_idx, (start_bond, num_bond) in enumerate(bond_scopes):
                    if num_bond == 0: 
                        continue
                    start_atom, num_atom = atom_scopes[mol_idx]
                    atom_map_nums = batch_entries[mol_idx][1]
                    directed_w = batch_entries[mol_idx][2]
                    # Iterate directed bonds in this molecule
                    for j in range(num_bond):
                        global_edge_idx = start_bond + j
                        u = batch_graph.edge_index[0, global_edge_idx].item()
                        v = batch_graph.edge_index[1, global_edge_idx].item()
                        u_local = u - start_atom
                        v_local = v - start_atom
                        w_val = 1.0
                        if 0 <= u_local < num_atom and 0 <= v_local < num_atom:
                            map_u = atom_map_nums[u_local]
                            map_v = atom_map_nums[v_local]
                            if map_u > 0 and map_v > 0 and directed_w:
                                w_val = directed_w.get((map_u, map_v), 1.0)
                        weight[global_edge_idx] = w_val
            else:
                # Fallback: assign weight=1 for all (if no directed weight info, which means no polymer edges)
                weight = torch.ones(N_bonds, device=device)
        # Forward pass
        h_atom, atom_batch_idx, mol_vecs, h_bond = model(batch_graph, weight)
        # Determine masked nodes and edges for this batch
        masked_node_indices = []
        masked_node_targets = []
        masked_edge_embs = []
        masked_edge_targets = []
        graph_targets = []
        # Use scopes to iterate molecules in batch
        for mol_idx, entry in enumerate(batch_entries):
            start_atom, num_atom = batch_graph.a_scope[mol_idx]
            start_bond, num_bond = batch_graph.b_scope[mol_idx]
            atom_nums = entry[3]
            # Node mask: choose a random real atom
            valid_atoms = [idx for idx, num in enumerate(atom_nums) if num != 0]
            if valid_atoms:
                local_idx = random.choice(valid_atoms)
                global_idx = start_atom + local_idx
                masked_node_indices.append(global_idx)
                masked_node_targets.append(element_to_class[atom_nums[local_idx]])
            else:
                logger.warning(f"No valid atom to mask for molecule {mol_idx} in batch (SMILES: {smiles_all[0]})")
            # Edge mask: choose a random bond (if any)
            if num_bond > 0:
                # Represent actual bonds by first directed edge index
                actual_bonds = []
                visited = set()
                for j in range(num_bond):
                    global_edge = start_bond + j
                    rev_edge = batch_graph.rev_edge_index[global_edge].item()
                    if global_edge in visited or rev_edge in visited:
                        continue
                    visited.add(global_edge); visited.add(rev_edge)
                    actual_bonds.append(global_edge)
                if actual_bonds:
                    chosen_edge = random.choice(actual_bonds)
                    bt_list = entry[5]
                    # Determine actual bond index from directed index
                    local_edge_idx = chosen_edge - start_bond
                    bond_idx = local_edge_idx // 2 if bt_list else None
                    if bond_idx is not None and bond_idx < len(bt_list):
                        bond_type = bt_list[bond_idx]
                        masked_edge_targets.append(bond_type_to_class[bond_type])
                    else:
                        masked_edge_targets.append(0)  # default to class 0 if unknown
                    # Compute edge embedding (average of two directions)
                    emb = h_bond[chosen_edge]
                    rev_idx = batch_graph.rev_edge_index[chosen_edge].item()
                    if rev_idx != chosen_edge:
                        emb = (emb + h_bond[rev_idx]) / 2
                    masked_edge_embs.append(emb)
        # Prepare target tensors
        node_pred_tensor = None
        edge_pred_tensor = None
        graph_pred_tensor = None
        node_loss = torch.tensor(0.0, device=device)
        edge_loss = torch.tensor(0.0, device=device)
        graph_loss = torch.tensor(0.0, device=device)
        if masked_node_indices:
            masked_node_indices = torch.tensor(masked_node_indices, device=device, dtype=torch.long)
            node_pred_tensor = model.node_head(h_atom[masked_node_indices])
            node_targets_tensor = torch.tensor(masked_node_targets, device=device, dtype=torch.long)
            node_loss = criterion(node_pred_tensor, node_targets_tensor)
        if masked_edge_embs:
            edge_pred_tensor = model.edge_head(torch.stack(masked_edge_embs))
            edge_targets_tensor = torch.tensor(masked_edge_targets, device=device, dtype=torch.long)
            edge_loss = criterion(edge_pred_tensor, edge_targets_tensor)
        # Graph predictions for all molecules in batch
        graph_pred_tensor = model.graph_head(mol_vecs)
        for entry in batch_entries:
            mc = entry[4]
            if mc <= 1: graph_targets.append(0)
            elif mc == 2: graph_targets.append(1)
            else: graph_targets.append(2 if graph_classes >= 3 else 1)
        graph_targets_tensor = torch.tensor(graph_targets, device=device, dtype=torch.long)
        graph_loss = criterion(graph_pred_tensor, graph_targets_tensor)
        # Backpropagation
        loss = node_loss + edge_loss + graph_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Accumulate loss values
        total_node_loss += node_loss.item()
        total_edge_loss += edge_loss.item()
        total_graph_loss += graph_loss.item()
        batch_count += 1
    # Log epoch loss
    avg_node_loss = total_node_loss / batch_count if batch_count > 0 else 0.0
    avg_edge_loss = total_edge_loss / batch_count if batch_count > 0 else 0.0
    avg_graph_loss = total_graph_loss / batch_count if batch_count > 0 else 0.0
    logger.info(f"Epoch {epoch}: node_loss={avg_node_loss:.4f}, edge_loss={avg_edge_loss:.4f}, graph_loss={avg_graph_loss:.4f}")

# Save the trained model and encoder checkpoint
full_model_path = os.path.join(args.save_dir, "ssl_model.pth")
encoder_model_path = os.path.join(args.save_dir, "encoder.pt")
torch.save(model.state_dict(), full_model_path)
torch.save(model.encoder.state_dict(), encoder_model_path)
logger.info(f"Saved full SSL model to {full_model_path}")
logger.info(f"Saved encoder (wD-MPNN) state dict to {encoder_model_path}")

# Example usage
# !python ssl_pretrain.py \
#   --data_path /content/drive/MyDrive/AI_MSE_Company/poly_chemprop_input_with_Xn.csv \
#   --polymer \
#   --save_dir /content/drive/MyDrive/AI_MSE_Company/ssl_chemprop_checkpoints \
#   --pretrain_frac 0.4 \
#   --epochs 30 \
#   --batch_size 32 \
#   --num_workers 2 \
#   --save_smiles_splits

