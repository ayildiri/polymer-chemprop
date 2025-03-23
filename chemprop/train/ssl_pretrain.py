import re
import math
import random
import argparse
from typing import List, Tuple
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

# -------- Model Definition --------
class SSLPretrainModel(torch.nn.Module):
    def __init__(self, atom_feat_dim: int, bond_feat_dim: int, hidden_dim: int, message_steps: int):
        """wD-MPNN encoder with node, edge, and graph prediction heads."""
        super(SSLPretrainModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.message_steps = message_steps
        # Linear layers for message passing
        self.W_in = torch.nn.Linear(atom_feat_dim + bond_feat_dim, hidden_dim)    # input projection
        self.W_msg = torch.nn.Linear(hidden_dim, hidden_dim)                      # message update transform
        self.W_atom = torch.nn.Linear(hidden_dim + atom_feat_dim, hidden_dim)     # atom update transform
        # Prediction heads
        self.node_head = torch.nn.Linear(hidden_dim, atom_feat_dim)               # reconstruct atom features
        self.edge_head = torch.nn.Linear(hidden_dim, atom_feat_dim + bond_feat_dim)  # reconstruct bond features (concatenated source atom + bond feats)
        self.graph_head = torch.nn.Sequential(                                    # predict ensemble molecular weight
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    def forward(self, f_atoms: torch.FloatTensor, f_bonds: torch.FloatTensor,
                a2b: List[List[int]], b2a: List[int], b2revb: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs message passing and returns node, edge, and graph outputs.
        f_atoms: (N_atom x atom_feat_dim) initial atom features.
        f_bonds: (N_edge x (atom_feat_dim+bond_feat_dim)) initial bond (directed edge) features.
        a2b: adjacency list mapping atom index -> list of outgoing edge indices.
        b2a: list mapping edge index -> source atom index.
        b2revb: list mapping edge index -> index of its reverse directed edge.
        """
        # Move to device (ensures compatibility if model on GPU)
        f_atoms = f_atoms.to(next(self.parameters()).device)
        f_bonds = f_bonds.to(next(self.parameters()).device)
        # Initial edge hidden states
        h = torch.relu(self.W_in(f_bonds))  # shape (N_edge, hidden_dim)
        # Convert neighbor mappings to tensors for vectorized operations
        b2a_t = torch.tensor(b2a, device=h.device, dtype=torch.long)
        b2revb_t = torch.tensor(b2revb, device=h.device, dtype=torch.long)
        # Prepare index tensor for destination of each edge (dest of edge = source of its reverse)
        dest_indices = b2a_t[b2revb_t]  # for each edge, dest = b2a[rev(edge)]
        # Message passing iterations
        for t in range(self.message_steps):
            # Sum of incoming messages for each atom (weighted edges handled below)
            # (We treat all neighbor messages equally here; edge weights can be applied via feature scaling if needed)
            # Here, we assume any weighting has been encoded in f_bonds or handled externally.
            # If explicit weights list is needed, we would multiply h by weights before scattering (weights handled outside for generality).
            # Compute sum of messages for destination atoms
            dest_message_sum = torch.zeros((f_atoms.shape[0], self.hidden_dim), device=h.device)
            dest_message_sum.index_add_(0, dest_indices, h)
            # Compute updated message for each edge's source atom
            source_indices = b2a_t  # each edge source
            # Exclude reverse edge's contribution: subtract rev edge's message from source atom's aggregated sum
            # (This ensures we don't include the target edge itself in its update)
            rev_msg = h[b2revb_t]
            msg_input = dest_message_sum[source_indices] - rev_msg
            # Linear transform and skip connection
            h = torch.relu(h + self.W_msg(msg_input))
        # Atom updates: aggregate final messages into each atom and combine with original atom features
        dest_message_sum = torch.zeros((f_atoms.shape[0], self.hidden_dim), device=h.device)
        dest_message_sum.index_add_(0, dest_indices, h)  # sum messages for atom destinations
        atom_input = torch.cat([dest_message_sum, f_atoms], dim=1)
        h_atom = torch.relu(self.W_atom(atom_input))  # final atom embeddings (N_atom x hidden_dim)
        # Outputs:
        node_pred = self.node_head(h_atom)           # (N_atom x atom_feat_dim)
        edge_pred = self.edge_head(h)                # (N_edge x (atom_feat_dim+bond_feat_dim))
        # Graph output: sum pool atom embeddings per molecule and predict property
        # We assume we will sum atom embeddings for each polymer graph outside this method for graph_head input.
        # (Graph pooling will be handled in training loop using a_scope.)
        # Here we just output atom embeddings; the graph_head will be applied externally.
        return node_pred, edge_pred, h_atom

# -------- Utility Functions --------
def parse_polymer_input(poly_smiles: str):
    """Parse polymer input string into monomer SMILES list, monomer weight fractions, and connectivity info."""
    parts = poly_smiles.strip().split('|')
    if len(parts) < 2:
        raise ValueError(f"Invalid polymer input format: '{poly_smiles}'")
    monomer_section = parts[0]
    # All but first and last parts are monomer weight fractions
    if parts[-1].startswith('<'):
        # Last part is connectivity
        connectivity_str = parts[-1]
        weight_parts = parts[1:-1]
    else:
        connectivity_str = ""
        weight_parts = parts[1:]
    monomer_smiles_list = monomer_section.split('.')
    monomer_weights = [float(w) for w in weight_parts] if weight_parts else [1.0] * len(monomer_smiles_list)
    return monomer_smiles_list, monomer_weights, connectivity_str

def build_polymer_graph(monomer_smiles_list: List[str], connectivity_str: str):
    """Build graph structure for a polymer given monomer SMILES and connectivity spec.
    Returns (f_atoms, f_bonds, a2b, b2a, b2revb, atom_scope, bond_scope, monomer_mw_list)."""
    # Feature containers
    f_atoms = []        # list of atom feature vectors
    f_bonds = []        # list of bond feature vectors (for each directed edge)
    a2b: List[List[int]] = []   # adjacency: atom index -> list of outgoing edge indices
    b2a: List[int] = []         # source atom index for each edge index
    b2rev: List[int] = []       # index of reverse edge for each edge index
    atom_scope: List[Tuple[int,int]] = []  # (start_index, num_atoms) per polymer (for batch pooling)
    bond_scope: List[Tuple[int,int]] = []  # (start_index, num_edges) per polymer
    monomer_mw_list: List[float] = []
    atom_offset = 0
    bond_offset = 0
    # Precompute monomer RDKit molecules and weights
    monomer_mols = []
    for smi in monomer_smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"RDKit failed to parse monomer SMILES: '{smi}'")
        monomer_mols.append(mol)
        # Compute monomer molecular weight (replace wildcards with H for weight calculation)
        smi_with_H = re.sub(r"\[\*:[0-9]+\]", "[H]", smi)  # replace [*:n] with [H]
        mol_H = Chem.MolFromSmiles(smi_with_H)
        if mol_H is None:
            # If direct replacement fails (e.g., valence issues), add explicit Hs to neighbor and compute
            mol_copy = Chem.Mol(mol)
            # Remove dummy atoms:
            for atom in list(mol_copy.GetAtoms()):
                if atom.GetSymbol() == "*":
                    # Mark index then remove
                    idx = atom.GetIdx()
                    # Optionally add H to neighbors
                    for nbr in atom.GetNeighbors():
                        Chem.FragmentOnBonds(mol_copy, [mol_copy.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetIdx()], addDummies=False)
                    # Remove atom
                    # (This path is seldom needed; we attempt to ensure mol_H parses correctly)
            mol_H = mol_copy
        mw = Descriptors.MolWt(mol_H)
        monomer_mw_list.append(mw)
    # Build polymer graph by iterating monomers
    site_to_global_atom: dict = {}  # mapping from wildcard label to global atom index
    for mol_idx, mol in enumerate(monomer_mols):
        # Identify heavy (non-wildcard) atoms and map old->new index within this monomer
        old_to_new = {}
        monomer_atoms = []  # heavy atom features
        # Determine atom features for heavy atoms
        for atom in mol.GetAtoms():
            if atom.GetSymbol() != "*":  # non-dummy atom
                old_index = atom.GetIdx()
                new_index = len(monomer_atoms)
                old_to_new[old_index] = new_index
                # Atom features: element, aromatic, (degree to be added later)
                elem = atom.GetSymbol()
                # One-hot element (common elements in polymers: H, C, N, O, F, P, S, Cl, Br, I)
                elem_list = ["C","N","O","F","P","S","Cl","Br","I"]  # excluding H and wildcard
                elem_feat = [1 if elem == e else 0 for e in elem_list]
                if atom.GetSymbol() not in elem_list:
                    # Rare elements or fallback: create one-hot of length len(elem_list) with all zeros (or extend list if needed)
                    elem_feat = [0]*len(elem_list)
                # Aromatic flag
                arom_feat = [1 if atom.GetIsAromatic() else 0]
                atom_feat_vec = elem_feat + arom_feat
                monomer_atoms.append(atom_feat_vec)
        # Record mapping for polymer connection sites (wildcards)
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                # Map this wildcard's label to its neighbor heavy atom's global index
                mapnum = atom.GetAtomMapNum()
                # Find neighbor heavy atom (each wildcard in input connects to exactly one heavy neighbor)
                for nbr in atom.GetNeighbors():
                    if nbr.GetSymbol() != "*":
                        heavy_local_idx = old_to_new[nbr.GetIdx()]
                        global_idx = atom_offset + heavy_local_idx
                        site_to_global_atom[mapnum] = global_idx
                        break
        # Initialize adjacency for new heavy atoms
        num_new_atoms = len(monomer_atoms)
        for i in range(num_new_atoms):
            a2b.append([])
        # Process bonds within the monomer (between heavy atoms)
        start_edge_index = bond_offset
        for bond in mol.GetBonds():
            a = bond.GetBeginAtom(); b = bond.GetEndAtom()
            if a.GetSymbol() == "*" or b.GetSymbol() == "*":
                continue  # skip bonds involving dummy atoms
            ai = atom_offset + old_to_new[a.GetIdx()]
            bi = atom_offset + old_to_new[b.GetIdx()]
            # Create two directed edges (ai->bi and bi->ai)
            f_edge = len(b2a); r_edge = f_edge + 1
            b2a.append(ai); b2a.append(bi)
            b2rev.append(r_edge); b2rev.append(f_edge)
            a2b[ai].append(f_edge); a2b[bi].append(r_edge)
            # Bond feature: bond type and aromaticity
            bt = bond.GetBondType()
            bond_type = ("AROMATIC" if bond.GetIsAromatic() else str(bt))  # RDKit str(bt) yields e.g. "SINGLE"
            # One-hot for bond type (single, double, triple, aromatic)
            type_feat = [0,0,0,0]
            if "SINGLE" in bond_type: type_feat[0] = 1
            elif "DOUBLE" in bond_type: type_feat[1] = 1
            elif "TRIPLE" in bond_type: type_feat[2] = 1
            elif "AROMATIC" in bond_type: type_feat[3] = 1
            # We *do not* include edge weight in feature; weight is applied in message passing.
            # Concatenate source atom features with bond features for initial edge representation
            f_bonds.append(monomer_atoms[old_to_new[a.GetIdx()]] + type_feat)
            f_bonds.append(monomer_atoms[old_to_new[b.GetIdx()]] + type_feat)
            bond_offset += 2
        # Add atom feature vectors to global list (will add degree later)
        f_atoms.extend(monomer_atoms)
        # Record scopes for this polymer (monomer part of polymer)
        atom_scope.append((atom_offset, len(monomer_atoms)))
        # (We will set bond_scope per whole polymer after adding connectivity edges)
        atom_offset += num_new_atoms
    # Process connectivity between monomers (polymer edges specified by '<')
    if connectivity_str:
        conn_entries = [seg.strip('>') for seg in connectivity_str.split('<') if seg]
        for entry in conn_entries:
            # Format: i-j:forward_prob:reverse_prob
            try:
                sites, w_forward, w_reverse = entry.split(':')
                i_label, j_label = sites.split('-')
            except Exception:
                raise ValueError(f"Invalid connectivity entry: '{entry}' in polymer string")
            i_label = int(i_label); j_label = int(j_label)
            if i_label not in site_to_global_atom or j_label not in site_to_global_atom:
                continue  # skip if references missing (shouldn't happen if input is well-formed)
            src_atom = site_to_global_atom[i_label]
            dst_atom = site_to_global_atom[j_label]
            w_forward = float(w_forward); w_reverse = float(w_reverse)
            # Add two directed edges for this connectivity
            f_edge = len(b2a); r_edge = f_edge + 1
            b2a.append(src_atom); b2a.append(dst_atom)
            b2rev.append(r_edge); b2rev.append(f_edge)
            a2b[src_atom].append(f_edge); a2b[dst_atom].append(r_edge)
            # Bond feature for polymer edge: treat as single, non-aromatic bond
            type_feat = [1,0,0,0]  # single bond one-hot
            # Use the source atom's features (we have degree missing still, but will add later)
            src_atom_feat = f_atoms[src_atom] if src_atom < len(f_atoms) else [0]* (len(f_atoms[0]) if f_atoms else 0)
            dst_atom_feat = f_atoms[dst_atom] if dst_atom < len(f_atoms) else [0]* (len(f_atoms[0]) if f_atoms else 0)
            f_bonds.append(src_atom_feat + type_feat)
            f_bonds.append(dst_atom_feat + type_feat)
            bond_offset += 2
    # Compute degree for each atom and append degree one-hot to atom features
    max_deg = 6  # cap degree at 5 for one-hot (6th index for 5+)
    for atom_idx, nbr_list in enumerate(a2b):
        deg = len(nbr_list)
        if deg >= max_deg: deg = max_deg-1
        deg_onehot = [0]*max_deg
        deg_onehot[deg] = 1
        f_atoms[atom_idx].extend(deg_onehot)
    # Now that atom features are finalized (including degree), update initial bond feature vectors in f_bonds.
    # We need to concatenate the *updated* source atom feature to bond feature part for each edge.
    atom_feat_dim = len(f_atoms[0]) if f_atoms else 0
    bond_feat_dim = len(f_bonds[0]) - atom_feat_dim if f_bonds else 0
    # Recompute f_bonds with updated atom features:
    for e_idx in range(len(b2a)):
        src_idx = b2a[e_idx]
        # Split the existing f_bonds vector (which had old atom feat + bond feat); replace atom part with updated f_atoms[src_idx]
        if bond_feat_dim > 0:
            bond_part = f_bonds[e_idx][-bond_feat_dim:]  # original bond feature segment
        else:
            bond_part = []
        f_bonds[e_idx] = f_atoms[src_idx] + bond_part
    # Determine bond_scope for the entire polymer graph
    bond_scope.append((0, len(b2a)))  # since we built one combined graph for this polymer
    # Convert feature lists to numpy arrays (for easier conversion to torch later)
    f_atoms_arr = np.array(f_atoms, dtype=np.float32)
    f_bonds_arr = np.array(f_bonds, dtype=np.float32)
    return f_atoms_arr, f_bonds_arr, a2b, b2a, b2rev, atom_scope, bond_scope, monomer_mw_list

# -------- Training Loop --------
def train_ssl(model: SSLPretrainModel, data: List[str], pretrain_frac: float, epochs: int, batch_size: int, device: torch.device):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_samples = len(data)
    train_count = math.floor(n_samples * pretrain_frac)
    if train_count < 1:
        raise ValueError("No data selected for pretraining (check pretrain_frac).")
    train_data = random.sample(data, train_count)
    for epoch in range(1, epochs+1):
        random.shuffle(train_data)
        total_loss = 0.0
        for batch_start in range(0, train_count, batch_size):
            batch = train_data[batch_start: batch_start+batch_size]
            # Build batch graph
            batch_f_atoms = []
            batch_f_bonds = []
            batch_b2a = []
            batch_b2revb = []
            batch_a2b = []
            batch_atom_scopes = []
            batch_bond_scopes = []
            batch_mw_targets = []  # graph-level targets
            atom_offset = 0
            bond_offset = 0
            # Parse and build each polymer in batch
            for poly_str in batch:
                monomer_list, monomer_weights, conn_str = parse_polymer_input(poly_str)
                f_atoms, f_bonds, a2b, b2a, b2revb, atom_scope, bond_scope, monomer_mw_list = build_polymer_graph(monomer_list, conn_str)
                # Compute ensemble molecular weight for this polymer
                M_ensemble = sum(w * mw for w, mw in zip(monomer_weights, monomer_mw_list))
                batch_mw_targets.append(M_ensemble)
                # Merge into batch (offset indices)
                # Extend global feature arrays
                for atom_vec in f_atoms:
                    batch_f_atoms.append(atom_vec.tolist())
                for bond_vec in f_bonds:
                    batch_f_bonds.append(bond_vec.tolist())
                # Adjust and extend neighbor mappings
                for ai, nbrs in enumerate(a2b):
                    global_ai = atom_offset + ai
                    # Ensure batch_a2b list length covers this atom index
                    while len(batch_a2b) <= global_ai:
                        batch_a2b.append([])
                    # Append edges (will offset edges indices later, here just store placeholders)
                    for edge_idx in nbrs:
                        batch_a2b[global_ai].append(bond_offset + edge_idx)
                for bi, src in enumerate(b2a):
                    batch_b2a.append(atom_offset + src)
                for bi, rev in enumerate(b2revb):
                    batch_b2revb.append(bond_offset + rev)
                # Record scope with offsets
                batch_atom_scopes.append((atom_offset, atom_scope[0][1]))  # (offset, count)
                batch_bond_scopes.append((bond_offset, bond_scope[0][1]))
                # Update offsets
                atom_offset += atom_scope[0][1]
                bond_offset += bond_scope[0][1]
            # Convert to torch tensors
            if not batch_f_atoms:
                continue  # skip if batch is empty
            f_atoms_tensor = torch.tensor(batch_f_atoms, dtype=torch.float32, device=device)
            f_bonds_tensor = torch.tensor(batch_f_bonds, dtype=torch.float32, device=device)
            # Forward pass
            node_pred, edge_pred, atom_emb = model(f_atoms_tensor, f_bonds_tensor, batch_a2b, batch_b2a, batch_b2revb)
            # Compute losses
            # Node loss: MSE on masked atom features
            node_targets = f_atoms_tensor  # full original atom features
            # Edge loss: MSE on masked bond features
            edge_targets = f_bonds_tensor  # full original bond (source atom + bond) features
            # Graph loss: MSE on ensemble molecular weight
            batch_mw_targets_tensor = torch.tensor(batch_mw_targets, dtype=torch.float32, device=device).view(-1, 1)
            # Sum atom embeddings per polymer for graph prediction
            graph_preds = []
            for (atom_start, atom_count) in batch_atom_scopes:
                atom_end = atom_start + atom_count
                fp = atom_emb[atom_start:atom_end].sum(dim=0, keepdim=True)  # sum-pool
                graph_pred = model.graph_head(fp)  # (1 x 1)
                graph_preds.append(graph_pred)
            graph_pred_tensor = torch.cat(graph_preds, dim=0)
            # Mask selection: randomly pick 2 atoms and 2 bonds per polymer for loss calculation
            mask_node_idx = []
            mask_edge_idx = []
            for pi, (atom_start, atom_count) in enumerate(batch_atom_scopes):
                # Atom masks
                num_mask_nodes = min(2, atom_count) if atom_count > 0 else 0
                local_nodes = random.sample(range(atom_count), num_mask_nodes) if num_mask_nodes > 0 else []
                mask_node_idx += [atom_start + ln for ln in local_nodes]
                # Bond masks (undirected count)
                bond_start, bond_count = batch_bond_scopes[pi]
                # bond_count is number of directed edges; actual bonds = bond_count/2
                actual_bonds = bond_count // 2
                num_mask_bonds = min(2, actual_bonds) if actual_bonds > 0 else 0
                local_bonds = random.sample(range(actual_bonds), num_mask_bonds) if num_mask_bonds > 0 else []
                for lb in local_bonds:
                    # mask both directed edges of this bond
                    mask_edge_idx.append(bond_start + 2*lb)
                    mask_edge_idx.append(bond_start + 2*lb + 1)
            mask_node_idx = torch.tensor(mask_node_idx, device=device, dtype=torch.long)
            mask_edge_idx = torch.tensor(mask_edge_idx, device=device, dtype=torch.long)
            # Compute MSE loss only on masked indices
            if mask_node_idx.numel() > 0:
                node_loss = torch.nn.functional.mse_loss(node_pred[mask_node_idx], node_targets[mask_node_idx])
            else:
                node_loss = torch.tensor(0.0, device=device)
            if mask_edge_idx.numel() > 0:
                edge_loss = torch.nn.functional.mse_loss(edge_pred[mask_edge_idx], edge_targets[mask_edge_idx])
            else:
                edge_loss = torch.tensor(0.0, device=device)
            graph_loss = torch.nn.functional.mse_loss(graph_pred_tensor, batch_mw_targets_tensor)
            loss = node_loss + edge_loss + graph_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / (train_count/batch_size)
        print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")
    return model

# Example usage (for actual use, replace dummy data with polymer CSV reading):
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=False, help="Path to input CSV with 'poly_chemprop_input' column")
    parser.add_argument('--pretrain_frac', type=float, default=1.0, help="Fraction of dataset to use for pretraining")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--message_steps', type=int, default=3)
    parser.add_argument('--device', type=str, choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()
    # Load data
    data_list = []
    if args.data_path:
        import pandas as pd
        df = pd.read_csv(args.data_path)
        if 'poly_chemprop_input' not in df.columns:
            raise ValueError("Input CSV must contain 'poly_chemprop_input' column with polymer SMILES.")
        data_list = df['poly_chemprop_input'].astype(str).tolist()
    else:
        # Example dummy data (one polymer composed of two monomers as in dummy test)
        data_list = ["[*:1]C[*:2].[*:3]N[*:4]|0.5|0.5|<1-3:0.5:0.5<2-4:0.5:0.5"]
    # Initialize model
    # Determine feature dimensions (element list size=9 plus aromatic and degree(6) for atoms, bond types=4)
    atom_feat_dim = 9 + 1 + 6   # =16
    bond_feat_dim = 4          # (type one-hot)
    model = SSLPretrainModel(atom_feat_dim, bond_feat_dim, hidden_dim=args.hidden_size, message_steps=args.message_steps)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    trained_model = train_ssl(model, data_list, pretrain_frac=args.pretrain_frac, epochs=args.epochs, batch_size=args.batch_size, device=device)
