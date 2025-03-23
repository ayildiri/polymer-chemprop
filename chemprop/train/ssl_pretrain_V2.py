import os
import math
import random
import numpy as np
import pandas as pd
from argparse import ArgumentParser, Namespace
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import Chemprop classes and utilities
from chemprop.data.utils import get_data  # data loading function
from chemprop.models import MoleculeModel  # Chemprop model class (MPNN + FFN)
from chemprop.utils import save_checkpoint  # function to save model in checkpoint format

# Parser for command-line arguments
parser = ArgumentParser()
parser.add_argument('--data_path', type=str, required=True, help='Path to input data (CSV or TXT file).')
parser.add_argument('--polymer', action='store_true', help='Flag indicating the input is polymer data with extended SMILES.')
parser.add_argument('--pretrain_frac', type=float, default=1.0, help='Fraction of data to use for pretraining (e.g., 0.4 for 40%).')
parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the pretrained model (and optional splits).')
parser.add_argument('--epochs', type=int, default=30, help='Number of pretraining epochs.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for SSL pretraining.')
parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader worker processes.')
parser.add_argument('--save_smiles_splits', action='store_true', help='If set, save train/val SMILES splits to files in save_dir.')
args = parser.parse_args()

# 1. Data Loading and Preprocessing
data_path = args.data_path
# Determine if the file has a header and identify the column with polymer (or molecule) SMILES
with open(data_path, 'r') as f:
    first_line = f.readline().strip()
# Heuristic: assume header if the first token is non-smiles (alphabetic and not too long)
tokens = first_line.split(',')
has_header = any(char.isalpha() for char in tokens[0]) and len(tokens[0]) < 20
# Read the file into a DataFrame
if data_path.endswith('.csv') or data_path.endswith('.txt'):
    df = pd.read_csv(data_path, header=0 if has_header else None)
else:
    raise ValueError("Unsupported file format: only CSV or TXT files are allowed.")
# If no header provided, assign default column name for the first column
if not has_header:
    df.columns = ['smiles'] + [f"prop_{i}" for i in range(1, len(df.columns))]
# Identify the column name that contains the polymer/molecule SMILES
smiles_col_candidates = ['smiles', 'SMILES', 'structure', 'polymer', 'poly_chemprop_input']
smiles_col = None
for col in df.columns:
    if col in smiles_col_candidates:
        smiles_col = col
        break
if smiles_col is None:
    # Default to first column if no known name is found
    smiles_col = df.columns[0]
smiles_list = df[smiles_col].astype(str).tolist()

# If polymer flag is set, process polymer SMILES:
# - Remove the optional degree of polymerization (~Xn) from each string
# - (The connectivity info after '|' is retained for graph construction)
if args.polymer:
    processed_smiles = []
    for s in smiles_list:
        # Remove anything after ~ (degree of polymerization), if present
        if '~' in s:
            s = s.split('~')[0]
        processed_smiles.append(s)
    smiles_list = processed_smiles

# If using only a fraction of data for pretraining, sample that fraction
total_data = len(smiles_list)
if args.pretrain_frac < 1.0:
    use_count = math.floor(total_data * args.pretrain_frac)
    if use_count < 1:
        use_count = 1
    # Fix a seed for reproducibility of the subset
    random.seed(0)
    subset_indices = random.sample(range(total_data), use_count)
    subset_indices.sort()
    smiles_list = [smiles_list[i] for i in subset_indices]
    df = df.iloc[subset_indices].reset_index(drop=True)

# Split data into training and validation sets (e.g., 90%/10% split)
# (Validation is used to monitor SSL loss, though labels are pseudo, to avoid overfitting)
val_size = max(1, int(0.1 * len(smiles_list))) if len(smiles_list) > 1 else 0
if val_size > 0:
    random.seed(0)
    indices = list(range(len(smiles_list)))
    random.shuffle(indices)
    val_indices = set(indices[:val_size])
    train_smiles = [smiles_list[i] for i in range(len(smiles_list)) if i not in val_indices]
    val_smiles = [smiles_list[i] for i in range(len(smiles_list)) if i in val_indices]
else:
    train_smiles = smiles_list
    val_smiles = []

# Optionally save the SMILES splits to disk for reproducibility
os.makedirs(args.save_dir, exist_ok=True)
if args.save_smiles_splits:
    train_path = os.path.join(args.save_dir, "smiles_train.txt")
    val_path = os.path.join(args.save_dir, "smiles_val.txt")
    with open(train_path, 'w') as f:
        for smi in train_smiles:
            f.write(f"{smi}\n")
    if val_smiles:
        with open(val_path, 'w') as f:
            for smi in val_smiles:
                f.write(f"{smi}\n")

# Compute graph-level pseudo-labels: ensemble polymer molecular weight
graph_labels = []
# Precompute atomic masses for efficiency (H through major elements, fallback to periodic table via RDKit if available)
try:
    from rdkit.Chem import Descriptors, MolFromSmiles
except ImportError:
    MolFromSmiles = None

for smi in smiles_list:
    # Calculate ensemble molecular weight = sum(weight of each monomer * its fraction)
    ensemble_weight = 0.0
    if args.polymer and '|' in smi:
        # Split polymer string into monomer part and ratio part
        main_part, *rest = smi.split('|')
        monomer_smiles = main_part  # part before first '|'
        # Extract numeric ratio values from the rest (ignore connectivity descriptors with '<')
        ratios = []
        for token in rest:
            if token == '' or '<' in token:
                break  # stop at connectivity info
            try:
                ratios.append(float(token))
            except:
                pass
        if not ratios:
            ratios = [1.0]  # if no ratios provided, assume single monomer with fraction 1
        # In case of a single monomer, ensure ratio list length matches number of monomers (which will be 1)
        # Split monomer SMILES by '.' to handle multi-monomer ensembles
        monomers = monomer_smiles.split('.')
        if len(ratios) != len(monomers):
            # Normalize or pad ratios if needed
            ratios = ratios[:len(monomers)]
            if len(ratios) < len(monomers):
                ratios += [1.0] * (len(monomers) - len(ratios))
            total = sum(ratios)
            ratios = [r/total for r in ratios] if total > 0 else [1.0] * len(ratios)
        # Calculate molecular weight of each monomer fragment
        weights = []
        for mono in monomers:
            # Use RDKit for exact molecular weight if available
            if MolFromSmiles:
                mol = MolFromSmiles(mono)
            else:
                mol = None
            if mol:
                # Calculate molecular weight excluding dummy atoms (symbol '*')
                mass = 0.0
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() != '*':
                        mass += atom.GetMass()
                weights.append(mass)
            else:
                # Fallback: approximate mass by summing atomic masses from symbols
                # (Assume uppercase letters as element start, handle simple cases)
                mass = 0.0
                elem = ""
                for char in mono:
                    if char.isalpha():
                        if char.isupper():
                            # new element symbol starts
                            if elem:
                                # add previous element mass
                                # simple periodic table subset
                                mass += {"H":1.008,"B":10.81,"C":12.01,"N":14.01,"O":16.00,"F":19.00,"P":30.97,"S":32.06,"Cl":35.45}.get(elem, 0.0)
                            elem = char
                        else:
                            # lowercase letter, continue element symbol
                            elem += char
                    elif char.isdigit():
                        continue  # skip digits (would handle count, but assume monomer SMILES explicitly list all atoms)
                    else:
                        # wildcard or bond, flush current element
                        if elem:
                            mass += {"H":1.008,"B":10.81,"C":12.01,"N":14.01,"O":16.00,"F":19.00,"P":30.97,"S":32.06,"Cl":35.45}.get(elem, 0.0)
                        elem = ""
                if elem:
                    mass += {"H":1.008,"B":10.81,"C":12.01,"N":14.01,"O":16.00,"F":19.00,"P":30.97,"S":32.06,"Cl":35.45}.get(elem, 0.0)
                weights.append(mass)
        # Compute weighted sum of monomer masses
        ensemble_weight = sum(w * r for w, r in zip(weights, ratios))
    else:
        # Non-polymer or no '|' present: treat the whole SMILES as one molecule
        if MolFromSmiles:
            mol = MolFromSmiles(smi)
        else:
            mol = None
        mass = 0.0
        if mol:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() != '*':
                    mass += atom.GetMass()
        else:
            # Approximate for simple molecules
            for ch in smi:
                if ch.isalpha() and ch.isupper():
                    mass += {"H":1.008,"B":10.81,"C":12.01,"N":14.01,"O":16.00,"F":19.00,"P":30.97,"S":32.06,"Cl":35.45}.get(ch, 0.0)
        ensemble_weight = mass
    graph_labels.append(ensemble_weight)
graph_labels = torch.tensor(graph_labels, dtype=torch.float32)

# 2. Define the SSL Pretraining Model (Encoder + Multi-head outputs)
class SSLPretrainModel(nn.Module):
    def __init__(self, atom_fdim, bond_fdim, hidden_size, depth):
        super(SSLPretrainModel, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        # Linear layers for message passing
        # W_i: transforms combined atom+bond features to hidden state
        self.W_initial = nn.Linear(bond_fdim, hidden_size)
        # W_h: transforms aggregated messages to new bond state (shared across message passing steps)
        self.W_message = nn.Linear(hidden_size, hidden_size)
        # Atom feature transform (to incorporate atom features at node level)
        self.W_atom = nn.Linear(atom_fdim, hidden_size)
        # Output heads
        self.node_head = nn.Linear(hidden_size, atom_fdim)   # predict original atom features
        self.edge_head = nn.Linear(hidden_size, bond_fdim - atom_fdim)  # predict original bond features (excluding attached atom part)
        self.graph_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        # Initialize weights (following default initialization in PyTorch)
    
    def forward(self, batch_graph):
        """Forward pass that returns node, edge, and graph predictions."""
        # Get graph attributes from BatchMolGraph
        # We assume BatchMolGraph provides:
        # atom_features: (N_atoms, atom_fdim)
        # bond_features: (N_bonds, bond_fdim) - combined atom/bond features for each directed bond
        # b2a: list or tensor mapping each bond index -> source atom index
        # a2b: list of lists mapping each atom index -> incoming bond indices
        atom_fea = torch.FloatTensor(batch_graph.f_atoms)  # shape (N_atoms, atom_fdim)
        bond_fea = torch.FloatTensor(batch_graph.f_bonds)  # shape (N_bonds, bond_fdim)
        b2a = torch.LongTensor(batch_graph.b2a)           # (N_bonds,) source atom indices
        a2b = batch_graph.a2b  # list of incoming bond lists per atom (length N_atoms)

        atom_fea = atom_fea.to(self.W_initial.weight.device)
        bond_fea = bond_fea.to(self.W_initial.weight.device)
        b2a = b2a.to(self.W_initial.weight.device)
        # Initialize bond hidden states
        bond_hidden = self.W_initial(bond_fea)  # (N_bonds, hidden_size)
        bond_hidden = torch.relu(bond_hidden)
        # Message passing iterations
        # Precompute reverse bond indices for each bond (to exclude reverse message)
        num_bonds = bond_hidden.size(0)
        # Assume bonds are stored in pairs (i, i^1 are reverse of each other)
        rev_idx = torch.arange(num_bonds, device=b2a.device)
        rev_idx = torch.where(rev_idx % 2 == 0, rev_idx+1, rev_idx-1)
        # Message passing loop
        for t in range(self.depth):
            # Sum incoming messages for each atom (source aggregation)
            # We scatter-add bond messages to source atoms
            source_atoms = b2a  # each bond's source atom
            atom_message_sum = torch.zeros((atom_fea.size(0), self.hidden_size), device=b2a.device)
            atom_message_sum = atom_message_sum.index_add(0, source_atoms, bond_hidden)
            # Now for each bond, get the sum at its source atom and subtract that bond's reverse message
            source_sum = atom_message_sum[b2a]            # (N_bonds, hidden) sum of messages at source atom of each bond
            rev_messages = bond_hidden[rev_idx]           # corresponding reverse bond hidden state for each bond
            message = source_sum - rev_messages          # exclude reverse bond contribution
            # Update bond hidden states
            bond_hidden = self.W_message(message)
            bond_hidden = torch.relu(bond_hidden)
        # After message passing, compute atom embeddings by aggregating incoming bond messages at each atom
        atom_message_sum = torch.zeros((atom_fea.shape[0], self.hidden_size), device=b2a.device)
        atom_message_sum = atom_message_sum.index_add(0, b2a, bond_hidden)
        atom_hidden = torch.relu(self.W_atom(atom_fea) + atom_message_sum)
        # Pool atom embeddings to get graph embeddings for each molecule in batch
        mol_vecs = []
        for start, length in batch_graph.a_scope:  # a_scope: list of (start_index, length) for atoms in each molecule
            if length == 0:
                mol_vec = torch.zeros(self.hidden_size, device=b2a.device)
            else:
                atom_indices = torch.arange(start, start+length, device=b2a.device, dtype=torch.long)
                mol_vec = atom_hidden[atom_indices].sum(dim=0)
            mol_vecs.append(mol_vec)
        mol_repr = torch.stack(mol_vecs, dim=0)  # (batch_size, hidden_size)
        # Output predictions
        node_pred = self.node_head(atom_hidden)           # (N_atoms, atom_fdim)
        edge_pred = self.edge_head(bond_hidden)           # (N_bonds, bond_fdim - atom_fdim)
        graph_pred = self.graph_head(mol_repr).squeeze(-1)  # (batch_size,)
        return node_pred, edge_pred, graph_pred

# Prepare data for PyTorch DataLoader using Chemprop's MoleculeDataset
# We use Chemprop's get_data to parse smiles into its MoleculeDatapoint objects (which include MolGraph building).
train_args = Namespace(
    smiles_columns=[smiles_col],
    target_columns=[],  # no real targets (self-supervised)
    dataset_type='regression',
    max_data_size=None,
    skip_invalid_smiles=False,
    seed=0,
    polymer=args.polymer
)
train_args.ignore_columns = []
train_args.features_path = None
train_args.features_generator = []
train_args.phase_features_path = None
train_args.atom_descriptors = None
train_args.atom_descriptors_path = None
train_args.bond_descriptors = None
train_args.max_data_size = None
train_args.cache_cutoff = 1e7
train_args.no_cache = True
train_args.smiles_columns = ["smiles"]
train_args.target_columns = []
train_args.number_of_molecules = 1
train_args.mol_cache_path = None
train_args.skip_invalid_smiles = True

train_data = get_data(path=data_path, args=train_args, skip_none_targets=True)
# Filter train and val sets within train_data based on our indices
train_idx_set = set(df.index[df[smiles_col].isin(train_smiles)])
val_idx_set = set(df.index[df[smiles_col].isin(val_smiles)])
train_dataset = [dp for i, dp in enumerate(train_data) if i in train_idx_set]
val_dataset = [dp for i, dp in enumerate(train_data) if i in val_idx_set]
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=lambda x: x)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=lambda x: x)

# Instantiate the SSL pretraining model
# Determine feature dimensions from one example (assuming non-empty dataset)
sample_graph = train_data[0].mol[0] if args.polymer else train_data[0].mol  # MoleculeDatapoint.mol may be a list for polymer (multicomponent)
atom_fdim = sample_graph.atom_fdim
bond_fdim = sample_graph.bond_fdim
hidden_size = 300  # use Chemprop default hidden size
depth = 3         # use Chemprop default number of message passing steps
model = SSLPretrainModel(atom_fdim, bond_fdim, hidden_size, depth)
model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 3. Self-supervised Training Loop
best_val_loss = float('inf')
best_state_dict = None

for epoch in range(1, args.epochs + 1):
    model.train()
    total_train_loss = 0.0
    for batch in train_loader:
        # batch is a list of MoleculeDatapoints; construct a BatchMolGraph for it
        batch_graph = batch[0].batch_graph() if hasattr(batch[0], 'batch_graph') else None
        if batch_graph is None:
            # Manually construct BatchMolGraph if not precomputed
            from chemprop.features import BatchMolGraph
            mol_graphs = [dp.mol[0] if isinstance(dp.mol, list) else dp.mol for dp in batch]
            batch_graph = BatchMolGraph(mol_graphs)
        # Prepare masks for random nodes and edges in the batch_graph
        num_atoms = sum([atoms for _, atoms in batch_graph.a_scope])
        num_bonds = batch_graph.num_bonds
        # Mask 2 atoms and 2 bonds per molecule (or fewer if molecule is small)
        mask_node_index = torch.zeros(num_atoms, dtype=torch.bool, device=model.graph_head[1].weight.device)
        mask_edge_index = torch.zeros(num_bonds, dtype=torch.bool, device=model.graph_head[1].weight.device)
        atom_offset = 0
        bond_offset = 0
        for (atom_start, atom_count), (bond_start, bond_count) in zip(batch_graph.a_scope, batch_graph.b_scope):
            # Pick up to 2 random atom indices in this molecule's atom range
            if atom_count > 0:
                choose_k = min(2, atom_count)
                rand_atoms = random.sample(range(atom_start, atom_start + atom_count), choose_k)
                for ai in rand_atoms:
                    mask_node_index[ai] = True
            # Pick up to 2 random bonds in this molecule's bond range
            if bond_count > 0:
                choose_k = min(2, bond_count)
                rand_bonds = random.sample(range(bond_start, bond_start + bond_count), choose_k)
                for bi in rand_bonds:
                    mask_edge_index[bi] = True
        # Zero-out (mask) the chosen atom and bond features in the BatchMolGraph
        batch_graph.f_atoms[mask_node_index.cpu().numpy()] = np.zeros(atom_fdim)
        batch_graph.f_bonds[mask_edge_index.cpu().numpy()] = np.zeros(bond_fdim)
        # Run model forward
        node_pred, edge_pred, graph_pred = model(batch_graph)
        # Get ground truth features and labels
        true_atom_features = torch.tensor(batch_graph.f_atoms, dtype=torch.float32, device=node_pred.device)
        true_bond_features = torch.tensor(batch_graph.f_bonds, dtype=torch.float32, device=edge_pred.device)
        true_graph_labels = []
        for dp in batch:
            # Find index of this datapoint in original smiles_list to fetch its graph label
            smi = dp.smiles
            idx = smiles_list.index(smi) if smi in smiles_list else None
            if idx is None:
                # fallback: if not found (should not happen normally), append 0
                true_graph_labels.append(0.0)
            else:
                true_graph_labels.append(graph_labels[idx].item())
        true_graph_labels = torch.tensor(true_graph_labels, dtype=torch.float32, device=graph_pred.device)
        # Compute losses (MSE), masking out unmasked items
        if mask_node_index.any():
            node_loss = ((node_pred[mask_node_index] - true_atom_features[mask_node_index]) ** 2).mean()
        else:
            node_loss = torch.tensor(0.0, device=node_pred.device)
        if mask_edge_index.any():
            edge_loss = ((edge_pred[mask_edge_index] - true_bond_features[mask_edge_index][:, atom_fdim:]) ** 2).mean()
        else:
            edge_loss = torch.tensor(0.0, device=edge_pred.device)
        graph_loss = ((graph_pred - true_graph_labels) ** 2).mean()
        loss = node_loss + edge_loss + graph_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * len(batch)  # accumulate weighted by batch size
    total_train_loss /= len(train_smiles)
    # Validation
    model.eval()
    val_loss = 0.0
    if val_smiles:
        with torch.no_grad():
            for batch in val_loader:
                batch_graph = batch[0].batch_graph() if hasattr(batch[0], 'batch_graph') else BatchMolGraph([dp.mol[0] if isinstance(dp.mol, list) else dp.mol for dp in batch])
                # Masking for validation (use the same procedure, though in validation we could mask deterministically or fully since we don't update)
                num_atoms = sum([atoms for _, atoms in batch_graph.a_scope])
                num_bonds = batch_graph.num_bonds
                mask_node_index = torch.zeros(num_atoms, dtype=torch.bool, device=model.graph_head[1].weight.device)
                mask_edge_index = torch.zeros(num_bonds, dtype=torch.bool, device=model.graph_head[1].weight.device)
                for (atom_start, atom_count), (bond_start, bond_count) in zip(batch_graph.a_scope, batch_graph.b_scope):
                    if atom_count > 0:
                        choose_k = min(2, atom_count)
                        for ai in random.sample(range(atom_start, atom_start + atom_count), choose_k):
                            mask_node_index[ai] = True
                    if bond_count > 0:
                        choose_k = min(2, bond_count)
                        for bi in random.sample(range(bond_start, bond_start + bond_count), choose_k):
                            mask_edge_index[bi] = True
                batch_graph.f_atoms[mask_node_index.cpu().numpy()] = np.zeros(atom_fdim)
                batch_graph.f_bonds[mask_edge_index.cpu().numpy()] = np.zeros(bond_fdim)
                node_pred, edge_pred, graph_pred = model(batch_graph)
                true_atom_features = torch.tensor(batch_graph.f_atoms, dtype=torch.float32, device=node_pred.device)
                true_bond_features = torch.tensor(batch_graph.f_bonds, dtype=torch.float32, device=edge_pred.device)
                true_graph_labels = []
                for dp in batch:
                    smi = dp.smiles
                    idx = smiles_list.index(smi) if smi in smiles_list else None
                    true_graph_labels.append(graph_labels[idx].item() if idx is not None else 0.0)
                true_graph_labels = torch.tensor(true_graph_labels, dtype=torch.float32, device=graph_pred.device)
                node_loss = ((node_pred[mask_node_index] - true_atom_features[mask_node_index]) ** 2).mean() if mask_node_index.any() else 0.0
                edge_loss = ((edge_pred[mask_edge_index] - true_bond_features[mask_edge_index][:, atom_fdim:]) ** 2).mean() if mask_edge_index.any() else 0.0
                graph_loss = ((graph_pred - true_graph_labels) ** 2).mean()
                val_loss += (node_loss + edge_loss + graph_loss).item() * len(batch)
        val_loss /= len(val_smiles)
    else:
        val_loss = total_train_loss  # if no val set, set val_loss equal to train_loss for monitoring
    print(f"Epoch {epoch}: Train SSL loss = {total_train_loss:.4f}, Val SSL loss = {val_loss:.4f}")
    # Check for best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# 4. Save the best SSL-pretrained model in Chemprop format
if best_state_dict is None:
    best_state_dict = model.state_dict()
# Load best weights into model (on CPU)
model.load_state_dict(best_state_dict)
model = model.cpu()
# Build a MoleculeModel for saving (encoder + single output head)
save_args = Namespace(
    # basic architecture hyperparams
    hidden_size=hidden_size,
    depth=depth,
    atom_messages=False,
    undirected=False,
    num_tasks=1,
    dataset_type='regression',
    # features and polymer flags
    features_generator=None,
    atom_descriptors=None,
    atom_features_size=0,
    bond_features_size=0,
    polymer=args.polymer
)
chemprop_model = MoleculeModel(save_args)
# Transfer encoder weights
cp_state = chemprop_model.state_dict()
for name, param in cp_state.items():
    # Map our SSLPretrainModel parameters to chemprop's names
    if name.startswith('encoder.mpn'):
        # Encoder weights
        if 'W_i.weight' in name:
            param.data.copy_(best_state_dict['W_initial.weight'])
        elif 'W_i.bias' in name:
            param.data.copy_(best_state_dict['W_initial.bias'])
        elif 'W_h.weight' in name:
            param.data.copy_(best_state_dict['W_message.weight'])
        elif 'W_h.bias' in name:
            param.data.copy_(best_state_dict['W_message.bias'])
        elif 'W_atom.weight' in name and 'W_atom' in best_state_dict:
            param.data.copy_(best_state_dict['W_atom.weight'])
        elif 'W_atom.bias' in name and 'W_atom' in best_state_dict:
            param.data.copy_(best_state_dict['W_atom.bias'])
        else:
            # For any other encoder params (if any), copy if present
            if name.replace('encoder.mpn.', '') in best_state_dict:
                param.data.copy_(best_state_dict[name.replace('encoder.mpn.', '')])
    elif name.startswith('ffn'):
        # Initialize Chemprop FFN weights (graph head) from our trained graph_head (use final layer)
        if name.endswith('.weight') and param.shape == best_state_dict['graph_head.3.weight'].shape:
            param.data.copy_(best_state_dict['graph_head.3.weight'])
        elif name.endswith('.bias') and param.shape == best_state_dict['graph_head.3.bias'].shape:
            param.data.copy_(best_state_dict['graph_head.3.bias'])
# Save the checkpoint
save_checkpoint(os.path.join(args.save_dir, 'model.pt'), chemprop_model, save_args)
print(f"SSL-pretrained model saved to {os.path.join(args.save_dir, 'model.pt')}")


