import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data_from_smiles

class WDMPNNEncoder(nn.Module):
    """Weighted Directed MPNN encoder for polymer graphs (node and bond embeddings)."""
    def __init__(self, atom_fdim: int, bond_fdim: int, hidden_size: int, depth: int = 3):
        super(WDMPNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        # Define layers for message passing
        self.W_i = nn.Linear(bond_fdim, hidden_size)        # Input projection for bond features
        self.W_h = nn.Linear(hidden_size, hidden_size)      # Message transformation
        self.W_o = nn.Linear(atom_fdim + hidden_size, hidden_size)  # Output transformation for atom+message concat

    def forward(self, bmg):
        # Extract graph features from BatchMolGraph (bmg)
        # Convert atom and bond features to tensors (keeping on CPU for now)
        if isinstance(bmg.f_atoms, list):
            f_atoms = torch.tensor(bmg.f_atoms, dtype=torch.float32)
        else:
            f_atoms = bmg.f_atoms.clone().float()
        if isinstance(bmg.f_bonds, list):
            f_bonds = torch.tensor(bmg.f_bonds, dtype=torch.float32)
        else:
            f_bonds = bmg.f_bonds.clone().float()
        # Mapping from bond index to atom index (source of directed bond)
        b2a = torch.tensor(bmg.b2a, dtype=torch.long)
        # Mapping from bond index to its reverse bond index
        b2revb = torch.tensor(bmg.b2revb, dtype=torch.long)

        # Move tensors to the same device as model parameters (CPU or GPU)
        device = next(self.parameters()).device
        f_atoms = f_atoms.to(device)
        f_bonds = f_bonds.to(device)
        b2a = b2a.to(device)
        b2revb = b2revb.to(device)

        # Initial bond hidden states (apply input layer and ReLU)
        H_bonds = torch.relu(self.W_i(f_bonds))
        # Message passing for (depth-1) iterations
        for _ in range(self.depth - 1):
            # Sum incoming messages for each destination atom
            # Destination atom of bond b is the source atom of its reverse bond
            dest_indices = b2a[b2revb]       # dest atom index for each bond
            n_bonds = H_bonds.size(0)
            n_atoms = f_atoms.size(0)
            # Scatter-add all bond hidden states to their destination atom index
            sum_msgs = torch.zeros(n_atoms, self.hidden_size, device=device)
            sum_msgs.index_add_(0, dest_indices, H_bonds)
            # Gather summed messages for each bond's source atom
            source_indices = b2a             # source atom index for each bond
            msgs = sum_msgs[source_indices]
            # Subtract the reverse bond's contribution to avoid using it in message
            msgs = msgs - H_bonds[b2revb]
            # Update bond hidden states: add transformed message and apply ReLU
            H_bonds = torch.relu(H_bonds + self.W_h(msgs))
        # After message passing, compute atom hidden states by combining final messages with original atom features
        dest_indices = b2a[b2revb]
        n_atoms = f_atoms.size(0)
        sum_msgs = torch.zeros(n_atoms, self.hidden_size, device=device)
        sum_msgs.index_add_(0, dest_indices, H_bonds)
        # Concatenate original atom features with aggregated message, then transform
        H_atoms = torch.relu(self.W_o(torch.cat([f_atoms, sum_msgs], dim=1)))
        return H_atoms, H_bonds

def main():
    # Parse command-line arguments using Chemprop's TrainArgs
    args = TrainArgs().parse_args()
    # Set up device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset CSV and preprocess polymer SMILES
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file '{args.data_path}' not found.")
    df = pd.read_csv(args.data_path)
    if 'poly_chemprop_input' not in df.columns:
        raise ValueError("Column 'poly_chemprop_input' not found in input CSV.")
    smiles_list = []
    for smi in df['poly_chemprop_input']:
        if pd.isna(smi):
            continue
        smi_str = str(smi).split('|')[0]  # Strip anything after '|' (polymer metadata)
        smi_str = smi_str.strip()
        if smi_str:
            smiles_list.append(smi_str)
    if len(smiles_list) == 0:
        raise ValueError("No valid polymer SMILES found after preprocessing input column.")

    # Create MoleculeDataset from SMILES using Chemprop utility (skipping invalid entries)
    smiles_data = [[s] for s in smiles_list]  # Each entry as list for Chemprop (one molecule per datapoint)
    dataset = get_data_from_smiles(smiles=smiles_data, skip_invalid_smiles=True)
    if len(dataset) == 0:
        raise ValueError("No valid molecules could be parsed from SMILES.")

    # Set up DataLoader with collate_fn to wrap batches in MoleculeDataset
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=lambda batch: MoleculeDataset(batch))

    # Determine feature dimensions for atoms and bonds from one sample
    sample_graph = dataset[0].batch_graph()  # BatchMolGraph for a single molecule
    if hasattr(sample_graph, 'atom_fdim') and hasattr(sample_graph, 'bond_fdim'):
        atom_fdim, bond_fdim = sample_graph.atom_fdim, sample_graph.bond_fdim
    else:
        # Fallback: infer dimensions from feature lists
        atom_fdim = len(sample_graph.f_atoms[0]) if len(sample_graph.f_atoms) > 0 else 0
        bond_fdim = len(sample_graph.f_bonds[0]) if len(sample_graph.f_bonds) > 0 else 0
    if atom_fdim == 0 or bond_fdim == 0:
        raise ValueError("Failed to determine atom or bond feature dimensions from data.")

    hidden_size = getattr(args, 'hidden_size', 300)
    depth = getattr(args, 'depth', 3)

    # Initialize model: wD-MPNN encoder and two SSL prediction heads
    encoder = WDMPNNEncoder(atom_fdim=atom_fdim, bond_fdim=bond_fdim,
                             hidden_size=hidden_size, depth=depth).to(device)
    node_head = nn.Linear(hidden_size, atom_fdim).to(device)   # predicts masked atom feature vector
    edge_head = nn.Linear(hidden_size, bond_fdim).to(device)   # predicts masked bond feature vector

    optimizer = torch.optim.Adam(list(encoder.parameters()) + 
                                 list(node_head.parameters()) + 
                                 list(edge_head.parameters()), lr=0.001)
    mse_loss = nn.MSELoss()  # Mean Squared Error loss

    # Training loop
    n_epochs = getattr(args, 'epochs', 1)
    for epoch in range(1, n_epochs + 1):
        encoder.train()
        node_head.train()
        edge_head.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch in dataloader:
            mol_batch = batch.batch_graph()  # Get BatchMolGraph for the batch
            n_atoms = len(mol_batch.f_atoms)
            n_bonds = len(mol_batch.f_bonds)
            if n_atoms == 0 or n_bonds == 0:
                # Skip empty batch (unlikely, unless molecule has no atoms/bonds)
                continue

            # Determine masking counts (15%)
            num_mask_atoms = max(1, int(0.15 * n_atoms)) if n_atoms > 0 else 0
            # For bonds, treat n_bonds as count of directed bonds; approximate 15% of undirected bonds
            num_mask_bonds = 0
            if n_bonds > 0:
                # Estimate number of undirected bonds as n_bonds/2
                undirected_count = n_bonds // 2  
                num_mask_bonds = max(1, int(0.15 * undirected_count)) if undirected_count > 0 else 0

            # Randomly select mask indices
            atom_indices = list(range(n_atoms))
            bond_indices = list(range(n_bonds))
            mask_atom_indices = set(torch.randperm(n_atoms)[:num_mask_atoms].tolist()) if num_mask_atoms > 0 else set()
            mask_bond_indices = set()
            if num_mask_bonds > 0:
                # Choose undirected bonds by sampling representatives (avoid double selecting reverse)
                reps = []
                for b_idx in range(n_bonds):
                    rev_idx = mol_batch.b2revb[b_idx]
                    if rev_idx > b_idx:  # take the smaller index as representative
                        reps.append(b_idx)
                if reps:
                    reps = torch.randperm(len(reps))[:num_mask_bonds].tolist()
                    for rep_idx in reps:
                        b_idx = reps[rep_idx] if isinstance(rep_idx, int) else rep_idx
                        # Actually reps is list of indices in the 'reps' list; adjust:
                        if isinstance(rep_idx, int):
                            b_idx = rep_idx
                        rev_idx = mol_batch.b2revb[b_idx]
                        mask_bond_indices.add(b_idx)
                        mask_bond_indices.add(rev_idx)
            # If an atom is masked, also mask all bonds emanating from it to prevent information leakage
            for atom_idx in mask_atom_indices:
                for b_idx, src_atom in enumerate(mol_batch.b2a):
                    if src_atom == atom_idx:
                        mask_bond_indices.add(b_idx)
                        mask_bond_indices.add(mol_batch.b2revb[b_idx])

            # Store original features for masked atoms and bonds, then mask them
            orig_atom_features = []
            for idx in mask_atom_indices:
                orig_atom_features.append(list(mol_batch.f_atoms[idx]))
                mol_batch.f_atoms[idx] = [0.0] * len(mol_batch.f_atoms[idx])  # mask atom feature vector
            orig_bond_features = []
            for idx in mask_bond_indices:
                orig_bond_features.append(list(mol_batch.f_bonds[idx]))
                mol_batch.f_bonds[idx] = [0.0] * len(mol_batch.f_bonds[idx])  # mask bond feature vector

            # Forward pass through encoder and SSL heads
            atom_emb, bond_emb = encoder(mol_batch)            # wD-MPNN encoder outputs
            pred_atom_features = node_head(atom_emb)          # predicted atom feature vectors
            pred_bond_features = edge_head(bond_emb)          # predicted bond feature vectors

            # Compute MSE loss on masked entries only
            loss = 0.0
            if mask_atom_indices:
                mask_atom_list = sorted(mask_atom_indices)
                target_atom = torch.tensor(orig_atom_features, dtype=torch.float32, device=device)
                pred_atom = pred_atom_features[mask_atom_list].to(device)
                loss_atoms = mse_loss(pred_atom, target_atom)
                loss += loss_atoms
            if mask_bond_indices:
                mask_bond_list = sorted(mask_bond_indices)
                target_bond = torch.tensor(orig_bond_features, dtype=torch.float32, device=device)
                pred_bond = pred_bond_features[mask_bond_list].to(device)
                loss_bonds = mse_loss(pred_bond, target_bond)
                loss += loss_bonds

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        print(f"Epoch {epoch}/{n_epochs} --- Average Loss: {avg_loss:.4f}")

    # Save the final model (encoder and heads) to the output directory
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'ssl_model.pt')
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'node_head_state_dict': node_head.state_dict(),
        'edge_head_state_dict': edge_head.state_dict()
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
