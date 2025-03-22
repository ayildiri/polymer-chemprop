import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count
from chemprop.utils import load_args, load_checkpoint
from chemprop.data.utils import get_data_from_smiles  # ✅ ADD THIS



# ----------- SSL-specific modules -------------
class SSLHead(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(SSLHead, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.predictor(x)


# ----------- Masking Utilities -----------------
def random_mask(tensor, mask_rate):
    """Randomly masks part of a tensor by zeroing."""
    mask = torch.rand(tensor.size(0), device=tensor.device) < mask_rate
    masked_tensor = tensor.clone()
    masked_tensor[mask] = 0.0
    return masked_tensor, mask


# ----------- Main SSL Pretraining Script -------
def main():
    # Parse training arguments
    args = TrainArgs().parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset from CSV

    # Load SMILES from CSV
    df = pd.read_csv(args.data_path)
    smiles = [[str(s)] for s in df[args.smiles_columns[0]].tolist()]
    
    # Create MoleculeDataset using Chemprop's utility
    data = get_data_from_smiles(
        smiles=smiles,
        skip_invalid_smiles=True,
        features_generator=None
    )
    data_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Initialize model and SSL heads
    model = MoleculeModel(args).to(args.device)
    hidden_size = args.hidden_size
    atom_fdim = model.encoder.encoder[0].atom_fdim
    bond_fdim = model.encoder.encoder[0].bond_fdim
    
    ssl_atom_head = SSLHead(hidden_size, atom_fdim).to(args.device)
    ssl_bond_head = SSLHead(hidden_size, bond_fdim).to(args.device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(ssl_atom_head.parameters()) + list(ssl_bond_head.parameters()),
        lr=1e-4
    )
    loss_fn = nn.MSELoss()

    print(f"\n🔧 Starting SSL pretraining for {args.epochs} epochs")
    print(f"🧠 Total model parameters: {param_count(model):,}\n")

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch in tqdm(data_loader, desc=f'Epoch {epoch + 1}/{args.epochs}'):
            mol_batch = batch.batch_graph()[0].to(args.device)
            atom_feats = mol_batch.f_atoms
            bond_feats = mol_batch.f_bonds

            # Encode with wD-MPNN
            atom_repr = model.encoder.encoder[0](mol_batch)

            # Mask features
            masked_atoms, atom_mask = random_mask(atom_feats, mask_rate=0.15)
            masked_bonds, bond_mask = random_mask(bond_feats, mask_rate=0.15)

            # Predict masked features
            atom_preds = ssl_atom_head(atom_repr)
            bond_preds = ssl_bond_head(atom_repr)

            # Compute loss only on masked entries
            atom_loss = loss_fn(atom_preds[atom_mask], atom_feats[atom_mask]) if atom_mask.any() else 0
            bond_loss = loss_fn(bond_preds[bond_mask], bond_feats[bond_mask]) if bond_mask.any() else 0
            loss = atom_loss + bond_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f'✅ Epoch {epoch + 1} completed. Avg Loss: {epoch_loss / len(data_loader):.4f}')

    # Save model
    checkpoint_path = os.path.join(args.save_dir, 'ssl_pretrained_model.pt')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n💾 Pretrained model saved to {checkpoint_path}")


if __name__ == '__main__':
    main()
