import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import get_data, get_data_from_smiles
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count
from chemprop.utils import load_args, load_checkpoint

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
    mask = torch.rand_like(tensor[:, 0]) < mask_rate
    masked_tensor = tensor.clone()
    masked_tensor[mask] = 0.0
    return masked_tensor, mask

# ----------- Main SSL Pretraining Script -------
def main():
    # Setup
    data_path = '/content/drive/MyDrive/AI_MSE_Company/poly_chemprop_input_with_Xn.csv'
    save_path = '/content/drive/MyDrive/AI_MSE_Company/ssl_checkpoints/ssl_pretrained_model.pt'
    smiles_column = 'poly_chemprop_input'
    batch_size = 32
    epochs = 10
    mask_rate = 0.15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load args and dataset
    args = TrainArgs().parse_args([
        '--data_path', data_path,
        '--dataset_type', 'regression',
        '--smiles_columns', smiles_column,
        '--target_columns', 'EA vs SHE (eV)',
        '--polymer'
    ])
    args.device = device
    data = get_data_from_smiles(
    path=args.data_path,
    smiles_columns=args.smiles_columns,
    target_columns=None,
    ignore_columns=[],
    args=args
    )
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    # Initialize model and SSL head
    model = MoleculeModel(args).to(device)
    hidden_size = args.hidden_size
    atom_fdim = model.encoder.encoder[0].atom_fdim
    bond_fdim = model.encoder.encoder[0].bond_fdim
    ssl_atom_head = SSLHead(hidden_size, atom_fdim).to(device)
    ssl_bond_head = SSLHead(hidden_size, bond_fdim).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(ssl_atom_head.parameters()) + list(ssl_bond_head.parameters()),
        lr=1e-4
    )
    loss_fn = nn.MSELoss()

    print(f"🔧 Starting SSL pretraining for {epochs} epochs")
    print(f"🧠 Total params: {param_count(model):,}")

    # Pretraining loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            mol_batch = batch.batch_graph()[0].to(device)
            atom_feats = mol_batch.f_atoms
            bond_feats = mol_batch.f_bonds

            # Encode with wD-MPNN
            atom_repr = model.encoder.encoder[0](mol_batch)

            # Masking
            masked_atoms, atom_mask = random_mask(atom_feats, mask_rate)
            masked_bonds, bond_mask = random_mask(bond_feats, mask_rate)

            # SSL Head prediction
            atom_preds = ssl_atom_head(atom_repr)
            bond_preds = ssl_bond_head(atom_repr)

            # Loss (only over masked positions)
            atom_loss = loss_fn(atom_preds[atom_mask], atom_feats[atom_mask]) if atom_mask.any() else 0
            bond_loss = loss_fn(bond_preds[bond_mask], bond_feats[bond_mask]) if bond_mask.any() else 0
            loss = atom_loss + bond_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'✅ Epoch {epoch+1} complete - Avg Loss: {epoch_loss / len(data_loader):.4f}')

    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"💾 Pretrained model saved to: {save_path}")

if __name__ == '__main__':
    main()

