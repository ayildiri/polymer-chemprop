import logging
from typing import Callable

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.models import MoleculeModel
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR


def train(model: MoleculeModel,
          data_loader: MoleculeDataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.
    """
    debug = logger.debug if logger is not None else print

    model.train()
    loss_sum = iter_count = 0
    
    # Determine which schedulers need per-batch stepping
    step_per_batch = isinstance(scheduler, NoamLR) or \
                    isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR) or \
                    isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR)

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, target_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch, data_weights_batch = \
            batch.batch_graph(), batch.features(), batch.targets(), batch.atom_descriptors(), \
            batch.atom_features(), batch.bond_features(), batch.data_weights()

        mask = torch.tensor([[x is not None for x in tb] for tb in target_batch], dtype=torch.bool)
        targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        if args.target_weights is not None:
            target_weights = torch.Tensor(args.target_weights)
        else:
            target_weights = torch.ones_like(targets)
        data_weights = torch.Tensor(data_weights_batch).unsqueeze(1)

        # Run model
        model.zero_grad()
        preds = model(mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)

        # Move tensors to correct device
        torch_device = preds.device
        mask = mask.to(torch_device)
        targets = targets.to(torch_device)
        target_weights = target_weights.to(torch_device)
        data_weights = data_weights.to(torch_device)

        # Calculate losses
        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * target_weights * data_weights * mask
        elif args.dataset_type == 'spectra':
            loss = loss_func(preds, targets, mask) * target_weights * data_weights * mask
        else:
            loss = loss_func(preds, targets) * target_weights * data_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += 1

        loss.backward()
        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # Step scheduler per batch if needed
        if step_per_batch:
            scheduler.step()

        n_iter += len(batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            # Get learning rates - handle different scheduler types
            if isinstance(scheduler, NoamLR):
                lrs = scheduler.get_lr()
            else:
                lrs = scheduler.get_last_lr()
                
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum = iter_count = 0

            lrs_str = ', '.join(f'lr_{i} = {lr:.4e}' for i, lr in enumerate(lrs))
            debug(f'Loss = {loss_avg:.4e}, PNorm = {pnorm:.4f}, GNorm = {gnorm:.4f}, {lrs_str}')

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter
