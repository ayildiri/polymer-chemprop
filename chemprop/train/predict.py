from typing import List, Tuple, Union, Optional

import torch
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel


def predict(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None) -> List[List[float]]:
            return_embeddings: bool = False  # ✅ NEW
) -> Union[List[List[float]], Tuple[List[List[float]], np.ndarray]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
    """
    model.eval()

    preds = []
    graph_embeddings = []  # ✅ NEW

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch = \
            batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(), batch.bond_features()

        # Make predictions
        with torch.no_grad():
            output = model(mol_batch, features_batch, atom_descriptors_batch,
                           atom_features_batch, bond_features_batch)

            # ✅ If model returns both predictions and graph embeddings
            if return_embeddings:
                if isinstance(output, tuple) and len(output) == 2:
                    batch_preds, batch_embeds = output
                    graph_embeddings.append(batch_embeds.cpu().numpy())
                else:
                    batch_preds = output  # fallback if embeddings aren't returned
            else:
                batch_preds = output

        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    if return_embeddings:
        import numpy as np
        graph_embeddings = np.concatenate(graph_embeddings, axis=0)
        return preds, graph_embeddings
    else:
        return preds
