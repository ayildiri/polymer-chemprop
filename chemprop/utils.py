from argparse import Namespace
import numpy as np
from typing import Dict, Union
import csv
from datetime import timedelta
from functools import wraps
import logging
import math
import os
import pickle
import re
from time import time
from typing import Any, Callable, List, Tuple, Union
import collections

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from chemprop.args import PredictArgs, TrainArgs
from chemprop.data import StandardScaler, MoleculeDataset, preprocess_smiles_columns, get_task_names
from chemprop.models import MoleculeModel
from chemprop.nn_utils import NoamLR
from chemprop.spectra_utils import sid_loss, sid_metric, wasserstein_loss, wasserstein_metric


def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. :code:`isfile == True`),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str,
                    model: MoleculeModel,
                    scaler: StandardScaler = None,
                    features_scaler: StandardScaler = None,
                    atom_descriptor_scaler: StandardScaler = None,
                    bond_feature_scaler: StandardScaler = None,
                    args: TrainArgs = None) -> None:
    """
    Saves a model checkpoint in a clean Chemprop-compatible format.
    """

    def scaler_to_dict(scaler_obj):
        return {
            'means': scaler_obj.means.tolist() if isinstance(scaler_obj.means, np.ndarray) else scaler_obj.means,
            'stds': scaler_obj.stds.tolist() if isinstance(scaler_obj.stds, np.ndarray) else scaler_obj.stds,
        } if scaler_obj is not None else None

    state = {
        'args': args.as_dict(),  # ✅ Store args as dict
        'state_dict': model.state_dict(),
        'data_scaler': scaler_to_dict(scaler),
        'features_scaler': scaler_to_dict(features_scaler),
        'atom_descriptor_scaler': scaler_to_dict(atom_descriptor_scaler),
        'bond_feature_scaler': scaler_to_dict(bond_feature_scaler)
    }

    torch.save(state, path)



import torch.serialization
from chemprop.args import TrainArgs  # ✅ needed for loading args safely

def load_checkpoint(path: str, device: torch.device = None, logger=None) -> Union[Dict, MoleculeModel]:
    """
    Loads a model checkpoint. Returns either raw state_dict or a MoleculeModel.
    Supports full training checkpoints or weights-only partial checkpoints.
    """
    debug = logger.debug if logger else print
    info = logger.info if logger else print

    try:
        state = torch.load(path, map_location=device or 'cpu', weights_only=False)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load checkpoint at {path}: {e}")

    # Case 1: Weights-only checkpoint (e.g., SSL pretraining)
    if isinstance(state, dict) and 'state_dict' in state and 'args' not in state:
        return state

    # Case 2: Full training checkpoint
    if isinstance(state, dict) and 'state_dict' in state and 'args' in state:
        args_dict = state['args']
        if isinstance(args_dict, Namespace):
            args_dict = vars(args_dict)
        args = TrainArgs().from_dict(args_dict, skip_unsettable=True)

        if device is not None:
            args.device = device

        model = MoleculeModel(args)
        model_state_dict = model.state_dict()
        loaded_state_dict = state['state_dict']

        # ⚠️ Skip missing or mismatched parameters
        pretrained_state_dict = {}
        for loaded_param_name in loaded_state_dict.keys():
            if re.match(r'(encoder\.encoder\.)([Wc])', loaded_param_name):
                param_name = loaded_param_name.replace('encoder.encoder', 'encoder.encoder.0')
            else:
                param_name = loaded_param_name

            if param_name not in model_state_dict:
                info(f'Warning: Pretrained parameter "{loaded_param_name}" not found in model.')
            elif model_state_dict[param_name].shape != loaded_state_dict[loaded_param_name].shape:
                info(f'Warning: Shape mismatch for "{loaded_param_name}". '
                     f'Checkpoint: {loaded_state_dict[loaded_param_name].shape}, '
                     f'Model: {model_state_dict[param_name].shape}')
            else:
                debug(f'✅ Loading pretrained parameter "{param_name}"')
                pretrained_state_dict[param_name] = loaded_state_dict[loaded_param_name]

        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict)

        if args.cuda:
            model = model.to(args.device)

        return model

    raise ValueError(f"❌ Checkpoint at {path} is not a valid Chemprop checkpoint.")


def overwrite_state_dict(loaded_param_name: str,
                        model_param_name: str,
                        loaded_state_dict: collections.OrderedDict,
                        model_state_dict: collections.OrderedDict,
                        logger: logging.Logger = None) -> collections.OrderedDict:
    """
    Overwrites a given parameter in the current model with the loaded model.
    :param loaded_param_name: name of parameter in checkpoint model.
    :param model_param_name: name of parameter in current model.
    :param loaded_state_dict: state_dict for checkpoint model.
    :param model_state_dict: state_dict for current model.
    :param logger: A logger.
    :return: The updated state_dict for the current model. 
    """
    debug = logger.debug if logger is not None else print

    
    if model_param_name not in model_state_dict:
        debug(f'Pretrained parameter "{model_param_name}" cannot be found in model parameters.')
        
    elif model_state_dict[model_param_name].shape != loaded_state_dict[loaded_param_name].shape:
        debug(f'Pretrained parameter "{loaded_param_name}" '
              f'of shape {loaded_state_dict[loaded_param_name].shape} does not match corresponding '
              f'model parameter of shape {model_state_dict[model_param_name].shape}.')
    
    else:
        debug(f'Loading pretrained parameter "{model_param_name}".')
        model_state_dict[model_param_name] = loaded_state_dict[loaded_param_name]    
    
    return model_state_dict


def load_frzn_model(model: torch.nn,
                    path: str,
                    current_args: Namespace = None,
                    cuda: bool = None,
                    logger: logging.Logger = None) -> MoleculeModel:
    """
    Loads a model checkpoint.
    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MoleculeModel.
    """
    debug = logger.debug if logger is not None else print

    loaded_mpnn_model = torch.load(path, map_location=lambda storage, loc: storage)
    loaded_state_dict = loaded_mpnn_model['state_dict']
    loaded_args = loaded_mpnn_model['args']

    model_state_dict = model.state_dict()
    
    if loaded_args.number_of_molecules==1 & current_args.number_of_molecules==1:      
        encoder_param_names = ['encoder.encoder.0.W_i.weight', 'encoder.encoder.0.W_h.weight', 'encoder.encoder.0.W_o.weight', 'encoder.encoder.0.W_o.bias']
        
        # Only freeze encoder if checkpoint_frzn is provided and frzn_encoder is True
        if current_args.checkpoint_frzn is not None and getattr(current_args, 'frzn_encoder', False):
            # Freeze the MPNN
            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(param_name, param_name, loaded_state_dict, model_state_dict)
        
        # Handle FFN freezing separately
        if current_args.frzn_ffn_layers > 0:         
            ffn_param_names = [['ffn.'+str(i*3+1)+'.weight','ffn.'+str(i*3+1)+'.bias'] for i in range(current_args.frzn_ffn_layers)]
            ffn_param_names = [item for sublist in ffn_param_names for item in sublist]
            
            # Freeze only FFN layers
            for param_name in ffn_param_names:
                model_state_dict = overwrite_state_dict(param_name, param_name, loaded_state_dict, model_state_dict)              
            
        if current_args.freeze_first_only:
            debug(f'WARNING: --freeze_first_only flag cannot be used with number_of_molecules=1 (flag is ignored)')
            
    elif (loaded_args.number_of_molecules==1) & (current_args.number_of_molecules>1):
        
        if (current_args.checkpoint_frzn is not None) & (current_args.freeze_first_only) & (not (current_args.frzn_ffn_layers > 0)): # Only freeze first MPNN
            encoder_param_names = ['encoder.encoder.0.W_i.weight', 'encoder.encoder.0.W_h.weight', 'encoder.encoder.0.W_o.weight', 'encoder.encoder.0.W_o.bias']
            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(param_name,param_name,loaded_state_dict,model_state_dict)
                
        if (current_args.checkpoint_frzn is not None) & (not current_args.freeze_first_only) & (not (current_args.frzn_ffn_layers > 0)): # Duplicate encoder from frozen checkpoint and overwrite all encoders
            loaded_encoder_param_names = ['encoder.encoder.0.W_i.weight', 'encoder.encoder.0.W_h.weight', 'encoder.encoder.0.W_o.weight', 'encoder.encoder.0.W_o.bias']*current_args.number_of_molecules
            model_encoder_param_names = [['encoder.encoder.'+str(mol_num)+'.W_i.weight', 'encoder.encoder.'+str(mol_num)+'.W_h.weight', 'encoder.encoder.'+str(mol_num)+'.W_o.weight', 'encoder.encoder.'+str(mol_num)+'.W_o.bias'] for mol_num in range(current_args.number_of_molecules)]
            model_encoder_param_names = [item for sublist in model_encoder_param_names for item in sublist]
            for loaded_param_name,model_param_name in zip(loaded_encoder_param_names,model_encoder_param_names):
                model_state_dict = overwrite_state_dict(loaded_param_name,model_param_name,loaded_state_dict,model_state_dict)
        
        if current_args.frzn_ffn_layers > 0: # Duplicate encoder from frozen checkpoint and overwrite all encoders + FFN layers
            raise Exception ('Number of molecules in checkpoint_frzn must be equal to current model for ffn layers to be frozen')
            
    elif (loaded_args.number_of_molecules>1 )& (current_args.number_of_molecules>1):
        if (loaded_args.number_of_molecules) !=( current_args.number_of_molecules):
            raise Exception('Number of molecules in checkpoint_frzn ({}) must match current model ({}) OR equal to 1.'.format(loaded_args.number_of_molecules,current_args.number_of_molecules))
        
        if current_args.freeze_first_only:
            raise Exception('Number of molecules in checkpoint_frzn ({}) must be equal to 1 for freeze_first_only to be used.'.format(loaded_args.number_of_molecules))
       
        if (current_args.checkpoint_frzn is not None) & (not (current_args.frzn_ffn_layers > 0)):
            encoder_param_names = [['encoder.encoder.'+str(mol_num)+'.W_i.weight', 'encoder.encoder.'+str(mol_num)+'.W_h.weight', 'encoder.encoder.'+str(mol_num)+'.W_o.weight', 'encoder.encoder.'+str(mol_num)+'.W_o.bias'] for mol_num in range(current_args.number_of_molecules)]
            encoder_param_names = [item for sublist in encoder_param_names for item in sublist]
            
            for param_name in encoder_param_names:
                model_state_dict = overwrite_state_dict(param_name,param_name,loaded_state_dict,model_state_dict)
        
        if current_args.frzn_ffn_layers > 0:
                
            encoder_param_names = [['encoder.encoder.'+str(mol_num)+'.W_i.weight', 'encoder.encoder.'+str(mol_num)+'.W_h.weight', 'encoder.encoder.'+str(mol_num)+'.W_o.weight', 'encoder.encoder.'+str(mol_num)+'.W_o.bias'] for mol_num in range(current_args.number_of_molecules)]
            encoder_param_names = [item for sublist in encoder_param_names for item in sublist]            
            ffn_param_names = [['ffn.'+str(i*3+1)+'.weight','ffn.'+str(i*3+1)+'.bias'] for i in range(current_args.frzn_ffn_layers)]
            ffn_param_names = [item for sublist in ffn_param_names for item in sublist]
            
            for param_name in encoder_param_names + ffn_param_names:
                model_state_dict = overwrite_state_dict(param_name,param_name,loaded_state_dict,model_state_dict)    
        
        if current_args.frzn_ffn_layers >= current_args.ffn_num_layers:
            raise Exception('Number of frozen FFN layers must be less than the number of FFN layers')
    
    # Load pretrained weights
    model.load_state_dict(model_state_dict)
    
    return model
    
def load_scalers(path: str) -> Tuple[StandardScaler, StandardScaler, StandardScaler, StandardScaler]:
    """
    Loads the scalers a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A tuple with the data :class:`~chemprop.data.scaler.StandardScaler`
             and features :class:`~chemprop.data.scaler.StandardScaler`.
    """
    state = torch.load(path, map_location=lambda storage, loc: storage)

    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None
    features_scaler = StandardScaler(state['features_scaler']['means'],
                                     state['features_scaler']['stds'],
                                     replace_nan_token=0) if state['features_scaler'] is not None else None

    if 'atom_descriptor_scaler' in state.keys():
        atom_descriptor_scaler = StandardScaler(state['atom_descriptor_scaler']['means'],
                                                state['atom_descriptor_scaler']['stds'],
                                                replace_nan_token=0) if state['atom_descriptor_scaler'] is not None else None
    else:
        atom_descriptor_scaler = None

    if 'bond_feature_scaler' in state.keys():
        bond_feature_scaler = StandardScaler(state['bond_feature_scaler']['means'],
                                            state['bond_feature_scaler']['stds'],
                                            replace_nan_token=0) if state['bond_feature_scaler'] is not None else None
    else:
        bond_feature_scaler = None

    return scaler, features_scaler, atom_descriptor_scaler, bond_feature_scaler

def build_optimizer(model: nn.Module, args: TrainArgs) -> Optimizer:
    """
    Builds a PyTorch Optimizer.

    :param model: The model to optimize.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing optimizer arguments.
    :return: An initialized Optimizer.
    """
    weight_decay = getattr(args, 'weight_decay', 0.0)
    params = [{'params': model.parameters(), 'lr': args.init_lr, 'weight_decay': weight_decay}]

    # Choose optimizer based on args
    if getattr(args, 'optimizer', 'adam') == 'adamw':
        return torch.optim.AdamW(params)
    else:
        return Adam(params)

def load_args(path: str) -> TrainArgs:
    """
    Loads the arguments a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: The :class:`~chemprop.args.TrainArgs` object that the model was trained with.
    """
    raw_args = torch.load(path, map_location=lambda storage, loc: storage)['args']

    # 🛠️ Convert Namespace back to TrainArgs
    if isinstance(raw_args, Namespace):
        raw_args = vars(raw_args)

    args = TrainArgs().from_dict(raw_args, skip_unsettable=True)
    return args

def load_task_names(path: str) -> List[str]:
    """
    Loads the task names a model was trained with.

    :param path: Path where model checkpoint is saved.
    :return: A list of the task names that the model was trained with.
    """
    return load_args(path).task_names


def get_loss_func(args: TrainArgs) -> nn.Module:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Arguments containing the dataset type ("classification", "regression", or "multiclass").
    :return: A PyTorch loss function.
    """
    if args.alternative_loss_function is not None:
        if args.dataset_type == 'spectra' and args.alternative_loss_function == 'wasserstein':
            return wasserstein_loss
        else:
            raise ValueError(f'Alternative loss function {args.alternative_loss_function} not '
                                'supported with dataset type {args.dataset_type}.')

    if args.dataset_type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')

    if args.dataset_type == 'regression':
        return nn.MSELoss(reduction='none')

    if args.dataset_type == 'multiclass':
        return nn.CrossEntropyLoss(reduction='none')

    if args.dataset_type == 'spectra':
        return sid_loss

    raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')


def prc_auc(targets: List[int], preds: List[float]) -> float:
    """
    Computes the area under the precision-recall curve.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed prc-auc.
    """
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def bce(targets: List[int], preds: List[float]) -> float:
    """
    Computes the binary cross entropy loss.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :return: The computed binary cross entropy.
    """
    # Don't use logits because the sigmoid is added in all places except training itself
    bce_func = nn.BCELoss(reduction='mean')
    loss = bce_func(target=torch.Tensor(targets), input=torch.Tensor(preds)).item()

    return loss


def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return math.sqrt(mean_squared_error(targets, preds))


def mse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed mse.
    """
    return mean_squared_error(targets, preds)


def accuracy(targets: List[int], preds: Union[List[float], List[List[float]]], threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.

    Alternatively, computes accuracy for a multiclass prediction task by picking the largest probability.

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0.
    :return: The computed accuracy.
    """
    if type(preds[0]) == list:  # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction

    return accuracy_score(targets, hard_preds)


def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    r"""
    Gets the metric function corresponding to a given metric name.

    Supports:

    * :code:`auc`: Area under the receiver operating characteristic curve
    * :code:`prc-auc`: Area under the precision recall curve
    * :code:`rmse`: Root mean squared error
    * :code:`mse`: Mean squared error
    * :code:`mae`: Mean absolute error
    * :code:`r2`: Coefficient of determination R\ :superscript:`2`
    * :code:`accuracy`: Accuracy (using a threshold to binarize predictions)
    * :code:`cross_entropy`: Cross entropy
    * :code:`binary_cross_entropy`: Binary cross entropy

    :param metric: Metric name.
    :return: A metric function which takes as arguments a list of targets and a list of predictions and returns.
    """
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse

    if metric == 'mse':
        return mse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score

    if metric == 'accuracy':
        return accuracy

    if metric == 'cross_entropy':
        return log_loss

    if metric == 'binary_cross_entropy':
        return bce
    
    if metric == 'sid':
        return sid_metric
    
    if metric == 'wasserstein':
        return wasserstein_metric

    raise ValueError(f'Metric "{metric}" not supported.')


def build_lr_scheduler(optimizer: Optimizer, args: TrainArgs, total_epochs: List[int] = None) -> _LRScheduler:
    """
    Builds a PyTorch learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: A :class:`~chemprop.args.TrainArgs` object containing learning rate arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """
    # Get total epochs and steps per epoch
    total_epochs = total_epochs or [args.epochs] * args.num_lrs
    steps_per_epoch = args.train_data_size // args.batch_size
    
    # Choose scheduler based on args.scheduler parameter
    if getattr(args, 'scheduler', 'noam') == 'noam':
        return NoamLR(
            optimizer=optimizer,
            warmup_epochs=[args.warmup_epochs],
            total_epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            init_lr=[args.init_lr],
            max_lr=[args.max_lr],
            final_lr=[args.final_lr]
        )
    elif getattr(args, 'scheduler', 'noam') == 'constant':
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    elif getattr(args, 'scheduler', 'noam') == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs[0] * steps_per_epoch,
            eta_min=args.final_lr
        )
    elif getattr(args, 'scheduler', 'noam') == 'cyclic':
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.init_lr,
            max_lr=args.max_lr,
            step_size_up=args.warmup_epochs * steps_per_epoch,
            step_size_down=(total_epochs[0] - args.warmup_epochs) * steps_per_epoch,
            cycle_momentum=False
        )
    else:
        # Default to NoamLR for backward compatibility
        return NoamLR(
            optimizer=optimizer,
            warmup_epochs=[args.warmup_epochs],
            total_epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            init_lr=[args.init_lr],
            max_lr=[args.max_lr],
            final_lr=[args.final_lr]
        )


def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    If a logger with that name already exists, simply returns that logger.
    Otherwise, creates a new logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of :code:`quiet`.
    One file handler (:code:`verbose.log`) saves all logs, the other (:code:`quiet.log`) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e., print only important info).
    :return: The logger.
    """

    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger


def timeit(logger_name: str = None) -> Callable[[Callable], Callable]:
    """
    Creates a decorator which wraps a function with a timer that prints the elapsed time.

    :param logger_name: The name of the logger used to record output. If None, uses :code:`print` instead.
    :return: A decorator which wraps a function with a timer that prints the elapsed time.
    """
    def timeit_decorator(func: Callable) -> Callable:
        """
        A decorator which wraps a function with a timer that prints the elapsed time.

        :param func: The function to wrap with the timer.
        :return: The function wrapped with the timer.
        """
        @wraps(func)
        def wrap(*args, **kwargs) -> Any:
            start_time = time()
            result = func(*args, **kwargs)
            delta = timedelta(seconds=round(time() - start_time))
            info = logging.getLogger(logger_name).info if logger_name is not None else print
            info(f'Elapsed time = {delta}')

            return result

        return wrap

    return timeit_decorator


def save_smiles_splits(data_path: str,
                       save_dir: str,
                       task_names: List[str] = None,
                       features_path: List[str] = None,
                       train_data: MoleculeDataset = None,
                       val_data: MoleculeDataset = None,
                       test_data: MoleculeDataset = None,
                       logger: logging.Logger = None,
                       smiles_columns: List[str] = None) -> None:
    """
    Saves a csv file with train/val/test splits of target data and additional features.
    Also saves indices of train/val/test split as a pickle file. Pickle file does not support repeated entries 
    with the same SMILES or entries entered from a path other than the main data path, such as a separate test path.

    :param data_path: Path to data CSV file.
    :param save_dir: Path where pickle files will be saved.
    :param task_names: List of target names for the model as from the function get_task_names().
        If not provided, will use datafile header entries.
    :param features_path: List of path(s) to files with additional molecule features.
    :param train_data: Train :class:`~chemprop.data.data.MoleculeDataset`.
    :param val_data: Validation :class:`~chemprop.data.data.MoleculeDataset`.
    :param test_data: Test :class:`~chemprop.data.data.MoleculeDataset`.
    :param smiles_columns: The name of the column containing SMILES. By default, uses the first column.
    :param logger: A logger for recording output.
    """
    makedirs(save_dir)
    
    info = logger.info if logger is not None else print
    save_split_indices = True

    if not isinstance(smiles_columns, list):
        smiles_columns = preprocess_smiles_columns(path=data_path, smiles_columns=smiles_columns)

    with open(data_path) as f:
        reader = csv.DictReader(f)

        indices_by_smiles = {}
        for i, row in enumerate(tqdm(reader)):
            smiles = tuple([row[column] for column in smiles_columns])
            if smiles in indices_by_smiles:
                save_split_indices = False
                info('Warning: Repeated SMILES found in data, pickle file of split indices cannot distinguish entries and will not be generated.')
                break
            indices_by_smiles[smiles] = i

    if task_names is None:
        task_names = get_task_names(path=data_path, smiles_columns=smiles_columns)

    features_header = []
    if features_path is not None:
        for feat_path in features_path:
            with open(feat_path, 'r') as f:
                reader = csv.reader(f)
                feat_header = next(reader)
                features_header.extend(feat_header)

    all_split_indices = []
    for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
        if dataset is None:
            continue

        with open(os.path.join(save_dir, f'{name}_smiles.csv'), 'w') as f:
            writer = csv.writer(f)
            if smiles_columns[0] == '':
                writer.writerow(['smiles'])
            else:
                writer.writerow(smiles_columns)
            for smiles in dataset.smiles():
                writer.writerow(smiles)

        with open(os.path.join(save_dir, f'{name}_full.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(smiles_columns + task_names)
            dataset_targets = dataset.targets()
            for i, smiles in enumerate(dataset.smiles()):
                writer.writerow(smiles + dataset_targets[i])

        if features_path is not None:
            dataset_features = dataset.features()
            with open(os.path.join(save_dir, f'{name}_features.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(features_header)
                writer.writerows(dataset_features)

        if save_split_indices:
            split_indices = []
            for smiles in dataset.smiles():
                index = indices_by_smiles.get(tuple(smiles))
                if index is None:
                    save_split_indices = False
                    info(f'Warning: SMILES string in {name} could not be found in data file, and likely came from a secondary data file. '
                    'The pickle file of split indices can only indicate indices for a single file and will not be generated.')
                    break
                split_indices.append(index)
            else:
                split_indices.sort()
                all_split_indices.append(split_indices)

        if name == 'train':
            data_weights = dataset.data_weights()
            if any([w != 1 for w in data_weights]):
                with open(os.path.join(save_dir, f'{name}_weights.csv'),'w') as f:
                    writer=csv.writer(f)
                    writer.writerow(['data weights'])
                    for weight in data_weights:
                        writer.writerow([weight])

    if save_split_indices:
        with open(os.path.join(save_dir, 'split_indices.pckl'), 'wb') as f:
            pickle.dump(all_split_indices, f)


def update_prediction_args(predict_args: PredictArgs,
                           train_args: TrainArgs,
                           missing_to_defaults: bool = True,
                           validate_feature_sources: bool = True) -> None:
    """
    Updates prediction arguments with training arguments loaded from a checkpoint file.
    If an argument is present in both, the prediction argument will be used.

    Also raises errors for situations where the prediction arguments and training arguments
    are different but must match for proper function.

    :param predict_args: The :class:`~chemprop.args.PredictArgs` object containing the arguments to use for making predictions.
    :param train_args: The :class:`~chemprop.args.TrainArgs` object containing the arguments used to train the model previously.
    :param missing_to_defaults: Whether to replace missing training arguments with the current defaults for :class: `~chemprop.args.TrainArgs`.
        This is used for backwards compatibility.
    :param validate_feature_sources: Indicates whether the feature sources (from path or generator) are checked for consistency between
        the training and prediction arguments. This is not necessary for fingerprint generation, where molecule features are not used.
    """
    for key, value in vars(train_args).items():
        if not hasattr(predict_args, key):
            setattr(predict_args, key, value)

    if missing_to_defaults:
        # If a default argument would cause different behavior than occurred in legacy checkpoints before the argument existed,
        # then that argument must be included in the `override_defaults` dictionary to force the legacy behavior.
        override_defaults = {
            'bond_features_scaling':False,
            'no_bond_features_scaling':True,
            'atom_descriptors_scaling':False,
            'no_atom_descriptors_scaling':True,
        }
        default_train_args=TrainArgs().parse_args(['--data_path', None, '--dataset_type', str(train_args.dataset_type)])
        for key, value in vars(default_train_args).items():
            if not hasattr(predict_args,key):
                setattr(predict_args,key,override_defaults.get(key,value))
    
    # Same number of molecules must be used in training as in making predictions
    if train_args.number_of_molecules != predict_args.number_of_molecules:
        raise ValueError('A different number of molecules was used in training '
                        f'model than is specified for prediction, {train_args.number_of_molecules} '
                         'smiles fields must be provided')

    # If atom-descriptors were used during training, they must be used when predicting and vice-versa
    if train_args.atom_descriptors != predict_args.atom_descriptors:
        raise ValueError('The use of atom descriptors is inconsistent between training and prediction. If atom descriptors '
                         ' were used during training, they must be specified again during prediction using the same type of '
                         ' descriptors as before. If they were not used during training, they cannot be specified during prediction.')

    # If bond features were used during training, they must be used when predicting and vice-versa
    if (train_args.bond_features_path is None) != (predict_args.bond_features_path is None):
        raise ValueError('The use of bond descriptors is different between training and prediction. If you used bond '
                         'descriptors for training, please specify a path to new bond descriptors for prediction.')

    # if atom or bond features were scaled, the same must be done during prediction
    if train_args.features_scaling != predict_args.features_scaling:
        raise ValueError('If scaling of the additional features was done during training, the '
                         'same must be done during prediction.')

    # If atom descriptors were used during training, they must be used when predicting and vice-versa
    if train_args.atom_descriptors != predict_args.atom_descriptors:
        raise ValueError('The use of atom descriptors is inconsistent between training and prediction. '
                         'If atom descriptors were used during training, they must be specified again '
                         'during prediction using the same type of descriptors as before. '
                         'If they were not used during training, they cannot be specified during prediction.')

    # If bond features were used during training, they must be used when predicting and vice-versa
    if (train_args.bond_features_path is None) != (predict_args.bond_features_path is None):
        raise ValueError('The use of bond descriptors is different between training and prediction. If you used bond'
                         'descriptors for training, please specify a path to new bond descriptors for prediction.')

    if validate_feature_sources:
        # If features were used during training, they must be used when predicting
        if (((train_args.features_path is None) != (predict_args.features_path is None))
            or ((train_args.features_generator is None) != (predict_args.features_generator is None))):
            raise ValueError('Features were used during training so they must be specified again during prediction '
                            'using the same type of features as before (with either --features_generator or '
                            '--features_path and using --no_features_scaling if applicable).')
