import json
from logging import Logger
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
from tqdm import trange
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.spectra_utils import normalize_spectra, load_phase_mask
from chemprop.args import TrainArgs
from chemprop.constants import MODEL_FILE_NAME
from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, MoleculeDataset, set_cache_graph, split_data
from chemprop.models import MoleculeModel
from chemprop.nn_utils import param_count, param_count_all
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, load_checkpoint, makedirs, \
    save_checkpoint, save_smiles_splits, load_frzn_model


def run_training(args: TrainArgs,
                 data: MoleculeDataset,
                 logger: Logger = None) -> Dict[str, List[float]]:
    """
    Loads data, trains a Chemprop model, and returns test scores for the model checkpoint with the highest validation score.

    :param args: A :class:`~chemprop.args.TrainArgs` object containing arguments for
                 loading data and training the Chemprop model.
    :param data: A :class:`~chemprop.data.MoleculeDataset` containing the data.
    :param logger: A logger to record output.
    :return: A dictionary mapping each metric in :code:`args.metrics` to a list of values for each task.

    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set pytorch seed for random initial weights
    torch.manual_seed(args.pytorch_seed)

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path,
                             args=args,
                             features_path=args.separate_test_features_path,
                             atom_descriptors_path=args.separate_test_atom_descriptors_path,
                             bond_features_path=args.separate_test_bond_features_path,
                             phase_features_path=args.separate_test_phase_features_path,
                             smiles_columns=args.smiles_columns,
                             logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path,
                            args=args,
                            features_path=args.separate_val_features_path,
                            atom_descriptors_path=args.separate_val_atom_descriptors_path,
                            bond_features_path=args.separate_val_bond_features_path,
                            phase_features_path=args.separate_val_phase_features_path,
                            smiles_columns = args.smiles_columns,
                            logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data,
                                              split_type=args.split_type,
                                              sizes=(0.8, 0.0, 0.2),
                                              seed=args.seed,
                                              num_folds=args.num_folds,
                                              args=args,
                                              logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data,
                                             split_type=args.split_type,
                                             sizes=(0.8, 0.2, 0.0),
                                             seed=args.seed,
                                             num_folds=args.num_folds,
                                             args=args,
                                             logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data,
                                                     split_type=args.split_type,
                                                     sizes=args.split_sizes,
                                                     seed=args.seed,
                                                     num_folds=args.num_folds,
                                                     args=args,
                                                     logger=logger)

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.save_smiles_splits:
        save_smiles_splits(
            data_path=args.data_path,
            save_dir=args.save_dir,
            task_names=args.task_names,
            features_path=args.features_path,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            smiles_columns=args.smiles_columns,
            logger=logger,
        )

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    if args.atom_descriptor_scaling and args.atom_descriptors is not None:
        atom_descriptor_scaler = train_data.normalize_features(replace_nan_token=0, scale_atom_descriptors=True)
        val_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        test_data.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
    else:
        atom_descriptor_scaler = None

    if args.bond_feature_scaling and args.bond_features_size > 0:
        bond_feature_scaler = train_data.normalize_features(replace_nan_token=0, scale_bond_features=True)
        val_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
        test_data.normalize_features(bond_feature_scaler, scale_bond_features=True)
    else:
        bond_feature_scaler = None
        
    # Optionally subsample training data if --train_frac < 1.0
    if args.train_frac < 1.0:
        subset_size = int(len(train_data) * args.train_frac)
        rng = np.random.default_rng(seed=args.seed)
        selected_indices = rng.choice(len(train_data), size=subset_size, replace=False)
        train_data = MoleculeDataset([train_data[i] for i in selected_indices])
        debug(f'Subsampled training set to {subset_size:,} entries (train_frac={args.train_frac})')
    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        scaler = train_data.normalize_targets()
    elif args.dataset_type == 'spectra':
        debug('Normalizing spectra and excluding spectra regions based on phase')
        args.spectra_phase_mask = load_phase_mask(args.spectra_phase_mask_path)
        for dataset in [train_data, test_data, val_data]:
            data_targets = normalize_spectra(
                spectra=dataset.targets(),
                phase_features=dataset.phase_features(),
                phase_mask=args.spectra_phase_mask,
                excluded_sub_value=None,
                threshold=args.spectra_target_floor,
            )
            dataset.set_targets(data_targets)
        scaler = None
    else:
        scaler = None

    # Get loss function
    loss_func = get_loss_func(args)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Automatically determine whether to cache
    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers

    # Create data loaders
    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
    val_data_loader = MoleculeDataLoader(
        dataset=val_data,
        batch_size=args.batch_size,
        num_workers=num_workers
    )
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=num_workers
    )

    if args.class_balance:
        debug(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)
            
        model = MoleculeModel(args)
        start_epoch = 0
        
        # Load/build model

        # 🔁 Option 1: Resume from full checkpoint
        if args.resume_from_checkpoint is not None:
            debug(f'🔁 Resuming full training from checkpoint: {args.resume_from_checkpoint}')
            checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer = build_optimizer(model, args)
            scheduler = build_lr_scheduler(optimizer, args)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            debug(f"➡️  Resumed at epoch {start_epoch}")

        # ❄️ Option 2: Load and freeze SSL weights
        elif args.checkpoint_frzn is not None:
            debug(f'❄️Loading and freezing parameters from {args.checkpoint_frzn}.')
            ssl_checkpoint = torch.load(args.checkpoint_frzn, map_location='cpu')
            
            # Safely unpack state_dict
            ssl_state_dict = ssl_checkpoint['state_dict']
            
            # Load encoder layers
            encoder_layers = model.encoder.encoder
            encoder_layers[0].W_i.load_state_dict({
                'weight': ssl_state_dict['W_initial.weight']
            })
            encoder_layers[0].W_h.load_state_dict({
                'weight': ssl_state_dict['W_message.weight']
            })
            
            # Freeze encoder weights
            for param in encoder_layers[0].W_i.parameters():
                param.requires_grad = False
            for param in encoder_layers[0].W_h.parameters():
                param.requires_grad = False
        
            # Optionally transfer and freeze FFN layers
            if args.frzn_ffn_layers > 0:
                debug(f'Transferring and freezing first {args.frzn_ffn_layers} FFN layers from SSL model.')
                ffn_state_dict = model.ffn.state_dict()
                for i in range(args.frzn_ffn_layers):
                    ffn_state_dict[f'{2*i}.weight'] = ssl_state_dict[f'graph_head.{2*i}.weight']
                    ffn_state_dict[f'{2*i}.bias'] = ssl_state_dict[f'graph_head.{2*i}.bias']
                
                model.ffn.load_state_dict(ffn_state_dict)   
                    
                for i in range(args.frzn_ffn_layers):
                    model.ffn[2*i].weight.requires_grad = False
                    model.ffn[2*i].bias.requires_grad = False

            debug(f"✅ First {args.frzn_ffn_layers} FFN layers frozen.")
                
            optimizer = build_optimizer(model, args)
            scheduler = build_lr_scheduler(optimizer, args)
        else:
            debug(f'🛠️ Building model {model_idx} from scratch.')
            model = MoleculeModel(args)
            optimizer = build_optimizer(model, args)
            scheduler = build_lr_scheduler(optimizer, args)
         
        # Optionally, overwrite weights:
        if args.checkpoint_frzn is not None:
            debug(f'Loading and freezing parameters from {args.checkpoint_frzn}.')
                
        debug(model)
        
        if args.checkpoint_frzn is not None:
            debug(f'Number of unfrozen parameters = {param_count(model):,}')
            debug(f'Total number of parameters = {param_count_all(model):,}')
        else:
            debug(f'Number of parameters = {param_count_all(model):,}')
        
        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler,
                        features_scaler, atom_descriptor_scaler, bond_feature_scaler, args)


        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in trange(start_epoch, args.epochs):
            debug(f'Epoch {epoch}')
            n_iter = train(
                model=model,
                data_loader=train_data_loader,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores = evaluate(
                model=model,
                data_loader=val_data_loader,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )
            debug(f"📊 Raw val_scores: {val_scores}")
            for metric, scores in val_scores.items():
                # Average validation score
                avg_val_score = np.nanmean(scores)
                debug(f'Validation {metric} = {avg_val_score:.6f}')
                writer.add_scalar(f'validation_{metric}', avg_val_score, n_iter)
                
                if args.show_individual_scores:
                    # Individual validation scores
                    for task_name, val_score in zip(args.task_names, scores):
                        debug(f'Validation {task_name} {metric} = {val_score:.6f}')
                        writer.add_scalar(f'validation_{task_name}_{metric}', val_score, n_iter)
                        # ✅ Save resume checkpoint after every epoch
            
                torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'epoch': epoch
                        }, os.path.join(save_dir, 'model.pt'))            

            # Save model checkpoint if improved validation score
            avg_val_score = np.nanmean(val_scores[args.metric])
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
            
                # ✅ ALSO save best full checkpoint for resume
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch
                }, os.path.join(save_dir, 'best_resume_checkpoint.pt'))

        # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, 'best_model.pt'), device=args.device, logger=logger)

        test_preds = predict(
            model=model,
            data_loader=test_data_loader,
            scaler=scaler
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metrics=args.metrics,
            dataset_type=args.dataset_type,
            logger=logger
        )

        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)

        # Average test score
        for metric, scores in test_scores.items():
            avg_test_score = np.nanmean(scores)
            info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
            writer.add_scalar(f'test_{metric}', avg_test_score, 0)

            if args.show_individual_scores and args.dataset_type != 'spectra':
                # Individual test scores
                for task_name, test_score in zip(args.task_names, scores):
                    info(f'Model {model_idx} test {task_name} {metric} = {test_score:.6f}')
                    writer.add_scalar(f'test_{task_name}_{metric}', test_score, n_iter)
        writer.close()

    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=args.num_tasks,
        metrics=args.metrics,
        dataset_type=args.dataset_type,
        logger=logger
    )

    for metric, scores in ensemble_scores.items():
        # Average ensemble score
        avg_ensemble_test_score = np.nanmean(scores)
        info(f'Ensemble test {metric} = {avg_ensemble_test_score:.6f}')

        # Individual ensemble scores
        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, scores):
                info(f'Ensemble test {task_name} {metric} = {ensemble_score:.6f}')

    # Save scores
    with open(os.path.join(args.save_dir, 'test_scores.json'), 'w') as f:
        json.dump(ensemble_scores, f, indent=4, sort_keys=True)

    # Optionally save test preds
    if args.save_preds:
        test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})

        for i, task_name in enumerate(args.task_names):
            test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]

        test_preds_dataframe.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)

    return ensemble_scores
