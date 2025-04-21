import json
import csv
from logging import Logger
import os
from typing import Dict, List, Union
from argparse import Namespace
import numpy as np
import pandas as pd
from tensorboardX import SummaryWriter
import torch
import pickle
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
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    torch.manual_seed(args.pytorch_seed)

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
                            smiles_columns=args.smiles_columns,
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
    elif args.split_type == 'predetermined' and args.folds_file is not None:
        # Handle grouped folds with optional train_fold_index (default 0)
        with open(args.folds_file, 'rb') as f:
            folds = pickle.load(f)
    
        train_idx = folds[getattr(args, 'train_fold_index', 0)]  # default to 0 if not provided
        val_idx = folds[args.val_fold_index]
        test_idx = folds[args.test_fold_index]
    
        train_data = MoleculeDataset([data[i] for i in train_idx])
        val_data = MoleculeDataset([data[i] for i in val_idx])
        test_data = MoleculeDataset([data[i] for i in test_idx])
    
        debug(f"Using predetermined grouped folds: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

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

    if args.train_frac < 1.0:
        subset_size = int(len(train_data) * args.train_frac)
        rng = np.random.default_rng(seed=args.seed)
        selected_indices = rng.choice(len(train_data), size=subset_size, replace=False)
        train_data = MoleculeDataset([train_data[i] for i in selected_indices])
        debug(f'Subsampled training set to {subset_size:,} entries (train_frac={args.train_frac})')
    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

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

    loss_func = get_loss_func(args)

    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    if len(data) <= args.cache_cutoff:
        set_cache_graph(True)
        num_workers = 0
    else:
        set_cache_graph(False)
        num_workers = args.num_workers

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

    train_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        class_balance=args.class_balance,
        shuffle=True,
        seed=args.seed
    )
                     
    # ➕ Create a new train loader without shuffling for evaluation
    train_eval_data_loader = MoleculeDataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=num_workers,
        shuffle=False  # This is key!
    )

    if args.class_balance:
        debug(f'With class_balance, effective train size = {train_data_loader.iter_size:,}')

    for model_idx in range(args.ensemble_size):
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        
        # ✅ Setup CSV logging path and header logic with integrity check
        csv_log_path = os.path.join(save_dir, 'train_val_loss_log.csv')
        write_header = False
        
        # Always create a fresh file if not resuming
        if args.resume_from_checkpoint is None:
            # Completely clear the file for a fresh run
            with open(csv_log_path, 'w') as f:
                pass  # clear file for fresh run
            write_header = True
        else:
            # Check if the header exists and is correct for resuming
            expected_header_start = 'epoch,train_avg_'
            try:
                with open(csv_log_path, 'r') as f:
                    first_line = f.readline().strip()
                if not first_line.startswith(expected_header_start):
                    write_header = True
            except FileNotFoundError:
                write_header = True

        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)

        model = MoleculeModel(args)
        start_epoch = 0

        if args.resume_from_checkpoint is not None:
            debug(f'🔁 Resuming full training from checkpoint: {args.resume_from_checkpoint}')
            checkpoint = torch.load(
                args.resume_from_checkpoint,
                map_location=args.device,
                weights_only=False
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            # ✅ DEBUG: Confirm FFN weights loaded
            print("✅ FFN weights after loading checkpoint:")
            for name, param in model.named_parameters():
                if "ffn" in name or "graph_head" in name:
                    print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
            optimizer = build_optimizer(model, args)
            scheduler = build_lr_scheduler(optimizer, args)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(args.device)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            debug(f"➡️  Resumed at epoch {start_epoch}")

        else:
            debug(f'🛠️ Building model {model_idx} from scratch.')
            model = MoleculeModel(args)
            optimizer = build_optimizer(model, args)
            scheduler = build_lr_scheduler(optimizer, args)

        # ✅ Load SSL checkpoint if specified via --checkpoint_frzn
        if args.checkpoint_frzn is not None:
            debug(f'📥 Loading pretrained checkpoint from {args.checkpoint_frzn}')
            checkpoint = torch.load(args.checkpoint_frzn, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
            # ✅ Unfreeze all layers unless --frzn_encoder or --frzn_ffn_layers are set
            if not getattr(args, 'frzn_encoder', False) and getattr(args, 'frzn_ffn_layers', 0) == 0:
                unfrozen = 0
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        unfrozen += 1
                        debug(f'🔓 Unfroze parameter: {name}')
                debug(f'✅ Unfroze {unfrozen} parameters from checkpoint.')

        # ✅ Freeze encoder weights if --frzn_encoder
        if getattr(args, 'frzn_encoder', False):
            for name, param in model.encoder.named_parameters():
                param.requires_grad = False
                debug(f'❄️  Frozen encoder layer: {name}')

        # ✅ Freeze graph_head layers if --frzn_ffn_layers > 0
        if getattr(args, 'frzn_ffn_layers', 0) > 0:
            if hasattr(model, 'graph_head'):
                ffn_params = list(model.graph_head.parameters())
            elif hasattr(model, 'ffn'):
                ffn_params = list(model.ffn.parameters())
            else:
                raise AttributeError("Model has no graph_head or ffn attribute")

            num_to_freeze = min(args.frzn_ffn_layers, len(ffn_params))
            for i in range(-num_to_freeze, 0):
                ffn_params[i].requires_grad = False
                debug(f'❄️  Frozen FFN layer {i} with shape {ffn_params[i].shape}')

        debug(model)
        debug(f'Number of parameters = {param_count_all(model):,}')

        if args.cuda:
            debug('Moving model to cuda')
        model = model.to(args.device)
        # ✅ DEBUG: Print frozen layers
        print("✅ Frozen layers in the model:")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"❄️  {name}")
                
        save_checkpoint(os.path.join(save_dir, MODEL_FILE_NAME), model, scaler,
                        features_scaler, atom_descriptor_scaler, bond_feature_scaler, args)

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
            
            # Only step per-epoch schedulers here (like ExponentialLR)
            # CosineAnnealingLR, CyclicLR, and NoamLR are now stepped per batch
            if isinstance(scheduler, ExponentialLR) and not (isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR) or 
                                      isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR) or
                                      isinstance(scheduler, NoamLR)):
                scheduler.step()
            
            elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR) or \
                 isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
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
            
            train_scores = evaluate(
                model=model,
                data_loader=train_eval_data_loader,
                num_tasks=args.num_tasks,
                metrics=args.metrics,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )
            
            # 🔸 Write CSV log
            with open(csv_log_path, 'a', newline='') as f:
                writer_csv = csv.writer(f)
                if write_header:
                    header = ['epoch']
                    for metric in args.metrics:
                        header.append(f'train_avg_{metric}')
                        header.append(f'val_avg_{metric}')
                        header += [f'train_{task}_{metric}' for task in args.task_names]
                        header += [f'val_{task}_{metric}' for task in args.task_names]
                    writer_csv.writerow(header)
                    write_header = False

                row = [epoch]
                for metric in args.metrics:
                    train_vals = train_scores[metric]
                    val_vals = val_scores[metric]
                    row.append(np.nanmean(train_vals))
                    row.append(np.nanmean(val_vals))
                    row += train_vals
                    row += val_vals
                writer_csv.writerow(row)

            debug(f"📊 Raw val_scores: {val_scores}")
            for metric, scores in val_scores.items():
                avg_val_score = np.nanmean(scores)
                debug(f'Validation {metric} = {avg_val_score:.6f}')
                writer.add_scalar(f'validation_{metric}', avg_val_score, n_iter)
                
                if args.show_individual_scores:
                    for task_name, val_score in zip(args.task_names, scores):
                        debug(f'Validation {task_name} {metric} = {val_score:.6f}')
                        writer.add_scalar(f'validation_{task_name}_{metric}', val_score, n_iter)

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch
            }, os.path.join(save_dir, 'model.pt'))

            avg_val_score = np.nanmean(val_scores[args.metric])
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch
                }, os.path.join(save_dir, 'best_resume_checkpoint.pt'))

                full_ckpt_path = os.path.join(save_dir, 'best_model_full.pt')
                args_to_save = TrainArgs()
                args_to_save.from_dict(vars(args))
                print(f"[DEBUG] args_to_save type = {type(args_to_save)}")        
                save_checkpoint(
                    path=full_ckpt_path,
                    model=model,
                    scaler=scaler,
                    features_scaler=features_scaler,
                    atom_descriptor_scaler=atom_descriptor_scaler,
                    bond_feature_scaler=bond_feature_scaler,
                    args=args_to_save
                )
                debug(f"✅ Saved full checkpoint for predict.py to: {full_ckpt_path}")

        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')

        checkpoint = torch.load(os.path.join(save_dir, 'best_resume_checkpoint.pt'), map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

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

        for metric, scores in test_scores.items():
            avg_test_score = np.nanmean(scores)
            info(f'Model {model_idx} test {metric} = {avg_test_score:.6f}')
            writer.add_scalar(f'test_{metric}', avg_test_score, 0)

            if args.show_individual_scores and args.dataset_type != 'spectra':
                for task_name, test_score in zip(args.task_names, scores):
                    info(f'Model {model_idx} test {task_name} {metric} = {test_score:.6f}')
                    writer.add_scalar(f'test_{task_name}_{metric}', test_score, n_iter)
        writer.close()

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
        avg_ensemble_test_score = np.nanmean(scores)
        info(f'Ensemble test {metric} = {avg_ensemble_test_score:.6f}')

        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, scores):
                info(f'Ensemble test {task_name} {metric} = {ensemble_score:.6f}')

    with open(os.path.join(args.save_dir, 'test_scores.json'), 'w') as f:
        json.dump(ensemble_scores, f, indent=4, sort_keys=True)

    if args.save_preds:
        test_preds_dataframe = pd.DataFrame(data={'smiles': test_data.smiles()})
        for i, task_name in enumerate(args.task_names):
            test_preds_dataframe[task_name] = [pred[i] for pred in avg_test_preds]
        test_preds_dataframe.to_csv(os.path.join(args.save_dir, 'test_preds.csv'), index=False)

    return ensemble_scores
