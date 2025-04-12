import argparse
import os
import random
import math
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# GNN model with node, edge, and graph prediction heads
class PolymerGNN(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=300, num_message_passing_layers=3):
        super(PolymerGNN, self).__init__()
        # Message-passing (encoder) layers
        self.num_layers = num_message_passing_layers
        self.node_mlps = nn.ModuleList()
        self.edge_mlps = nn.ModuleList()
        for layer in range(num_message_passing_layers):
            # Linear transformations for node and edge features (placeholder for actual GNN layers)
            in_node_dim = node_feat_dim if layer == 0 else hidden_dim
            in_edge_dim = edge_feat_dim if layer == 0 else hidden_dim
            self.node_mlps.append(nn.Linear(in_node_dim, hidden_dim))
            self.edge_mlps.append(nn.Linear(in_edge_dim, hidden_dim))
        # Node masking head: predicts original node features from final node embeddings
        self.node_pred_head = nn.Linear(hidden_dim, node_feat_dim)
        # Edge masking head: predicts original edge features from final edge embeddings
        self.edge_pred_head = nn.Linear(hidden_dim, edge_feat_dim)
        # Graph-level head: 3-layer MLP mapping graph embedding to a scalar (e.g., molecular weight)
        self.graph_head_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.graph_head_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.graph_head_fc3 = nn.Linear(hidden_dim, 1)  # outputs a single regression value

    def forward(self, data):
        # Input: data.x (node features), data.edge_index, data.edge_attr (edge features), data.batch (for batching)
        x = data.x
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        # Message passing: update node (and edge) features through the layers
        for i in range(self.num_layers):
            x = F.relu(self.node_mlps[i](x))
            if edge_attr is not None:
                edge_attr = F.relu(self.edge_mlps[i](edge_attr))
            # (Note: In an actual GNN, neighbor aggregation using data.edge_index would be applied here)
        # Heads computations:
        node_pred = self.node_pred_head(x)           # node feature reconstruction for each node
        edge_pred = None
        if edge_attr is not None:
            edge_pred = self.edge_pred_head(edge_attr)  # edge feature reconstruction for each edge
        # Graph embedding via pooling, then graph-level prediction
        if DataLoader is not None and hasattr(data, 'batch'):
            graph_embed = global_add_pool(x, data.batch)  # sum-pooling of node embeddings per graph
        else:
            graph_embed = x.sum(dim=0, keepdim=True) if x.dim() > 1 else x
        h = F.relu(self.graph_head_fc1(graph_embed))
        h = F.relu(self.graph_head_fc2(h))
        graph_pred = self.graph_head_fc3(h)          # predicted molecular weight (pseudo-label)
        return node_pred, edge_pred, graph_pred

##### Stage 1: Node/Edge Masking Pretraining #####
def train_stage1(model, train_loader, val_set, criterion_node, criterion_edge, optimizer, scheduler, args, device):
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    print('Starting Stage 1: Node & Edge Masking Pretraining...')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        batch_count = 0
        # Training loop
        for batch in train_loader:
            batch = batch.to(device) if hasattr(batch, 'to') else batch
            optimizer.zero_grad()
            # Randomly mask `mask_num_nodes` nodes and `mask_num_edges` edges in each graph of the batch
            masked_nodes = []
            original_node_feats = []
            masked_edges = []
            original_edge_feats = []
            if DataLoader is not None and hasattr(batch, 'batch'):  # batched graphs
                num_graphs = batch.batch.max().item() + 1
                # Identify node indices for each graph in the batch
                for g in range(num_graphs):
                    node_idx = (batch.batch == g).nonzero(as_tuple=True)[0]
                    if node_idx.numel() > 0:
                        # Select random nodes to mask in graph g
                        perm = torch.randperm(node_idx.numel(), device=device)
                        sel_nodes = node_idx[perm[:min(args.mask_num_nodes, node_idx.numel())]]
                        masked_nodes.append(sel_nodes)
                        original_node_feats.append(batch.x[sel_nodes].clone())
                        batch.x[sel_nodes] = 0  # mask node features
                    # Select random edges to mask in graph g
                    if hasattr(batch, 'edge_index') and hasattr(batch, 'edge_attr'):
                        # Find edges belonging to graph g (both end nodes in node_idx range)
                        edge_idx = []
                        for e in range(batch.edge_index.size(1)):
                            u = batch.edge_index[0, e].item()
                            if batch.batch[u].item() == g:  # edge e belongs to graph g
                                edge_idx.append(e)
                        if edge_idx:
                            edge_idx = torch.tensor(edge_idx, device=device)
                            perm_e = torch.randperm(edge_idx.numel(), device=device)
                            sel_edges = edge_idx[perm_e[:min(args.mask_num_edges, edge_idx.numel())]]
                            masked_edges.append(sel_edges)
                            original_edge_feats.append(batch.edge_attr[sel_edges].clone())
                            batch.edge_attr[sel_edges] = 0  # mask edge features
            else:
                # Single-graph batch (no batch attribute)
                N = batch.x.size(0)
                if N > 0:
                    perm = torch.randperm(N, device=device)
                    sel_nodes = perm[:min(args.mask_num_nodes, N)]
                    masked_nodes.append(sel_nodes)
                    original_node_feats.append(batch.x[sel_nodes].clone())
                    batch.x[sel_nodes] = 0
                if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                    E = batch.edge_attr.size(0)
                    if E > 0:
                        perm_e = torch.randperm(E, device=device)
                        sel_edges = perm_e[:min(args.mask_num_edges, E)]
                        masked_edges.append(sel_edges)
                        original_edge_feats.append(batch.edge_attr[sel_edges].clone())
                        batch.edge_attr[sel_edges] = 0
            # Forward pass
            node_pred, edge_pred, graph_pred = model(batch)
            # Compute masking losses (graph_pred is ignored in Stage 1 as graph_loss_weight=0)
            loss_node = 0.0
            loss_edge = 0.0
            if node_pred is not None and masked_nodes:
                masked_nodes_cat = torch.cat(masked_nodes)
                orig_node_cat = torch.cat(original_node_feats)
                loss_node = criterion_node(node_pred[masked_nodes_cat], orig_node_cat)
            if edge_pred is not None and masked_edges:
                masked_edges_cat = torch.cat(masked_edges)
                orig_edge_cat = torch.cat(original_edge_feats)
                loss_edge = criterion_edge(edge_pred[masked_edges_cat], orig_edge_cat)
            loss = loss_node + loss_edge
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            batch_count += 1
            # (No need to restore masked features here, as batch is newly created each loop iteration)
        avg_train_loss = total_train_loss / batch_count if batch_count > 0 else 0.0

        # Validation: evaluate masking reconstruction error on val_set
        model.eval()
        total_val_loss = 0.0
        val_count = 0
        for data in val_set:  # iterate each graph in validation set
            data = data.to(device) if hasattr(data, 'to') else data
            # Apply a fixed mask for validation (e.g., mask first N nodes/edges for consistency)
            mask_n = min(args.mask_num_nodes, data.x.size(0))
            val_mask_nodes = torch.arange(mask_n, device=device)
            orig_nodes = data.x[val_mask_nodes].clone()
            data.x[val_mask_nodes] = 0
            val_mask_edges = None
            orig_edges = None
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                mask_e = min(args.mask_num_edges, data.edge_attr.size(0))
                if mask_e > 0:
                    val_mask_edges = torch.arange(mask_e, device=device)
                    orig_edges = data.edge_attr[val_mask_edges].clone()
                    data.edge_attr[val_mask_edges] = 0
            # Forward pass on the masked val graph
            with torch.no_grad():
                node_pred, edge_pred, _ = model(data)
                loss_n = 0.0
                loss_e = 0.0
                if node_pred is not None and val_mask_nodes.numel() > 0:
                    loss_n = criterion_node(node_pred[val_mask_nodes], orig_nodes)
                if edge_pred is not None and val_mask_edges is not None and val_mask_edges.numel() > 0:
                    loss_e = criterion_edge(edge_pred[val_mask_edges], orig_edges)
                total_val_loss += (loss_n + loss_e).item()
            # Restore original features
            data.x[val_mask_nodes] = orig_nodes
            if val_mask_edges is not None:
                data.edge_attr[val_mask_edges] = orig_edges
            val_count += 1
        avg_val_loss = total_val_loss / val_count if val_count > 0 else 0.0

        # Logging
        print(f"Stage 1 Epoch {epoch}/{args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
            # Save best Stage 1 model checkpoint
            torch.save(model.state_dict(), 'stage1_best_model.pt')
        else:
            patience_counter += 1
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        if patience_counter >= args.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch} for Stage 1.")
            break

    # Load best model weights for next stage
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

##### Stage 2: Graph-Level Molecular Weight Prediction #####
def train_stage2(model, train_loader, val_loader, criterion_graph, optimizer, scheduler, args, device):
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    print('Starting Stage 2: Graph-Level Molecular Weight Prediction...')
    for epoch in range(1, args.epochs_stage2 + 1):
        model.train()
        total_train_loss = 0.0
        batch_count = 0
        # Training loop (graph-level regression)
        for batch in train_loader:
            batch = batch.to(device) if hasattr(batch, 'to') else batch
            optimizer.zero_grad()
            _, _, graph_pred = model(batch)  # only graph_pred is used
            # Assume the molecular weight label is stored in batch.mw or batch.y
            target = None
            if hasattr(batch, 'mw'):
                target = batch.mw
            elif hasattr(batch, 'y'):
                target = batch.y  # alternative label attribute
            if target is None:
                raise ValueError("No graph-level target (molecular weight) found in batch data for Stage 2.")
            target = target.to(device)
            loss = criterion_graph(graph_pred.view_as(target), target.float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            batch_count += 1
        avg_train_loss = total_train_loss / batch_count if batch_count > 0 else 0.0

        # Validation loop (graph-level)
        model.eval()
        total_val_loss = 0.0
        batch_count_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device) if hasattr(batch, 'to') else batch
                _, _, graph_pred = model(batch)
                target = batch.mw if hasattr(batch, 'mw') else batch.y if hasattr(batch, 'y') else None
                if target is None:
                    raise ValueError("No graph-level target found in validation data for Stage 2.")
                target = target.to(device)
                loss_val = criterion_graph(graph_pred.view_as(target), target.float())
                total_val_loss += loss_val.item()
                batch_count_val += 1
        avg_val_loss = total_val_loss / batch_count_val if batch_count_val > 0 else 0.0

        print(f"Stage 2 Epoch {epoch}/{args.epochs_stage2} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
            torch.save(model.state_dict(), 'stage2_best_model.pt')
        else:
            patience_counter += 1
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        if patience_counter >= args.early_stop_patience_stage2:
            print(f"Early stopping triggered at epoch {epoch} for Stage 2.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# Utility: Save SMILES and embeddings to CSV
def save_embeddings_csv(model, dataset, filename='ssl_embeddings.csv', device='cpu'):
    model.eval()
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Prepare header
        embed_dim = model.graph_head_fc1.in_features if hasattr(model, 'graph_head_fc1') else None
        header = ["SMILES"]
        if embed_dim:
            header += [f"embed_{i}" for i in range(embed_dim)]
        else:
            header.append("embedding_vector")
        writer.writerow(header)
        # Compute embeddings for each polymer
        with torch.no_grad():
            for data in dataset:
                data = data.to(device) if hasattr(data, 'to') else data
                # Get graph embedding (output of second FC layer in graph head)
                x = data.x
                edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
                for i in range(model.num_layers if hasattr(model, 'num_layers') else 0):
                    x = F.relu(model.node_mlps[i](x))
                    if edge_attr is not None and i < len(model.edge_mlps):
                        edge_attr = F.relu(model.edge_mlps[i](edge_attr))
                if DataLoader is not None and hasattr(data, 'batch'):
                    graph_embed = global_add_pool(x, data.batch)
                else:
                    graph_embed = x.sum(dim=0, keepdim=True) if x.dim() > 1 else x
                # Pass through first two FC layers of graph head to get embedding vector
                h = F.relu(model.graph_head_fc1(graph_embed))
                h = F.relu(model.graph_head_fc2(h))
                embed_vec = h.view(-1).cpu().numpy().tolist()
                smiles = data.smiles if hasattr(data, 'smiles') else ''
                if embed_dim:
                    row = [smiles] + [f"{v:.6f}" for v in embed_vec]
                else:
                    embed_str = " ".join(f"{v:.6f}" for v in embed_vec)
                    row = [smiles, embed_str]
                writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Two-Stage Self-Supervised Pretraining for Polymer GNN")
    # Stage 1 parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for Stage 1 (node/edge masking pretraining)')
    parser.add_argument('--mask_num_nodes', type=int, default=2, help='Number of nodes to mask per graph in Stage 1')
    parser.add_argument('--mask_num_edges', type=int, default=2, help='Number of edges to mask per graph in Stage 1')
    parser.add_argument('--early_stop_patience', type=int, default=10, help='Early stopping patience for Stage 1')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='LR scheduler patience for Stage 1 (ReduceLROnPlateau)')
    # Stage 2 parameters
    parser.add_argument('--epochs_stage2', type=int, default=0, help='Number of epochs for Stage 2 (graph-level task). If 0, Stage 2 is skipped.')
    parser.add_argument('--early_stop_patience_stage2', type=int, default=10, help='Early stopping patience for Stage 2')
    parser.add_argument('--scheduler_patience_stage2', type=int, default=5, help='LR scheduler patience for Stage 2')
    parser.add_argument('--freeze_encoder_stage2', action='store_true', help='Freeze encoder (GNN message-passing layers) during Stage 2')
    # General parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for optimizer (both stages)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--device', type=str, default='cpu', help="Compute device: 'cpu' or 'cuda'")
    # (Add other arguments as needed, e.g., data file paths)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or 'cpu' in args.device else 'cuda')
    # Load dataset (user should replace this placeholder with actual data loading)
    train_data = []  # list of training graphs (torch_geometric Data objects or similar)
    val_data = []    # list of validation graphs
    # Each Data in train_data/val_data should have attributes: x, edge_index, edge_attr, smiles, and mw (molecular weight label)

    # Create DataLoaders for train (and val if using mini-batch for Stage 2)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True) if DataLoader and len(train_data) > 0 else train_data
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False) if DataLoader and len(val_data) > 0 else val_data

    # Initialize model
    if len(train_data) == 0:
        raise ValueError("Training dataset is empty. Please load training data.")
    node_feat_dim = train_data[0].x.size(1)
    edge_feat_dim = train_data[0].edge_attr.size(1) if hasattr(train_data[0], 'edge_attr') and train_data[0].edge_attr is not None else 0
    model = PolymerGNN(node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim).to(device)
    # Define loss functions
    criterion_node = nn.MSELoss()
    criterion_edge = nn.MSELoss()
    criterion_graph = nn.MSELoss()
    # Optimizer and LR scheduler for Stage 1
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.scheduler_patience, verbose=True)

    # Run Stage 1 pretraining
    model = train_stage1(model, train_loader, val_data, criterion_node, criterion_edge, optimizer, scheduler, args, device)

    # Run Stage 2 training (if requested)
    if args.epochs_stage2 and args.epochs_stage2 > 0:
        # Load best Stage 1 model checkpoint
        model.load_state_dict(torch.load('stage1_best_model.pt', map_location=device))
        # Freeze encoder layers for Stage 2 if specified
        if args.freeze_encoder_stage2:
            for name, param in model.named_parameters():
                if name.startswith('node_mlps') or name.startswith('edge_mlps'):
                    param.requires_grad = False
        # New optimizer and scheduler for Stage 2 (only updating unfrozen parameters)
        optimizer2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', patience=args.scheduler_patience_stage2, verbose=True)
        # Train on graph-level task
        model = train_stage2(model, train_loader, val_loader, criterion_graph, optimizer2, scheduler2, args, device)
    else:
        print("Skipping Stage 2 (epochs_stage2 is 0).")

    # Save SMILES and embeddings of all (train+val) polymers to CSV
    all_data = train_data + val_data
    save_embeddings_csv(model, all_data, filename='ssl_embeddings.csv', device=device)
    print("Saved polymer embeddings to ssl_embeddings.csv")

if __name__ == "__main__":
    main()

