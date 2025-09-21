#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TFG - Predicción de lanzamientos y fintas en penaltis de balonmano
Script único y reproducible para:
  - Cargar CSV(s) de embeddings (con fusión opcional)
  - Unir con folds
  - Preprocesar y normalizar
  - Entrenar y evaluar un Transformer con CV estratificada (5-fold)
  - Balancear con SMOTE SOLO en train
"""

import os
import argparse
import copy
import ast
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup


# ----------------------------
# Utilidades y carga de datos
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cargar_embeddings(csv_path):
    """
    Lee un CSV con columnas: ['video_id','features','label'].
    - label: fusiona clase 2 -> 1 (binario)
    - features: parsea lista string -> np.array (seq_len, feat_dim)
    """
    df = pd.read_csv(csv_path)
    req_cols = {'video_id', 'features', 'label'}
    if not req_cols.issubset(df.columns):
        raise ValueError(f"{csv_path} debe contener columnas {req_cols}")
    df = df[['video_id', 'features', 'label']].copy()
    df['label'] = df['label'].replace(2, 1)
    df['features'] = df['features'].apply(lambda x: np.array(ast.literal_eval(x)))
    return df


def cargar_folds(folds_path):
    """
    CSV con columnas: ['fold','video_id','label'] (o similar).
    """
    df = pd.read_csv(folds_path)
    # Normalizamos nombres esperados
    if len(df.columns) >= 3:
        df = df.iloc[:, :3]
        df.columns = ['fold', 'video_id', 'label']
    else:
        raise ValueError("El archivo de folds debe tener al menos 3 columnas (fold, video_id, label)")
    return df[['fold', 'video_id']]


def fusionar_embeddings(df1, df2):
    """
    Concatena las features por canal: (seq_len, d1) + (seq_len, d2) -> (seq_len, d1+d2).
    Requiere que el orden de video_id coincida.
    """
    df1 = df1.sort_values('video_id').reset_index(drop=True)
    df2 = df2.sort_values('video_id').reset_index(drop=True)
    if df1['video_id'].tolist() != df2['video_id'].tolist():
        raise ValueError("video_id desalineados entre los dos embeddings")

    concatenados = [np.concatenate([a, b], axis=1) for a, b in zip(df1['features'], df2['features'])]
    out = pd.DataFrame({
        'video_id': df1['video_id'],
        'features': concatenados,
        'label': df1['label'].values  
    })
    return out


def ensamblar_dataset(df_embed, df_folds):
    """
    Une embeddings con folds.
    Devuelve X (N, T, D), y (N,), folds (N,)
    """
    df = df_embed.merge(df_folds, on='video_id', how='inner')
    X = np.stack(df['features'].values)  
    y = df['label'].values.astype(int)    
    folds = df['fold'].values.astype(int) 
    return X, y, folds


# ----------------------------
# Dataset y SMOTE
# ----------------------------
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def balancear_con_smote(X_train, y_train):
    """
    Aplica SMOTE solo sobre train.
    Repliega la secuencia sobre el eje temporal para SMOTE y después reconstituye.
    """
    ns, sl, dim = X_train.shape
    X_flat = X_train.reshape(-1, dim)
    y_flat = np.repeat(y_train, sl)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_flat, y_flat)

    ns_res = len(y_res) // sl
    X_seq = X_res.reshape(ns_res, sl, dim)
    y_seq = y_res[::sl]
    return X_seq, y_seq


# ----------------------------
# Modelo: Transformer
# ----------------------------
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, T, C)
        w = torch.softmax(self.attn(x), dim=1)  
        return (x * w).sum(dim=1)               


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len + 1, d_model)  
        position = torch.arange(0, max_len + 1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_layers=4, num_heads=16, dropout=0.5, max_len=100):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn_pool = AttentionPooling(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # logits
        )

    def forward(self, x):
        # x: (B, T, D)
        B = x.size(0)
        x = self.input_proj(x)              
        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)      
        x = self.pos_enc(x)                 
        x = self.encoder(x)                 
        x = self.norm(x)
        h = self.attn_pool(x)               
        h = self.dropout(h)
        logits = self.fc(h).squeeze(1)      
        return logits


# ----------------------------
# Entrenamiento + validación
# ----------------------------
def train_model(model, train_loader, val_loader, epochs=50, warmup_ratio=0.1, patience=15):
    """
    Entrena con BCEWithLogitsLoss (+pos_weight), scheduler linear + warmup,
    early stopping por F1 en validación.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # pos_weight para desbalance
    all_labels = torch.tensor([y for _, y in train_loader.dataset], dtype=torch.float32)
    n_pos = (all_labels == 1).sum().item()
    n_neg = (all_labels == 0).sum().item()
    pos_weight = torch.tensor([max(1.0, n_neg / max(1, n_pos))], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-3)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_f1, wait, best_state = 0.0, 0, None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # Validación
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                logits = model(X).cpu()
                preds = (torch.sigmoid(logits) > 0.5).int()
                y_pred += preds.tolist()
                y_true += y.tolist()

        f1 = f1_score(y_true, y_pred)
        print(f"Epoch {epoch:03d}/{epochs} | TrainLoss={np.mean(train_losses):.4f} | ValF1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"— Early stopping en epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def plot_roc_pr_curve(y_true, y_probs, title_suffix=""):
    assert np.all((y_probs >= 0) & (y_probs <= 1)), "Probabilidades fuera de [0,1]."

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = average_precision_score(y_true, y_probs)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC' + title_suffix); plt.legend(); plt.grid(alpha=.3)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, lw=2, label=f'AP={pr_auc:.2f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR' + title_suffix); plt.legend(); plt.grid(alpha=.3)

    plt.tight_layout()
    plt.show()


def cross_validation_transformer(X, y, folds, usar_smote=True, batch_size=32):
    """
    CV 5-fold fija:
      - eval_fold = fold de test
      - val_fold  = (eval_fold + 1) % 5
      - scaler se ajusta con (train+val) de TODOS MENOS eval_fold
    """
    n_folds = 5
    epochs = 50

    accs, f1s = [], []
    all_y_true, all_y_probs = [], []

    for eval_fold in range(n_folds):
        print(f"\n=== Fold test={eval_fold} / val={(eval_fold+1)%n_folds} ===")
        set_seed(42 + eval_fold)

        # normalización: fit con todo menos el fold de evaluación (train+val)
        X_scaler = X[folds != eval_fold].reshape(-1, X.shape[-1])
        scaler = StandardScaler().fit(X_scaler)
        X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

        val_fold = (eval_fold + 1) % n_folds
        train_idx = (folds != eval_fold) & (folds != val_fold)
        val_idx   = (folds == val_fold)
        test_idx  = (folds == eval_fold)

        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_val,   y_val   = X_scaled[val_idx],   y[val_idx]
        X_test,  y_test  = X_scaled[test_idx],  y[test_idx]

        if usar_smote:
            X_train, y_train = balancear_con_smote(X_train, y_train)

        train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(SequenceDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(SequenceDataset(X_test,  y_test),  batch_size=batch_size, shuffle=False)

        input_dim = X.shape[2]
        model = TransformerClassifier(input_dim=input_dim)

        # Entrenamiento con early stopping por F1 val
        model = train_model(model, train_loader, val_loader, epochs=epochs)

        # Evaluación en test
        model.eval()
        y_true, y_pred, y_probs = [], [], []
        device = next(model.parameters()).device
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                logits = model(Xb).cpu()
                probs = torch.sigmoid(logits).numpy()
                preds = (probs > 0.5).astype(int)
                y_true += yb.numpy().tolist()
                y_pred += preds.tolist()
                y_probs += probs.tolist()

        acc = accuracy_score(y_true, y_pred)
        f1  = f1_score(y_true, y_pred)
        print(f"Fold {eval_fold} | Test Acc={acc:.4f} | Test F1={f1:.4f}")

        accs.append(acc); f1s.append(f1)
        all_y_true.extend(y_true); all_y_probs.extend(y_probs)

    print("\n=== Resultados globales (CV 5-fold) ===")
    print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"F1-score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    plot_roc_pr_curve(np.array(all_y_true), np.array(all_y_probs), title_suffix=" (CV)")

    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


# ----------------------------
# Main (CLI mínima)
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="TFG - Transformer + SMOTE desde CSV (con fusión opcional)")
    parser.add_argument("--csv1", type=str, required=True, help="Ruta al primer CSV de embeddings (obligatorio)")
    parser.add_argument("--csv2", type=str, default=None, help="Ruta al segundo CSV (opcional, para fusión)")
    parser.add_argument("--folds", type=str, required=True, help="Ruta al CSV de folds (fold, video_id, label)")
    parser.add_argument("--no_smote", action="store_true", help="Desactivar SMOTE (por defecto, se usa)")
    args = parser.parse_args()

    set_seed(42)

    print("Cargando embeddings...")
    df1 = cargar_embeddings(args.csv1)
    if args.csv2:
        df2 = cargar_embeddings(args.csv2)
        df_embed = fusionar_embeddings(df1, df2)
        print(f"Fusión realizada: dims -> {df_embed['features'].iloc[0].shape}")
    else:
        df_embed = df1

    print("Cargando folds...")
    df_folds = cargar_folds(args.folds)

    print("Ensamblando dataset...")
    X, y, folds = ensamblar_dataset(df_embed, df_folds)
    print(f"Dataset: X={X.shape}, y={y.shape}, folds={folds.shape}")

    usar_smote = not args.no_smote
    print(f"SMOTE en train: {usar_smote}")

    print("Iniciando validación cruzada con Transformer (5 folds, 50 epochs fijos)...")
    cross_validation_transformer(X, y, folds, usar_smote=usar_smote, batch_size=32)


if __name__ == "__main__":
    main()

