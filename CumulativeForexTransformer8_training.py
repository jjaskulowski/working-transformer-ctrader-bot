# -*- coding: utf-8 -*-
"""
Decoder-based Transformer – EUR/USD sequence prediction with dual outputs (FIXED)
================================================================================
* Architecture: Transformer Decoder with causal (triangular) mask
* Dual outputs:
  1. Next-step OHLCV diff prediction (autoregressive)
  2. Threshold exceedance probabilities (20 levels)
* Weighted loss: sequence positions weighted 0.5 → 1.0 (later = more important)
* Self-contained – only needs pandas, numpy, torch, x-transformers.
"""
from __future__ import annotations

# ── stdlib ───────────────────────────────────────────────────────
import os, argparse, gc
from pathlib import Path
from typing import Tuple, List, Union, Dict

# ── third‑party ──────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import Decoder

# ── hyper‑parameters ────────────────────────────────────────────
SEQ_LEN: int = 200            # length of H1 window
# 10 thresholds in %: 0.01, 0.014, 0.02 … 0.224 (√2 ratio)
PCT_THRESHOLDS: List[float] = np.log1p(
    0.0001 * (np.sqrt(2) ** np.arange(10))
).round(8).tolist()

TEST_RATIO: float = .2
LOOKAHEAD: int = 1            # hours to look ahead
BATCH_SIZE: int = 32
ACCUMULATION_STEPS: int = 4
NUM_EPOCHS: int = 1200
LR: float = 3e-5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# ── data utilities ──────────────────────────────────────────────

def ohlcv_to_percent_features(seq: np.ndarray) -> np.ndarray:
    """Return 7‑dim features per bar (Δ%, logVol, range%, std%)."""
    ohlc = seq[:, :4].astype(np.float32); vol = seq[:, 4].astype(np.float32)
    close = ohlc[:, 3]
    prev_close = np.concatenate(([close[0]], close[:-1]))
    pct = np.log1p((ohlc - prev_close[:, None]) / prev_close[:, None])                 # Δ% O/H/L/C
    v_log = np.log1p(vol);
    v_norm = (v_log - v_log.min()) / (v_log.max() - v_log.min() + 1e-9)
    range_pct = (ohlc[:,1] - ohlc[:,2]) / prev_close
    rolling_std = (pd.Series(close).rolling(14).std().fillna(0).values / prev_close
                   if len(close) >= 14 else np.zeros_like(close))
    feat = np.concatenate([pct, v_norm[:, None], range_pct[:, None], rolling_std[:, None]], axis=1)
    return feat.astype(np.float32)


def prepare_decoder_sequences_fixed(data: np.ndarray, seq_len: int = SEQ_LEN, lookahead: int = LOOKAHEAD):
    """
    FIXED: Prepare sequences for decoder training with dual targets.
    
    Key fixes:
    1. Pre-compute all features to avoid redundant calculations
    2. Fix next_diff logic for true autoregressive prediction
    3. Consistent sequence lengths for both targets
    4. Better threshold target encoding with timing info
    
    Returns:
    - X: input features [N, seq_len, 7]
    - targets: dict with 'next_diff' and 'thresholds'
    """
    print(f"Preparing decoder sequences (FIXED VERSION)...")
    print(f"  Data shape: {data.shape}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Lookahead: {lookahead}")
    
    # Pre-compute ALL features to avoid redundant calculations
    print(f"  Pre-computing features...")
    all_features = []
    for i in range(len(data) - seq_len + 1):
        chunk = data[i:i + seq_len]
        all_features.append(ohlcv_to_percent_features(chunk))
    all_features = np.array(all_features)  # [N, seq_len, 7]
    print(f"  Pre-computed features shape: {all_features.shape}")
    
    X, targets = [], {'next_diff': [], 'thresholds': []}
    thr = np.array(PCT_THRESHOLDS, np.float32)
    
    # Adjusted range to ensure consistent sequence lengths
    total_sequences = len(all_features) - lookahead
    print(f"  Total sequences to process: {total_sequences}")
    
    for i in range(total_sequences):
        if i % 10000 == 0:
            print(f"    Processing sequence {i+1}/{total_sequences}")
            
        # Input sequence
        features = all_features[i]  # [seq_len, 7]
        X.append(features)
        
        # Target 1: FIXED next-step diff prediction
        if i + 1 < len(all_features):
            next_features = all_features[i + 1]  # [seq_len, 7]
            # True next-step prediction: predict how features will change
            next_diff = next_features - features  # [seq_len, 7]
        else:
            # For last sequence, use zeros (shouldn't happen with adjusted range)
            next_diff = np.zeros_like(features)
        
        targets['next_diff'].append(next_diff.astype(np.float32))
        
        # Target 2: FIXED threshold exceedance (consistent with sequence length)
        # Create threshold targets for each position in sequence
        threshold_sequence = []
        
        for pos in range(seq_len):
            # For each position, check if future price exceeds thresholds
            if i + lookahead < len(data):
                current_idx = i + pos
                if current_idx < len(data):
                    current_price = data[current_idx, 0]  # Open price
                    future_idx = min(current_idx + lookahead, len(data) - 1)
                    future_price = data[future_idx, 3]  # Close price
                    
                    log_move = np.log(future_price / current_price) if current_price > 0 else 0.0
                    
                    pos_targets = []
                    for t in thr:
                        # UP threshold
                        up_exceeded = 1.0 if log_move >= t else 0.0
                        pos_targets.append(up_exceeded)
                        
                        # DOWN threshold  
                        down_exceeded = 1.0 if log_move <= -t else 0.0
                        pos_targets.append(down_exceeded)
                    
                    threshold_sequence.append(pos_targets)
                else:
                    # Padding for out-of-bounds positions
                    threshold_sequence.append([0.0] * 20)
            else:
                # Padding for sequences near end
                threshold_sequence.append([0.0] * 20)
        
        targets['thresholds'].append(np.array(threshold_sequence, dtype=np.float32))  # [seq_len, 20]
    
    # Convert to numpy arrays
    print(f"Converting to numpy arrays...")
    X = np.stack(X) if X else np.array([])
    for key in targets:
        if targets[key]:
            targets[key] = np.stack(targets[key])
        else:
            targets[key] = np.array([])
    
    print(f"Final shapes:")
    print(f"  X: {X.shape}")
    for key in targets:
        print(f"  {key}: {targets[key].shape}")
    
    # Verify shapes match
    if len(targets['next_diff']) > 0 and len(targets['thresholds']) > 0:
        assert targets['next_diff'].shape[:2] == targets['thresholds'].shape[:2], \
            f"Shape mismatch: next_diff {targets['next_diff'].shape} vs thresholds {targets['thresholds'].shape}"
        print(f"  ✅ Shape verification passed!")
            
    return X, targets


def load_decoder_sequences(csv: Union[str, Path]):
    """Load and prepare sequences for decoder training."""
    print(f"Loading data from: {csv}")
    df = (pd.read_csv(csv, parse_dates=['Timestamp'])
            .sort_values('Timestamp', ignore_index=True))
    print(f"Loaded {len(df)} rows")
    print(f"Date range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    X, targets = prepare_decoder_sequences_fixed(df[['Open','High','Low','Close','Volume']].values)
    split = int(len(X) * (1 - TEST_RATIO))
    
    print(f"Splitting data:")
    print(f"  Train samples: {split}")
    print(f"  Test samples: {len(X) - split}")
    print(f"  Test ratio: {TEST_RATIO:.1%}")
    
    # Convert to tensors and split
    print(f"Converting to tensors...")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    targets_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in targets.items()}
    
    # Split each target type
    train_X = X_tensor[:split]
    test_X = X_tensor[split:]
    train_targets = {k: v[:split] for k, v in targets_tensor.items()}
    test_targets = {k: v[split:] for k, v in targets_tensor.items()}
    
    return train_X, test_X, train_targets, test_targets


class DualOutputDecoderTransformer(nn.Module):
    """
    Decoder-based Transformer with dual outputs:
    1. Autoregressive OHLCV diff prediction
    2. Threshold exceedance probabilities
    """
    
    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 256,
        depth: int = 8,
        heads: int = 8,
        ff_dropout: float = 0.1,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        
        # Input projection
        self.proj = nn.Linear(input_dim, d_model, bias=False)
        
        # Decoder with causal mask (Decoder has causal=True by default)
        self.decoder = Decoder(
            dim=d_model,
            depth=depth,
            heads=heads,
            rotary_pos_emb=True,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout
        )
        
        self.norm = nn.LayerNorm(d_model)
        
        # Dual output heads
        # Head 1: Next-step diff prediction (autoregressive)
        self.diff_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_model // 2, input_dim)  # Predict 7-dim diff
        )
        
        # Head 2: Threshold probabilities
        self.threshold_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(), 
            nn.Dropout(ff_dropout),
            nn.Linear(d_model // 2, 20)  # 20 threshold probabilities
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, T, input_dim] → dual predictions
        Returns dict with 'next_diff' and 'thresholds'
        """
        # Project input features
        h = self.proj(x)  # [B, T, d_model]
        
        # Apply decoder with causal attention
        h = self.decoder(h)  # [B, T, d_model]
        h = self.norm(h)
        
        # Dual outputs
        return {
            'next_diff': self.diff_head(h),      # [B, T, 7] - autoregressive prediction
            'thresholds': self.threshold_head(h) # [B, T, 20] - threshold probabilities
        }


class WeightedDualLoss(nn.Module):
    """
    FIXED: Weighted loss function for dual outputs with conservative weighting.
    
    Fixes:
    1. More conservative position weighting (0.5 → 1.0 instead of 0.1 → 1.0)
    2. Better shape handling and validation
    3. Improved error messages
    """
    
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha  # Balance between diff and threshold losses
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        predictions: {'next_diff': [B, T, 7], 'thresholds': [B, T, 20]}
        targets: {'next_diff': [B, T, 7], 'thresholds': [B, T, 20]}
        """
        # Validate shapes
        pred_diff_shape = predictions['next_diff'].shape
        pred_thresh_shape = predictions['thresholds'].shape
        targ_diff_shape = targets['next_diff'].shape
        targ_thresh_shape = targets['thresholds'].shape
        
        assert pred_diff_shape == targ_diff_shape, \
            f"next_diff shape mismatch: pred {pred_diff_shape} vs target {targ_diff_shape}"
        assert pred_thresh_shape == targ_thresh_shape, \
            f"thresholds shape mismatch: pred {pred_thresh_shape} vs target {targ_thresh_shape}"
        
        B, T = pred_diff_shape[:2]
        
        # FIXED: More conservative position weights (0.5 → 1.0 instead of 0.1 → 1.0)
        pos_weights = torch.linspace(0.5, 1.0, T, device=predictions['next_diff'].device)
        pos_weights = pos_weights.view(1, T, 1)  # [1, T, 1]
        
        # Loss 1: Next diff prediction (MSE)
        diff_loss = F.mse_loss(
            predictions['next_diff'], 
            targets['next_diff'], 
            reduction='none'
        )  # [B, T, 7]
        
        # Apply position weighting
        diff_loss = (diff_loss * pos_weights).mean()
        
        # Loss 2: Threshold probabilities (BCE)
        threshold_loss = F.binary_cross_entropy_with_logits(
            predictions['thresholds'],
            targets['thresholds'],
            reduction='none'
        )  # [B, T, 20]
        
        # Apply position weighting
        pos_weights_thresh = pos_weights.expand(-1, -1, 20)  # [1, T, 20]
        threshold_loss = (threshold_loss * pos_weights_thresh).mean()
        
        # Combined weighted loss
        total_loss = self.alpha * diff_loss + (1 - self.alpha) * threshold_loss
        
        return total_loss, {'diff_loss': diff_loss.item(), 'threshold_loss': threshold_loss.item()}


def _batches(X, Y, bs):
    """Generate batches for training."""
    idx = torch.randperm(len(X))
    for i in range(0, len(X), bs):
        sel = idx[i:i+bs]
        batch_X = X[sel].to(DEVICE)
        batch_Y = {k: v[sel].to(DEVICE) for k, v in Y.items()}
        yield batch_X, batch_Y


def train_dual_model(model, trX, trY, teX, teY):
    """Train the dual-output decoder model."""
    print(f"Starting training...")
    print(f"  Device: {DEVICE}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  Training samples: {len(trX)}")
    print(f"  Validation samples: {len(teX)}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Accumulation steps: {ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LR}")
    print(f"  Max epochs: {NUM_EPOCHS}")
    
    # Validate data shapes
    print(f"  Data shape validation:")
    print(f"    trX: {trX.shape}")
    for key in trY:
        print(f"    trY[{key}]: {trY[key].shape}")
    
    model.to(DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, 'min', 0.8, 25, cooldown=5, min_lr=1e-7, verbose=True
    )
    
    loss_fn = WeightedDualLoss(alpha=0.5)  # Equal weight to both losses
    print(f"  Loss function: WeightedDualLoss (alpha=0.5, conservative weighting)")
    
    best = np.inf
    patience = 0
    print(f"\nStarting training loop...")
    
    for ep in range(1, NUM_EPOCHS + 1):
        model.train()
        tot_loss = 0
        tot_diff_loss = 0
        tot_thresh_loss = 0
        n = 0
        opt.zero_grad()
        
        # Progress tracking
        num_batches = len(trX) // BATCH_SIZE + (1 if len(trX) % BATCH_SIZE else 0)
        if ep == 1:
            print(f"  Batches per epoch: {num_batches}")
        
        for i, (bx, by) in enumerate(_batches(trX, trY, BATCH_SIZE)):
            if ep <= 3 and i % 50 == 0:  # Show progress for first few epochs
                print(f"    Epoch {ep}, batch {i+1}/{num_batches}")
            
            try:
                with torch.cuda.amp.autocast():
                    pred = model(bx)
                    loss, loss_dict = loss_fn(pred, by)
                    loss = loss / ACCUMULATION_STEPS
                
                scaler.scale(loss).backward()
                tot_loss += loss.item() * ACCUMULATION_STEPS
                tot_diff_loss += loss_dict['diff_loss']
                tot_thresh_loss += loss_dict['threshold_loss']
                n += 1
                
                if (i + 1) % ACCUMULATION_STEPS == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
                    
            except Exception as e:
                print(f"    Error in batch {i}: {e}")
                print(f"    Batch shapes: bx={bx.shape}, by_keys={list(by.keys())}")
                for k, v in by.items():
                    print(f"      by[{k}]: {v.shape}")
                raise
        
        # Handle remaining gradients
        if n % ACCUMULATION_STEPS != 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
        
        trL = tot_loss / n
        trDL = tot_diff_loss / n
        trTL = tot_thresh_loss / n
        
        # Validation
        model.eval()
        vL = 0
        vDL = 0
        vTL = 0
        vN = 0
        
        with torch.no_grad():
            for bx, by in _batches(teX, teY, BATCH_SIZE * 2):
                try:
                    with torch.cuda.amp.autocast():
                        pred = model(bx)
                        loss, loss_dict = loss_fn(pred, by)
                    vL += loss.item()
                    vDL += loss_dict['diff_loss']
                    vTL += loss_dict['threshold_loss']
                    vN += 1
                except Exception as e:
                    print(f"    Validation error: {e}")
                    continue
        
        if vN > 0:
            vL /= vN
            vDL /= vN
            vTL /= vN
        else:
            print("    Warning: No valid validation batches!")
            vL = vDL = vTL = float('inf')
            
        sched.step(vL)
        
        print(f"Ep{ep:4d} train {trL:.4f} (diff:{trDL:.4f}, thresh:{trTL:.4f}) | "
              f"val {vL:.4f} (diff:{vDL:.4f}, thresh:{vTL:.4f}) | lr {opt.param_groups[0]['lr']:.1e}")
        
        if vL < best:
            best = vL
            patience = 0
            print(f"  *** New best validation loss: {vL:.4f} ***")
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': vL,
                'val_diff_loss': vDL,
                'val_threshold_loss': vTL,
                'epoch': ep
            }, 'best_decoder_model_fixed.pth')
            print(f"  Model saved to: best_decoder_model_fixed.pth")
        else:
            patience += 1
            if patience > 60:
                print(f'Early stopping after {ep} epochs (patience: {patience})')
                break
            elif patience % 10 == 0:
                print(f"  No improvement for {patience} epochs...")
    
    print(f"Training completed!")
    print(f"  Best validation loss: {best:.4f}")
    print(f"  Final epoch: {ep}")


def build_decoder_model(input_dim: int = 7) -> DualOutputDecoderTransformer:
    """Build decoder model with good defaults."""
    return DualOutputDecoderTransformer(
        input_dim=input_dim,
        d_model=256,
        depth=8,
        heads=8
    )


# ── main ───────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*80)
    print("DECODER-BASED TRANSFORMER FOR EUR/USD PREDICTION (FIXED)")
    print("="*80)
    
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data_EURUSD_Hour.csv')
    ap.add_argument('--weights')
    args = ap.parse_args()
    
    print(f"Configuration:")
    print(f"  CSV file: {args.csv}")
    print(f"  Weights: {args.weights if args.weights else 'None (training from scratch)'}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Lookahead: {LOOKAHEAD}")
    print(f"  Thresholds: {len(PCT_THRESHOLDS)} levels")
    print()
    
    # Load data
    trX, teX, trY, teY = load_decoder_sequences(args.csv)
    print()
    
    # Build model
    print("Building model...")
    model = build_decoder_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Load weights if provided
    if args.weights:
        print(f"Loading weights from: {args.weights}")
        ckpt = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        if 'epoch' in ckpt:
            print(f"  Loaded from epoch: {ckpt['epoch']}")
        if 'val_loss' in ckpt:
            print(f"  Previous validation loss: {ckpt['val_loss']:.4f}")
    print()
    
    # Start training
    train_dual_model(model, trX, trY, teX, teY)
    
    print("="*80)
    print("TRAINING COMPLETED")
    print("="*80)
