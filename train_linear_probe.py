"""
VideoMAE Linear Probe Script (Stage 2 - Frozen Encoder).
Evaluates representation quality by training only a linear classifier on top
of a frozen pre-trained encoder.

Usage:
    python train_linear_probe.py --config configs/videomae_tiny_ssv2_linear_probe.yaml
    python train_linear_probe.py --config configs/videomae_tiny_ssv2_linear_probe.yaml --pretrain output/pretrain/checkpoint_best.pth
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from models.videomae import build_finetune_model, build_linear_probe_model
from dataset.ssv2_dataset import build_dataset
from utils.train_utils import (
    load_config, AverageMeter, CosineScheduler,
    save_checkpoint, accuracy,
    TensorBoardLogger, format_time,
)


def parse_args():
    parser = argparse.ArgumentParser('VideoMAE Linear Probe')
    parser.add_argument('--config', type=str,
                        default='configs/videomae_tiny_ssv2_linear_probe.yaml')
    parser.add_argument('--pretrain', type=str, default=None,
                        help='Path to pre-trained checkpoint')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume linear probe from checkpoint')
    return parser.parse_args()



def train_one_epoch(model, dataloader, criterion, optimizer, scheduler,
                    scaler, device, epoch, config, logger, global_step):
    """Train for one epoch (only the linear head is updated)."""
    model.eval()          # keep BN/dropout frozen in the encoder
    model.head.train()    # only the linear head in train mode

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    grad_accum = config['linear_probe']['gradient_accumulation']
    log_freq = config['linear_probe']['log_freq']
    use_amp = config['linear_probe']['use_amp'] and device.type == 'cuda'

    optimizer.zero_grad()
    start_time = time.time()

    for step, (videos, labels) in enumerate(dataloader):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Skip invalid samples
        valid = labels >= 0
        if valid.sum() == 0:
            continue
        videos = videos[valid]
        labels = labels[valid]

        # Forward (encoder is frozen, no grad needed for it)
        if use_amp:
            with autocast('cuda'):
                logits = model(videos)
                loss = criterion(logits, labels) / grad_accum
        else:
            logits = model(videos)
            loss = criterion(logits, labels) / grad_accum

        # Backward
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (step + 1) % grad_accum == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=5.0)
                optimizer.step()

            optimizer.zero_grad()
            lr = scheduler.step()
            global_step += 1

            if logger:
                logger.log_scalar('linear_probe/train_loss',
                                  loss.item() * grad_accum, global_step)
                logger.log_scalar('linear_probe/lr', lr, global_step)

        # Metrics
        with torch.no_grad():
            acc1, acc5 = accuracy(logits, labels, topk=(1, 5))

        loss_meter.update(loss.item() * grad_accum, videos.size(0))
        acc1_meter.update(acc1.item(), videos.size(0))
        acc5_meter.update(acc5.item(), videos.size(0))

        if (step + 1) % log_freq == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (step + 1) * (len(dataloader) - step - 1)
            print(f"  Epoch [{epoch}] Step [{step+1}/{len(dataloader)}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"Acc@1: {acc1_meter.avg:.1f}% "
                  f"Acc@5: {acc5_meter.avg:.1f}% "
                  f"LR: {scheduler.get_lr():.2e} "
                  f"ETA: {format_time(eta)}")

    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg, global_step


@torch.no_grad()
def validate(model, dataloader, criterion, device, config):
    """Validate on validation set."""
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    use_amp = config['linear_probe']['use_amp'] and device.type == 'cuda'

    for videos, labels in dataloader:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        valid = labels >= 0
        if valid.sum() == 0:
            continue
        videos = videos[valid]
        labels = labels[valid]

        if use_amp:
            with autocast('cuda'):
                logits = model(videos)
                loss = criterion(logits, labels)
        else:
            logits = model(videos)
            loss = criterion(logits, labels)

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        loss_meter.update(loss.item(), videos.size(0))
        acc1_meter.update(acc1.item(), videos.size(0))
        acc5_meter.update(acc5.item(), videos.size(0))

    return loss_meter.avg, acc1_meter.avg, acc5_meter.avg


def main():
    args = parse_args()
    config = load_config(args.config)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Build datasets
    print("\n Building datasets...")
    train_dataset = build_dataset(config, mode='train')
    val_dataset = build_dataset(config, mode='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['linear_probe']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if config['data']['num_workers'] > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['linear_probe']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=False,
    )

    # Build model (frozen encoder + fresh linear head)
    print("\n Building model...")
    pretrain_path = args.pretrain or config['linear_probe'].get('pretrain_ckpt')
    model = build_linear_probe_model(config, pretrain_path)
    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {n_total / 1e6:.2f}M")
    print(f"  Trainable params: {n_trainable / 1e6:.4f}M  (linear head only)")

    # Loss function (no label smoothing for linear probe — keep it clean)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config['linear_probe'].get('label_smoothing', 0.0))

    # Optimizer — only the linear head parameters
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=config['linear_probe']['lr'],
        weight_decay=config['linear_probe']['weight_decay'],
        betas=(0.9, 0.999),
    )

    # Scheduler
    steps_per_epoch = len(train_loader) // config['linear_probe']['gradient_accumulation']
    scheduler = CosineScheduler(
        optimizer,
        base_lr=config['linear_probe']['lr'],
        min_lr=config['linear_probe']['min_lr'],
        epochs=config['linear_probe']['epochs'],
        warmup_epochs=config['linear_probe']['warmup_epochs'],
        steps_per_epoch=max(1, steps_per_epoch),
    )

    # Mixed precision
    scaler = GradScaler('cuda', enabled=config['linear_probe']['use_amp'] and device.type == 'cuda')

    # Logger
    log_dir = os.path.join(config['linear_probe']['output_dir'], 'logs')
    logger = TensorBoardLogger(log_dir)

    # Resume
    start_epoch = 0
    best_acc = 0.0
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        global_step = ckpt.get('global_step', 0)
        print(f"  → Resumed from epoch {start_epoch}, best_acc={best_acc:.1f}%")

    # Training
    print(f"\nStarting linear probe for {config['linear_probe']['epochs']} epochs")
    print(f"   Effective batch size: "
          f"{config['linear_probe']['batch_size'] * config['linear_probe']['gradient_accumulation']}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print()

    for epoch in range(start_epoch, config['linear_probe']['epochs']):
        epoch_start = time.time()

        # Train
        train_loss, train_acc1, train_acc5, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, config, logger, global_step)

        # Validate
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, device, config)

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch} [{format_time(epoch_time)}] | "
              f"Train Loss: {train_loss:.4f} Acc@1: {train_acc1:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc@1: {val_acc1:.1f}% Acc@5: {val_acc5:.1f}%")

        # Log
        if logger:
            logger.log_scalar('linear_probe/val_loss', val_loss, epoch)
            logger.log_scalar('linear_probe/val_acc1', val_acc1, epoch)
            logger.log_scalar('linear_probe/val_acc5', val_acc5, epoch)
            logger.log_scalar('linear_probe/train_acc1', train_acc1, epoch)

        # Save
        is_best = val_acc1 > best_acc
        best_acc = max(val_acc1, best_acc)

        if (epoch + 1) % config['linear_probe']['save_freq'] == 0 or is_best:
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
                'global_step': global_step,
                'config': config,
            }
            save_checkpoint(state, config['linear_probe']['output_dir'],
                            f'checkpoint_epoch{epoch}.pth')
            if is_best:
                save_checkpoint(state, config['linear_probe']['output_dir'],
                                'checkpoint_best.pth')
                print(f"  New best! Val Acc@1: {val_acc1:.1f}%")

    logger.close()
    print(f"\n Linear probe complete! Best Val Acc@1: {best_acc:.1f}%")



if __name__ == '__main__':
    main()