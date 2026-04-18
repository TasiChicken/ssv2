import os
import csv
import json
import time
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

from models.videomae import build_finetune_model
from dataset.ssv2_dataset import build_dataset
from utils.train_utils import load_config, AverageMeter, accuracy, format_time


def parse_args():
    parser = argparse.ArgumentParser('VideoMAE Full Evaluation')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to fine-tuned checkpoint')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Optional override for evaluation output dir')
    return parser.parse_args()


def safe_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def build_label_maps(labels_json_path):
    with open(labels_json_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)   # clean_template -> class_id

    id_to_name = {int(v): k for k, v in label_map.items()}
    return label_map, id_to_name


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes, id_to_name, use_amp=True):
    model.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc3_meter = AverageMeter()
    acc5_meter = AverageMeter()
    acc10_meter = AverageMeter()

    valid_samples = 0
    invalid_samples = 0

    all_labels = []
    all_preds = []
    all_topk_indices = []
    all_topk_probs = []
    all_max_probs = []
    all_is_correct = []
    sample_rows = []

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    start_time = time.time()

    for step, batch in enumerate(dataloader):
        if len(batch) == 3:
            videos, labels, metas = batch
        else:
            raise ValueError("Expected dataloader to return (videos, labels, metas)")

        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        valid_mask = labels >= 0
        batch_invalid = int((~valid_mask).sum().item())
        invalid_samples += batch_invalid

        if valid_mask.sum() == 0:
            continue

        videos = videos[valid_mask]
        labels = labels[valid_mask]

        # metas 會被 DataLoader 自動整理成 dict of lists
        filtered_metas = {}
        for k, v in metas.items():
            filtered = []
            for i, keep in enumerate(valid_mask.cpu().tolist()):
                if keep:
                    filtered.append(v[i])
            filtered_metas[k] = filtered

        if use_amp and device.type == 'cuda':
            with autocast('cuda'):
                logits = model(videos)
                loss = F.cross_entropy(logits, labels)
        else:
            logits = model(videos)
            loss = F.cross_entropy(logits, labels)

        probs = F.softmax(logits, dim=1)

        top1, top3, top5, top10 = accuracy(logits, labels, topk=(1, 3, 5, 10))

        batch_size = videos.size(0)
        valid_samples += batch_size

        loss_meter.update(loss.item(), batch_size)
        acc1_meter.update(top1.item(), batch_size)
        acc3_meter.update(top3.item(), batch_size)
        acc5_meter.update(top5.item(), batch_size)
        acc10_meter.update(top10.item(), batch_size)

        max_probs, preds = probs.max(dim=1)

        k_for_save = min(10, probs.size(1))
        topk_probs, topk_indices = probs.topk(k_for_save, dim=1, largest=True, sorted=True)

        labels_np = labels.cpu().numpy()
        preds_np = preds.cpu().numpy()
        max_probs_np = max_probs.cpu().numpy()
        topk_indices_np = topk_indices.cpu().numpy()
        topk_probs_np = topk_probs.cpu().numpy()

        for i in range(batch_size):
            true_id = int(labels_np[i])
            pred_id = int(preds_np[i])
            is_correct = int(true_id == pred_id)

            class_total[true_id] += 1
            class_correct[true_id] += is_correct
            confusion[true_id, pred_id] += 1

            all_labels.append(true_id)
            all_preds.append(pred_id)
            all_topk_indices.append(topk_indices_np[i].tolist())
            all_topk_probs.append(topk_probs_np[i].tolist())
            all_max_probs.append(float(max_probs_np[i]))
            all_is_correct.append(is_correct)

            topk_names = [id_to_name.get(int(x), f"class_{x}") for x in topk_indices_np[i]]

            sample_rows.append({
                'index': filtered_metas.get('index', [''])[i],
                'video_id': filtered_metas.get('video_id', [''])[i],
                'template': filtered_metas.get('template', [''])[i],
                'clean_template': filtered_metas.get('clean_template', [''])[i],
                'video_path': filtered_metas.get('video_path', [''])[i],
                'load_error': filtered_metas.get('load_error', [''])[i],
                'true_label_id': true_id,
                'true_label_name': id_to_name.get(true_id, f"class_{true_id}"),
                'pred_label_id': pred_id,
                'pred_label_name': id_to_name.get(pred_id, f"class_{pred_id}"),
                'correct': is_correct,
                'top1_prob': float(max_probs_np[i]),
                'top10_label_ids': topk_indices_np[i].tolist(),
                'top10_label_names': topk_names,
                'top10_probs': [float(x) for x in topk_probs_np[i].tolist()],
            })

        if (step + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"[{step+1}/{len(dataloader)}] "
                f"Loss {loss_meter.avg:.4f} | "
                f"Acc@1 {acc1_meter.avg:.2f}% | "
                f"Acc@5 {acc5_meter.avg:.2f}% | "
                f"Valid {valid_samples} | Invalid {invalid_samples} | "
                f"{format_time(elapsed)}"
            )

    elapsed = time.time() - start_time

    per_class_results = []
    for class_id in range(num_classes):
        total = class_total.get(class_id, 0)
        correct = class_correct.get(class_id, 0)
        acc = correct / total if total > 0 else 0.0
        per_class_results.append({
            'class_id': class_id,
            'class_name': id_to_name.get(class_id, f'class_{class_id}'),
            'num_samples': total,
            'num_correct': correct,
            'accuracy': acc,
        })

    present_class_accs = [x['accuracy'] for x in per_class_results if x['num_samples'] > 0]
    mean_class_accuracy = float(np.mean(present_class_accs)) if present_class_accs else 0.0

    all_is_correct_np = np.array(all_is_correct, dtype=np.int64)
    all_max_probs_np = np.array(all_max_probs, dtype=np.float32)

    if len(all_max_probs_np) > 0:
        mean_max_prob = float(all_max_probs_np.mean())
        correct_conf = float(all_max_probs_np[all_is_correct_np == 1].mean()) if np.any(all_is_correct_np == 1) else 0.0
        wrong_conf = float(all_max_probs_np[all_is_correct_np == 0].mean()) if np.any(all_is_correct_np == 0) else 0.0
    else:
        mean_max_prob = 0.0
        correct_conf = 0.0
        wrong_conf = 0.0

    confusion_pairs = []
    for true_id in range(num_classes):
        for pred_id in range(num_classes):
            if true_id == pred_id:
                continue
            cnt = int(confusion[true_id, pred_id])
            if cnt > 0:
                confusion_pairs.append({
                    'true_label_id': true_id,
                    'true_label_name': id_to_name.get(true_id, f'class_{true_id}'),
                    'pred_label_id': pred_id,
                    'pred_label_name': id_to_name.get(pred_id, f'class_{pred_id}'),
                    'count': cnt,
                })
    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

    summary = {
        'num_samples': int(valid_samples),
        'num_invalid_samples': int(invalid_samples),
        'num_classes': int(num_classes),
        'num_classes_present': int(sum(1 for x in per_class_results if x['num_samples'] > 0)),
        'val_loss': float(loss_meter.avg),
        'top1': float(acc1_meter.avg),
        'top3': float(acc3_meter.avg),
        'top5': float(acc5_meter.avg),
        'top10': float(acc10_meter.avg),
        'macro_top1': float(mean_class_accuracy * 100.0),
        'mean_class_accuracy': float(mean_class_accuracy),
        'mean_max_prob': float(mean_max_prob),
        'mean_correct_confidence': float(correct_conf),
        'mean_wrong_confidence': float(wrong_conf),
        'eval_time_sec': float(elapsed),
        'eval_time_hms': format_time(elapsed),
    }

    return summary, per_class_results, sample_rows, confusion, confusion_pairs


def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_predictions_csv(path, rows):
    fieldnames = [
        'index',
        'video_id',
        'template',
        'clean_template',
        'video_path',
        'load_error',
        'true_label_id',
        'true_label_name',
        'pred_label_id',
        'pred_label_name',
        'correct',
        'top1_prob',
        'top10_label_ids',
        'top10_label_names',
        'top10_probs',
    ]
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_to_write = dict(row)
            row_to_write['top10_label_ids'] = json.dumps(row_to_write['top10_label_ids'], ensure_ascii=False)
            row_to_write['top10_label_names'] = json.dumps(row_to_write['top10_label_names'], ensure_ascii=False)
            row_to_write['top10_probs'] = json.dumps(row_to_write['top10_probs'], ensure_ascii=False)
            writer.writerow(row_to_write)


def print_report(summary, per_class_results, confusion_pairs):
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"Valid samples            : {summary['num_samples']}")
    print(f"Invalid samples          : {summary['num_invalid_samples']}")
    print(f"Validation loss          : {summary['val_loss']:.4f}")
    print(f"Top-1 accuracy           : {summary['top1']:.2f}%")
    print(f"Top-3 accuracy           : {summary['top3']:.2f}%")
    print(f"Top-5 accuracy           : {summary['top5']:.2f}%")
    print(f"Top-10 accuracy          : {summary['top10']:.2f}%")
    print(f"Macro Top-1              : {summary['macro_top1']:.2f}%")
    print(f"Mean class accuracy      : {summary['mean_class_accuracy']:.4f}")
    print(f"Mean max prob            : {summary['mean_max_prob']:.4f}")
    print(f"Mean correct confidence  : {summary['mean_correct_confidence']:.4f}")
    print(f"Mean wrong confidence    : {summary['mean_wrong_confidence']:.4f}")
    print(f"Eval time                : {summary['eval_time_hms']}")
    print("=" * 70)

    ranked = sorted(per_class_results, key=lambda x: x['accuracy'], reverse=True)
    non_zero = [x for x in ranked if x['num_samples'] > 0]

    print("\nTop 10 classes:")
    for row in non_zero[:10]:
        print(f"  [{row['class_id']:3d}] {row['accuracy']*100:6.2f}% "
              f"({row['num_correct']:3d}/{row['num_samples']:3d}) {row['class_name']}")

    print("\nBottom 10 classes:")
    for row in non_zero[-10:]:
        print(f"  [{row['class_id']:3d}] {row['accuracy']*100:6.2f}% "
              f"({row['num_correct']:3d}/{row['num_samples']:3d}) {row['class_name']}")

    print("\nTop 15 confusion pairs:")
    for row in confusion_pairs[:15]:
        print(f"  {row['true_label_name']}  -->  {row['pred_label_name']} : {row['count']}")


def main():
    args = parse_args()
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = config['data']['num_workers']

    print("\nBuilding validation dataset...")
    val_dataset = build_dataset(config, mode='val')
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("\nBuilding model...")
    model = build_finetune_model(config)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model = model.to(device)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Checkpoint epoch : {ckpt.get('epoch', '?')}")

    _, id_to_name = build_label_maps(config['data']['labels_json'])
    num_classes = config['model']['num_classes']

    summary, per_class_results, sample_rows, confusion, confusion_pairs = evaluate(
        model=model,
        dataloader=val_loader,
        device=device,
        num_classes=num_classes,
        id_to_name=id_to_name,
        use_amp=config['finetune']['use_amp'],
    )

    summary['checkpoint'] = args.checkpoint
    summary['checkpoint_epoch'] = ckpt.get('epoch', None)
    summary['config'] = args.config

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(config['finetune']['output_dir'], 'eval')

    ensure_dir(output_dir)

    summary_path = os.path.join(output_dir, 'eval_summary.json')
    per_class_path = os.path.join(output_dir, 'per_class_results.json')
    pred_csv_path = os.path.join(output_dir, 'predictions.csv')
    confusion_npy_path = os.path.join(output_dir, 'confusion_matrix.npy')
    confusion_pairs_path = os.path.join(output_dir, 'confusion_top_pairs.json')

    save_json(summary_path, summary)
    save_json(per_class_path, per_class_results)
    save_predictions_csv(pred_csv_path, sample_rows)
    np.save(confusion_npy_path, confusion)
    save_json(confusion_pairs_path, confusion_pairs[:200])

    print_report(summary, per_class_results, confusion_pairs)

    print("\nSaved files:")
    print(f"  {summary_path}")
    print(f"  {per_class_path}")
    print(f"  {pred_csv_path}")
    print(f"  {confusion_npy_path}")
    print(f"  {confusion_pairs_path}")


if __name__ == '__main__':
    main()