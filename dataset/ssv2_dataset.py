"""
SSv2 Dataset for VideoMAE training.
Loads .webm video files, applies temporal sampling and spatial transforms.
Supports both pre-training (no labels) and fine-tuning (with labels).
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import VideoTransform, uniform_temporal_subsample


def _load_video_cv2(video_path, num_frames):
    """Load video frames using OpenCV (fallback, always available on Windows)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 300  # fallback estimate

    indices = uniform_temporal_subsample(total, num_frames)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        elif len(frames) > 0:
            frames.append(frames[-1].copy())  # repeat last frame
        else:
            raise RuntimeError(f"Cannot read frame {idx} from {video_path}")

    cap.release()
    return np.stack(frames, axis=0)  # [T, H, W, C]


def _load_video_decord(video_path, num_frames):
    """Load video frames using decord (faster, GPU-capable)."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        indices = uniform_temporal_subsample(total, num_frames)
        frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C] RGB
        return frames
    except ImportError:
        return _load_video_cv2(video_path, num_frames)
    except Exception:
        return _load_video_cv2(video_path, num_frames)


class SSv2Dataset(Dataset):
    """
    Something-Something V2 Dataset.

    Args:
        root_dir: Root directory containing videos/ and labels/
        split_json: Path to train.json or validation.json
        labels_json: Path to labels.json (template → class_id mapping)
        num_frames: Number of frames to sample per video
        img_size: Spatial resolution for frames
        mode: 'train' or 'val'
        subset_size: If set, randomly sample this many videos from the split
        use_decord: Whether to use decord for video loading
    """
    def __init__(self, root_dir, split_json, labels_json,
                 num_frames=8, img_size=128, mode='train',
                 subset_size=None, use_decord=True):
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, 'videos')
        self.num_frames = num_frames
        self.mode = mode
        self.use_decord = use_decord
        self.transform = VideoTransform(img_size=img_size, mode=mode)

        # Load label mapping: template → class_id
        with open(labels_json, 'r') as f:
            self.label_map = json.load(f)

        # Load split annotations
        with open(split_json, 'r') as f:
            self.annotations = json.load(f)

        # Subset if requested
        if subset_size and subset_size < len(self.annotations):
            rng = np.random.RandomState(42)  # fixed seed for reproducibility
            indices = rng.choice(len(self.annotations), subset_size, replace=False)
            self.annotations = [self.annotations[i] for i in sorted(indices)]

        # Filter out samples whose video files don't exist (optional, can be slow)
        # self.annotations = [a for a in self.annotations if os.path.exists(
        #     os.path.join(self.video_dir, f"{a['id']}.webm"))]

        print(f"[SSv2Dataset] Loaded {len(self.annotations)} samples "
              f"(mode={mode}, num_frames={num_frames}, img_size={img_size})")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        video_id = ann['id']
        template = ann.get('template', '')

        # Get class label from template
        # train.json templates use "[something]" but labels.json uses "something"
        clean_template = template.replace('[', '').replace(']', '')
        label = int(self.label_map.get(clean_template, -1))

        # Load video
        video_path = os.path.join(self.video_dir, f"{video_id}.webm")

        meta = {
            'index': idx,
            'video_id': str(video_id),
            'template': template,
            'clean_template': clean_template,
            'video_path': video_path,
        }

        try:
            if self.use_decord:
                frames = _load_video_decord(video_path, self.num_frames)
            else:
                frames = _load_video_cv2(video_path, self.num_frames)
        except Exception as e:
            # On failure, return a zero tensor and label -1
            print(f"[WARNING] Failed to load {video_path}: {e}")
            dummy = np.zeros((self.num_frames, self.transform.img_size, self.transform.img_size, 3), dtype=np.uint8)
            tensor = self.transform(dummy)
            return tensor, -1

        # Apply transforms: [T, H, W, C] → [C, T, H, W]
        tensor = self.transform(frames)
        meta['load_error'] = ''
        return tensor, label, meta


class SSv2PretrainDataset(SSv2Dataset):
    """
    SSv2 Dataset for pre-training (masked autoencoder).
    Returns video tensor + boolean mask (no labels needed for reconstruction).
    """
    def __init__(self, root_dir, split_json, labels_json,
                 num_frames=8, img_size=128, mode='train',
                 subset_size=None, use_decord=True,
                 mask_ratio=0.9, patch_size=16, tubelet_size=2,
                 masking_mode='tube'):
        super().__init__(root_dir, split_json, labels_json,
                         num_frames, img_size, mode, subset_size, use_decord)
        self.mask_ratio = mask_ratio
        self.masking_mode = masking_mode
        self.patch_size_px = patch_size
        self.tubelet_size_t = tubelet_size
        # Calculate mask dimensions
        self.t_patches = num_frames // tubelet_size
        self.h_patches = img_size // patch_size
        self.w_patches = img_size // patch_size
        self.num_patches_per_frame = self.h_patches * self.w_patches
        self.total_patches = self.t_patches * self.num_patches_per_frame
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)

        if masking_mode == 'flow_guided_persistent':
            from models.masking import FlowGuidedPersistentMaskingGenerator
            self.flow_mask_gen = FlowGuidedPersistentMaskingGenerator(
                input_size=(self.t_patches, self.h_patches, self.w_patches),
                mask_ratio=mask_ratio,
                patch_size=patch_size,
                tubelet_size=tubelet_size,
            )

    def _generate_mask(self):
        """Generate tube masking: same spatial mask across all temporal patches."""
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        # Tile across temporal dimension (tube masking)
        mask = np.tile(mask_per_frame, (self.t_patches,)).flatten()
        return torch.from_numpy(mask).bool()  # True = masked

    def _extract_gray_frames(self, tensor):
        """Recover uint8 grayscale frames from a normalised tensor for flow."""
        import cv2
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        unnorm = ((tensor * std + mean).clamp(0, 1) * 255.0)
        frames = unnorm.permute(1, 2, 3, 0).numpy().astype(np.uint8)  # [T,H,W,C]
        gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])
        return gray  # [T, H, W]

    def __getitem__(self, idx):
        tensor, _, _ = super().__getitem__(idx)

        if self.masking_mode == 'flow_guided_persistent':
            gray = self._extract_gray_frames(tensor)
            mask_np = self.flow_mask_gen(gray)
            mask = torch.from_numpy(mask_np).bool()
        else:
            mask = self._generate_mask()

        return tensor, mask


def build_pretraining_dataset(config):
    """Build pre-training dataset from config dict."""
    masking_mode = config['masking'].get('mode', 'tube')
    dataset = SSv2PretrainDataset(
        root_dir=config['data']['root_dir'],
        split_json=config['data']['train_json'],
        labels_json=config['data']['labels_json'],
        num_frames=config['model']['num_frames'],
        img_size=config['model']['img_size'],
        mode='train',
        subset_size=config['data'].get('subset_size'),
        mask_ratio=config['masking']['ratio'],
        patch_size=config['model']['patch_size'],
        tubelet_size=config['model']['tubelet_size'],
        masking_mode=masking_mode,
    )
    return dataset


def build_dataset(config, mode='train'):
    """Build fine-tuning dataset from config dict."""
    if mode == 'train':
        split_json = config['data']['train_json']
        subset_size = config['data'].get('subset_size')
    else:
        split_json = config['data']['val_json']
        subset_size = config['data'].get('val_subset_size')

    dataset = SSv2Dataset(
        root_dir=config['data']['root_dir'],
        split_json=split_json,
        labels_json=config['data']['labels_json'],
        num_frames=config['model']['num_frames'],
        img_size=config['model']['img_size'],
        mode=mode,
        subset_size=subset_size,
    )
    return dataset
