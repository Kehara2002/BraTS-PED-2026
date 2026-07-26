#!/usr/bin/env python3
"""
BraTS-PED Task 02 inference, ported from predict_test_masks.ipynb.

Four binary specialists, each predicting one tumour class, merged by priority
in original image space, with 10-augmentation TTA and majority voting.

Container contract imposed by the Synapse evaluation harness:
  /input   read-only, one directory per case holding four modality NIfTIs
  /output  read-write, receives exactly one label map per case
  no network at runtime, so weights and SynthStrip are baked into the image

Deviations from the notebook, all forced by that contract:
  * SynthStrip runs in-process instead of via `docker run`
  * TTA scratch files go to /tmp rather than under the output directory
  * predictions are written flat to /output, not into a pred_full/ subfolder
  * the pandas/matplotlib summary export is dropped
Model architecture, transforms, inference parameters, merge priority, TTA
augmentations, voting, and post-processing are unchanged.
"""

import argparse
import copy
import gc
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from scipy import ndimage
from scipy.ndimage import rotate as scipy_rotate

torch.backends.cudnn.benchmark = True

from monai.utils import set_determinism
from monai.data import Dataset, set_track_meta, MetaTensor
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    NormalizeIntensityd, CropForegroundd, Invertd,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR as _SwinUNETR

set_track_meta(True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("brats-ped")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Baked-in asset locations ─────────────────────────────────────────────────
MODEL_DIR = Path("/model/models")
CKPT_ET = MODEL_DIR / "best_swinunetr_et_new.pth"
CKPT_NET = MODEL_DIR / "best_swinunetr_net_new.pth"
CKPT_ED = MODEL_DIR / "best_3dunet_ed.pth"
CKPT_CC = MODEL_DIR / "best_3dunet_cc.pth"

SYNTHSTRIP_HOME = Path(os.environ.get("FREESURFER_HOME", "/opt/synthstrip"))
SYNTHSTRIP_BIN = SYNTHSTRIP_HOME / "mri_synthstrip"

# ── Configuration, carried over verbatim ─────────────────────────────────────
SEED = 42
set_determinism(seed=SEED)
USE_AMP = True
IMAGE_KEYS = ["t1n", "t1c", "t2w", "t2f"]
LABEL_NAMES = {0: "BG", 1: "ET", 2: "NET", 3: "CC", 4: "ED"}
NUM_CLASSES = 5

SKIP_DESKULL = False
SKULL_BORDER = 1
SKULL_NO_CSF = False

SWIN_FEATURE_SIZE = 48
SWIN_DEPTHS = (2, 2, 2, 2)
SWIN_NUM_HEADS = (3, 6, 12, 24)
SWIN_DROP_RATE = 0.0
SWIN_ATTN_DROP = 0.0
SWIN_DROP_PATH = 0.0

ED_PATCH = (96, 96, 96);  ED_SW_ROI = (96, 96, 96);  ED_SW_OVERLAP = 0.25; ED_SW_BATCH = 4
CC_PATCH = (64, 64, 64);  CC_SW_ROI = (64, 64, 64);  CC_SW_OVERLAP = 0.50; CC_SW_BATCH = 4
ET_PATCH = (96, 96, 96);  ET_SW_ROI = (96, 96, 96);  ET_SW_OVERLAP = 0.50; ET_SW_BATCH = 4
NET_PATCH = (96, 96, 96); NET_SW_ROI = (96, 96, 96); NET_SW_OVERLAP = 0.25; NET_SW_BATCH = 4

UNET_CHANNELS = (32, 64, 128, 256, 320)
UNET_STRIDES = (2, 2, 2, 2)
NUM_RES_UNITS = 2
UNET_IN_CH = 4
UNET_OUT_CH = 2

TTA_ENABLED = True
TTA_AUG_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
TTA_VOTE_FRAC = 0.5

DEFAULT_MIN_VOXELS = {1: 100, 2: 200, 3: 500, 4: 130}

SCRATCH = Path(tempfile.gettempdir())


# ══ Model definitions ════════════════════════════════════════════════════════
class ChannelAttention3D(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        mid = max(channels // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.sigmoid(self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x)))
        return x * scale.view(scale.size(0), -1, 1, 1, 1)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        cat = torch.cat([x.mean(dim=1, keepdim=True),
                         x.max(dim=1, keepdim=True)[0]], dim=1)
        return x * self.sigmoid(self.conv(cat))


class CBAM3D(nn.Module):
    def __init__(self, channels, ratio=16, spatial_kernel=7):
        super().__init__()
        self.ch = ChannelAttention3D(channels, ratio)
        self.sp = SpatialAttention3D(spatial_kernel)

    def forward(self, x):
        return self.sp(self.ch(x))


class CBAMUNet3D(nn.Module):
    """ED specialist. Crop source during training: T2f."""

    def __init__(self, in_channels=4, out_channels=2,
                 channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2), num_res_units=2):
        super().__init__()
        self.backbone = UNet(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            channels=channels, strides=strides, num_res_units=num_res_units,
            norm=Norm.INSTANCE, dropout=0.0,
        )

    def forward(self, x):
        return self.backbone(x)


class SEBlock3D(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        mid = max(channels // ratio, 1)
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.excitation(self.squeeze(x))
        return x * scale.view(scale.size(0), -1, 1, 1, 1)


class SEUNet3D(nn.Module):
    """CC specialist. Crop source during training: T2w."""

    def __init__(self, in_channels=4, out_channels=2,
                 channels=(32, 64, 128, 256, 320), strides=(2, 2, 2, 2), num_res_units=2):
        super().__init__()
        self.backbone = UNet(
            spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
            channels=channels, strides=strides, num_res_units=num_res_units,
            norm=Norm.INSTANCE, dropout=0.0,
        )
        self.se_blocks = nn.ModuleList([SEBlock3D(c) for c in channels[:-1]])

    def forward(self, x):
        return self.backbone(x)


class SwinUNETRWrapper(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, img_size=None,
                 feature_size=48, depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
                 drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0,
                 use_checkpoint=False):
        super().__init__()
        self.net = _SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            depths=depths,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
        )

    def forward(self, x):
        return self.net(x)


def build_specialist(patch_size, out_channels=2):
    return SwinUNETRWrapper(
        in_channels=4, out_channels=out_channels, img_size=patch_size,
        feature_size=SWIN_FEATURE_SIZE, depths=SWIN_DEPTHS, num_heads=SWIN_NUM_HEADS,
        drop_rate=SWIN_DROP_RATE, attn_drop_rate=SWIN_ATTN_DROP,
        dropout_path_rate=SWIN_DROP_PATH, use_checkpoint=False,
    )


def _load_checkpoint(model, ckpt_path, label):
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = raw.get("state_dict", raw) if isinstance(raw, dict) else raw

    clean = {}
    for k, v in sd.items():
        nk = k.replace("_orig_mod.", "").replace("module.", "")
        nk = re.sub(r"\.sub(\d+)", r".submodule.\1", nk)
        nk = nk.replace(".subconv.", ".submodule.conv.")
        nk = nk.replace(".subresidual.", ".submodule.residual.")
        nk = nk.replace(".channel_att.", ".ch.")
        nk = nk.replace(".spatial_att.", ".sp.")
        clean[nk] = v

    missing, unexpected = model.load_state_dict(clean, strict=True)
    if missing:
        raise RuntimeError(f"{label}: missing keys: {missing[:5]}")
    if unexpected:
        raise RuntimeError(f"{label}: unexpected keys: {unexpected[:5]}")
    model.to(DEVICE).eval()
    log.info("loaded %s", label)
    return model


# ══ Data discovery ═══════════════════════════════════════════════════════════
def find_modality(case_dir, key):
    hits = sorted(case_dir.glob(f"*-{key}.nii.gz"))
    if not hits:
        hits = sorted([
            p for p in case_dir.glob(f"*{key}*.nii*")
            if "mask" not in p.name.lower() and "seg" not in p.name.lower()
        ])
    return str(hits[0]) if hits else None


def discover_cases(data_root: Path):
    entries, skipped = [], []
    for d in sorted(p for p in data_root.iterdir() if p.is_dir()):
        mods = {k: find_modality(d, k) for k in IMAGE_KEYS}
        miss = [k for k, v in mods.items() if v is None]
        if miss:
            skipped.append((d.name, miss))
            continue
        entries.append({**mods, "subject_id": d.name})
    for sid, miss in skipped:
        log.warning("skipped %s, missing modalities: %s", sid, miss)
    return entries


# ══ Skull stripping, in-process ══════════════════════════════════════════════
def deskull_case(entry, image_keys=IMAGE_KEYS, mask_modality="t1c",
                 border=SKULL_BORDER, no_csf=SKULL_NO_CSF):
    """
    Identical to the notebook's SynthStrip step, except the model is invoked
    directly rather than through `docker run`. The evaluation container has no
    Docker daemon and no network, so the subprocess call had to go.
    """
    sid = entry["subject_id"]
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"skull_{sid}_", dir=SCRATCH))
    new_entry = dict(entry)

    mask_input = Path(entry[mask_modality])
    # Uncompressed on purpose. These files exist for seconds before being read
    # back and deleted, so gzip is pure single-threaded CPU cost for no benefit.
    # Contents are bit-identical either way.
    tmp_mask = tmp_dir / "brain_mask.nii"
    tmp_brain = tmp_dir / "_brain_throwaway.nii"

    cmd = [
        sys.executable, str(SYNTHSTRIP_BIN),
        "-i", str(mask_input),
        "-o", str(tmp_brain),
        "-m", str(tmp_mask),
        "--border", str(border),
    ]
    if no_csf:
        cmd.append("--no-csf")

    env = dict(os.environ)
    env["FREESURFER_HOME"] = str(SYNTHSTRIP_HOME)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0 or not tmp_mask.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(
            f"[deskull_case] SynthStrip failed for {sid}\n"
            f"  CMD: {' '.join(cmd)}\n"
            f"  STDOUT: {result.stdout[-500:]}\n"
            f"  STDERR: {result.stderr[-500:]}"
        )

    mask_arr = nib.load(tmp_mask).get_fdata().astype(bool)

    for key in image_keys:
        img = nib.load(entry[key])
        stripped = img.get_fdata().astype(np.float32) * mask_arr
        out_path = tmp_dir / f"{key}_stripped.nii"
        nib.save(nib.Nifti1Image(stripped, img.affine, img.header), str(out_path))
        new_entry[key] = str(out_path)

    new_entry["_tmpdir"] = str(tmp_dir)
    return new_entry


def maybe_deskull(entry):
    if SKIP_DESKULL:
        return {**entry, "_tmpdir": None}
    return deskull_case(entry)


# ══ Transforms ═══════════════════════════════════════════════════════════════
def _make_infer_tf(crop_key):
    return Compose([
        LoadImaged(keys=IMAGE_KEYS),
        EnsureChannelFirstd(keys=IMAGE_KEYS),
        Orientationd(keys=IMAGE_KEYS, axcodes="RAS"),
        NormalizeIntensityd(keys=IMAGE_KEYS, nonzero=True, channel_wise=True),
        CropForegroundd(keys=IMAGE_KEYS, source_key=crop_key, allow_smaller=True),
    ])


def _make_invertd(tf, crop_key):
    return Invertd(
        keys="pred", transform=tf, orig_keys=crop_key,
        meta_keys="pred_meta_dict", orig_meta_keys=f"{crop_key}_meta_dict",
        meta_key_postfix="meta_dict", nearest_interp=True, to_tensor=True,
    )


# ══ Inference ════════════════════════════════════════════════════════════════
def _infer_one(tf, model, entry, roi, sw_batch, overlap):
    ds = Dataset(data=[entry], transform=tf)
    item = ds[0]
    inp = torch.cat([item[k].unsqueeze(0).to(DEVICE) for k in IMAGE_KEYS], dim=1)
    with torch.no_grad(), autocast(enabled=USE_AMP):
        logits = sliding_window_inference(
            inp, roi, sw_batch, model, overlap=overlap,
            mode="gaussian", progress=False,
        )
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()
    return probs, item


def _inject_and_invert(item, pred_argmax, invertd_fn, crop_key):
    p = pred_argmax
    if p.dim() == 3:
        p = p.unsqueeze(0)

    ref_mt = item[crop_key]
    pred_meta = MetaTensor(
        p.float().cpu(),
        meta=ref_mt.meta.copy() if hasattr(ref_mt, "meta") else {},
        applied_operations=list(ref_mt.applied_operations)
        if hasattr(ref_mt, "applied_operations") else [],
    )
    item["pred"] = pred_meta

    inv_batch = invertd_fn(item)
    pred_inv = inv_batch["pred"]
    pred_np = pred_inv.cpu().numpy() if isinstance(pred_inv, torch.Tensor) else np.asarray(pred_inv)
    pred_orig = np.squeeze(pred_np).astype(np.uint8)

    if hasattr(pred_inv, "affine"):
        affine = pred_inv.affine.cpu().numpy()
    else:
        meta_dict = inv_batch.get("pred_meta_dict", {})
        ref_path = meta_dict.get("filename_or_obj")
        affine = nib.load(ref_path).affine if ref_path else np.eye(4)

    meta_dict = inv_batch.get("pred_meta_dict", inv_batch.get(f"{crop_key}_meta_dict", {}))
    ref_path = meta_dict.get("filename_or_obj")
    orig_hdr = nib.load(ref_path).header.copy() if ref_path else nib.Nifti1Header()
    orig_hdr.set_data_dtype(np.uint8)

    return pred_orig, affine, orig_hdr


class Engine:
    """Holds the four specialists and their transform/inverse pairs."""

    def __init__(self):
        self.model_ed = _load_checkpoint(
            CBAMUNet3D(in_channels=UNET_IN_CH, out_channels=UNET_OUT_CH,
                       channels=UNET_CHANNELS, strides=UNET_STRIDES,
                       num_res_units=NUM_RES_UNITS),
            CKPT_ED, "CBAMUNet3D  [ED,  crop=t2f]")
        self.model_cc = _load_checkpoint(
            SEUNet3D(in_channels=UNET_IN_CH, out_channels=UNET_OUT_CH,
                     channels=UNET_CHANNELS, strides=UNET_STRIDES,
                     num_res_units=NUM_RES_UNITS),
            CKPT_CC, "SEUNet3D    [CC,  crop=t2w]")
        self.model_net = _load_checkpoint(
            build_specialist(NET_PATCH), CKPT_NET, "SwinUNETR   [NET, crop=t1c]")
        self.model_et = _load_checkpoint(
            build_specialist(ET_PATCH), CKPT_ET, "SwinUNETR   [ET,  crop=t1c]")

        self.tf_ed = _make_infer_tf("t2f")
        self.tf_cc = _make_infer_tf("t2w")
        self.tf_et = _make_infer_tf("t1c")
        self.tf_net = _make_infer_tf("t1c")

        self.invertd_ed = _make_invertd(self.tf_ed, "t2f")
        self.invertd_cc = _make_invertd(self.tf_cc, "t2w")
        self.invertd_et = _make_invertd(self.tf_et, "t1c")
        self.invertd_net = _make_invertd(self.tf_net, "t1c")

    @torch.no_grad()
    def ensemble_predict(self, entry):
        probs_ed, item_ed = _infer_one(self.tf_ed, self.model_ed, entry,
                                       ED_SW_ROI, ED_SW_BATCH, ED_SW_OVERLAP)
        probs_cc, item_cc = _infer_one(self.tf_cc, self.model_cc, entry,
                                       CC_SW_ROI, CC_SW_BATCH, CC_SW_OVERLAP)
        probs_et, item_et = _infer_one(self.tf_et, self.model_et, entry,
                                       ET_SW_ROI, ET_SW_BATCH, ET_SW_OVERLAP)
        probs_net, item_net = _infer_one(self.tf_net, self.model_net, entry,
                                         NET_SW_ROI, NET_SW_BATCH, NET_SW_OVERLAP)

        pred_ed_crop = probs_ed.argmax(dim=0).long()
        pred_cc_crop = probs_cc.argmax(dim=0).long()
        pred_et_crop = probs_et.argmax(dim=0).long()
        pred_net_crop = probs_net.argmax(dim=0).long()

        ed_orig, affine, orig_hdr = _inject_and_invert(
            copy.deepcopy(item_ed), pred_ed_crop, self.invertd_ed, "t2f")
        cc_orig, _, _ = _inject_and_invert(
            copy.deepcopy(item_cc), pred_cc_crop, self.invertd_cc, "t2w")
        et_orig, _, _ = _inject_and_invert(
            copy.deepcopy(item_et), pred_et_crop, self.invertd_et, "t1c")
        net_orig, _, _ = _inject_and_invert(
            copy.deepcopy(item_net), pred_net_crop, self.invertd_net, "t1c")

        # Priority merge, ET innermost and highest priority.
        final_label = np.zeros_like(ed_orig, dtype=np.uint8)
        final_label[ed_orig == 1] = 4
        final_label[cc_orig == 1] = 3
        final_label[net_orig == 1] = 2
        final_label[et_orig == 1] = 1
        return final_label, affine, orig_hdr


# ══ Post-processing ══════════════════════════════════════════════════════════
def remove_small_components(label_map, min_voxels_per_label=None):
    if min_voxels_per_label is None:
        min_voxels_per_label = DEFAULT_MIN_VOXELS
    cleaned = label_map.copy()
    struct = ndimage.generate_binary_structure(3, 2)
    for lbl in [1, 2, 3, 4]:
        mask = (label_map == lbl)
        if not mask.any():
            continue
        labeled_blobs, n_blobs = ndimage.label(mask, structure=struct)
        blob_sizes = ndimage.sum(mask, labeled_blobs, range(1, n_blobs + 1))
        threshold = min_voxels_per_label.get(lbl, 10)
        for idx, size in enumerate(blob_sizes, start=1):
            if size < threshold:
                cleaned[labeled_blobs == idx] = 0
    return cleaned


def remove_distant_components(label_map, max_distance_mm=15.0, voxel_size_mm=1.0):
    cleaned = label_map.copy()
    struct = ndimage.generate_binary_structure(3, 2)
    max_dist_vox = max_distance_mm / voxel_size_mm
    for lbl in [1, 2, 3, 4]:
        mask = (label_map == lbl)
        if not mask.any():
            continue
        labeled_blobs, n_blobs = ndimage.label(mask, structure=struct)
        if n_blobs <= 1:
            continue
        blob_sizes = ndimage.sum(mask, labeled_blobs, range(1, n_blobs + 1))
        largest_idx = int(np.argmax(blob_sizes)) + 1
        dist_from_main = ndimage.distance_transform_edt(~(labeled_blobs == largest_idx))
        for idx in range(1, n_blobs + 1):
            if idx == largest_idx:
                continue
            if dist_from_main[labeled_blobs == idx].min() > max_dist_vox:
                cleaned[labeled_blobs == idx] = 0
    return cleaned


def enforce_net_ed_dependency(label_map, ed_max_dist_from_net_mm=15.0, voxel_size_mm=1.0):
    cleaned = label_map.copy()
    net_mask = (cleaned == 2)
    ed_mask = (cleaned == 4)

    n_slices = cleaned.shape[2]
    for z in range(n_slices):
        if not net_mask[:, :, z].any():
            cleaned[:, :, z][cleaned[:, :, z] == 4] = 0

    ed_mask_after_r1 = (cleaned == 4)

    if not net_mask.any():
        cleaned[ed_mask_after_r1] = 0
        return cleaned

    max_dist_vox = ed_max_dist_from_net_mm / voxel_size_mm
    dist_to_net = ndimage.distance_transform_edt(~net_mask)

    struct = ndimage.generate_binary_structure(3, 2)
    labeled_ed, n_blobs = ndimage.label(ed_mask_after_r1, structure=struct)

    for idx in range(1, n_blobs + 1):
        blob_voxels = labeled_ed == idx
        if dist_to_net[blob_voxels].min() > max_dist_vox:
            cleaned[blob_voxels] = 0

    return cleaned


# ══ Test-time augmentation ═══════════════════════════════════════════════════
def _aug_volumes(volumes_dict, aug_id):
    if aug_id == 0:
        return {k: v.copy() for k, v in volumes_dict.items()}, lambda m: m.copy()

    elif aug_id in (1, 2, 3, 4):
        angle = {1: 5.0, 2: -5.0, 3: 10.0, 4: -10.0}[aug_id]
        aug = {}
        for k, v in volumes_dict.items():
            aug[k] = scipy_rotate(v.astype(np.float32), angle, axes=(0, 1),
                                  reshape=False, order=1, mode="constant", cval=0.0)

        def _invert_rot(mask, _angle=-angle):
            return scipy_rotate(mask.astype(np.float32), _angle, axes=(0, 1),
                                reshape=False, order=0, mode="constant",
                                cval=0.0).round().astype(np.uint8)
        return aug, _invert_rot

    elif aug_id == 5:
        aug = {k: np.flip(v, axis=0).copy() for k, v in volumes_dict.items()}
        return aug, lambda m: np.flip(m, axis=0).copy()

    elif aug_id == 6:
        aug = {k: np.flip(v, axis=1).copy() for k, v in volumes_dict.items()}
        return aug, lambda m: np.flip(m, axis=1).copy()

    elif aug_id == 7:
        aug = {k: np.flip(v, axis=2).copy() for k, v in volumes_dict.items()}
        return aug, lambda m: np.flip(m, axis=2).copy()

    elif aug_id in (8, 9):
        # NOTE: these are no-ops. NormalizeIntensityd z-scores each channel over
        # its nonzero voxels downstream, and z-scoring is invariant to constant
        # scaling, so the network sees the same input as aug 0. Kept to preserve
        # the notebook's vote weighting; see SUBMISSION_STEPS.md.
        scale = {8: 1.05, 9: 0.95}[aug_id]
        aug = {k: (v.astype(np.float32) * scale) for k, v in volumes_dict.items()}
        return aug, lambda m: m.copy()

    raise ValueError(f"Unknown aug_id: {aug_id}")


def _save_aug_niftis(aug_vols, entry, tmp_dir, aug_id):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    new_entry = {}
    for key in IMAGE_KEYS:
        orig_img = nib.load(entry[key])
        # Uncompressed: 40 of these per case, each read back within seconds and
        # then deleted. gzip here is single-threaded CPU for nothing. The array
        # LoadImaged reads is identical either way.
        tmp_path = tmp_dir / f"aug{aug_id}_{key}.nii"
        nib.save(
            nib.Nifti1Image(aug_vols[key].astype(np.float32),
                            orig_img.affine, orig_img.header),
            str(tmp_path),
        )
        new_entry[key] = str(tmp_path)
    new_entry["subject_id"] = entry["subject_id"]
    return new_entry


def _vote_masks(mask_list, vote_frac=0.5):
    n = len(mask_list)
    shape = mask_list[0].shape
    threshold = vote_frac * n

    counts = {lbl: np.zeros(shape, dtype=np.int16) for lbl in [1, 2, 3, 4]}
    for mask in mask_list:
        for lbl in [1, 2, 3, 4]:
            counts[lbl] += (mask == lbl).astype(np.int16)

    voted = np.zeros(shape, dtype=np.uint8)
    for lbl in [4, 3, 2, 1]:
        voted[counts[lbl] > threshold] = lbl
    return voted


@torch.no_grad()
def tta_ensemble_predict(engine, entry, aug_ids=None, vote_frac=None):
    aug_ids = TTA_AUG_IDS if aug_ids is None else aug_ids
    vote_frac = TTA_VOTE_FRAC if vote_frac is None else vote_frac

    sid = entry["subject_id"]
    # Scratch lives in /tmp. The notebook put this under the output root, which
    # would leave stray NIfTIs in /output for the scorer to trip over.
    tmp_dir = SCRATCH / "_tta_tmp" / sid

    raw_vols = {}
    for key in IMAGE_KEYS:
        raw_vols[key] = nib.load(entry[key]).get_fdata().astype(np.float32)

    collected_masks = []
    affine = None
    orig_hdr = None

    try:
        for aug_id in aug_ids:
            aug_vols, invert_fn = _aug_volumes(raw_vols, aug_id)
            aug_entry = _save_aug_niftis(aug_vols, entry, tmp_dir, aug_id)

            pred_mask, _aff, _hdr = engine.ensemble_predict(aug_entry)
            if affine is None:
                affine = _aff
            if orig_hdr is None:
                orig_hdr = _hdr

            inv_mask = invert_fn(pred_mask)
            collected_masks.append(inv_mask)
            log.info("    aug %d done, fg=%d", aug_id, int((inv_mask > 0).sum()))

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    voted_label = _vote_masks(collected_masks, vote_frac=vote_frac)
    voted_label = remove_small_components(voted_label)
    voted_label = remove_distant_components(voted_label, max_distance_mm=15.0)
    voted_label = enforce_net_ed_dependency(voted_label, ed_max_dist_from_net_mm=15.0)
    return voted_label, affine, orig_hdr


# ══ Entry point ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="BraTS-PED Task 02 inference")
    parser.add_argument("--input-dir", type=Path, default=Path("/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("/output"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("device=%s  amp=%s  tta=%s  deskull=%s",
             DEVICE, USE_AMP, TTA_ENABLED, not SKIP_DESKULL)

    if not SKIP_DESKULL and not SYNTHSTRIP_BIN.exists():
        log.error("SynthStrip absent at %s. Rebuild the image with the "
                  "synthstrip/ directory present, or set SKIP_DESKULL=True.",
                  SYNTHSTRIP_BIN)
        sys.exit(1)

    entries = discover_cases(args.input_dir)
    if not entries:
        log.error("no cases found under %s", args.input_dir)
        sys.exit(1)
    log.info("%d case(s) queued", len(entries))

    engine = Engine()

    t_total = time.time()
    failures = 0

    for idx, entry in enumerate(entries, 1):
        sid = entry["subject_id"]
        log.info("[%d/%d] %s", idx, len(entries), sid)
        t0 = time.time()
        stripped_entry = None
        try:
            stripped_entry = maybe_deskull(entry)
            if TTA_ENABLED:
                final_label, affine, orig_hdr = tta_ensemble_predict(engine, stripped_entry)
            else:
                final_label, affine, orig_hdr = engine.ensemble_predict(stripped_entry)
                final_label = remove_small_components(final_label)
                final_label = remove_distant_components(final_label, max_distance_mm=15.0)
                final_label = enforce_net_ed_dependency(final_label, ed_max_dist_from_net_mm=15.0)

            # Flat into /output, named exactly {subject_id}.nii.gz
            out_path = args.output_dir / f"{sid}.nii.gz"
            nib.save(nib.Nifti1Image(final_label.astype(np.uint8), affine, orig_hdr),
                     str(out_path))

            vox = {LABEL_NAMES[l]: int((final_label == l).sum()) for l in range(NUM_CLASSES)}
            log.info("[%d/%d] %s done in %.1fs  %s",
                     idx, len(entries), sid, time.time() - t0,
                     {k: v for k, v in vox.items() if k != "BG"})
            del final_label

        except Exception:
            failures += 1
            log.error("[%d/%d] %s FAILED\n%s", idx, len(entries), sid, traceback.format_exc())

        finally:
            if stripped_entry:
                tmpdir = stripped_entry.get("_tmpdir")
                if tmpdir:
                    shutil.rmtree(tmpdir, ignore_errors=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    log.info("%d/%d succeeded in %.1fs", len(entries) - failures, len(entries),
             time.time() - t_total)
    sys.exit(1 if failures == len(entries) else 0)


if __name__ == "__main__":
    main()
