#!/usr/bin/env python3
"""Extract native-resolution frozen Tiny ImageNet features; never trains a trunk.

ImageNet pretraining can overlap Tiny ImageNet. These features support only a
controlled transfer classifier study, not a clean representation benchmark.
"""
import argparse
import hashlib
import io
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class ArchiveImages(Dataset):
    def __init__(self, archive):
        self.archive = str(archive)
        self.z = None
        prefix = 'tiny-imagenet-200/'
        with zipfile.ZipFile(archive) as z:
            self.classes = sorted(z.read(prefix + 'wnids.txt').decode().split())
            class_idx = {c: i for i, c in enumerate(self.classes)}
            train = sorted(n for n in z.namelist() if n.startswith(prefix+'train/') and n.endswith('.JPEG'))
            annotations = dict(line.split('\t')[:2] for line in z.read(prefix+'val/val_annotations.txt').decode().splitlines())
            val = sorted(n for n in z.namelist() if n.startswith(prefix+'val/images/') and n.endswith('.JPEG'))
        self.names = train + val
        self.ntrain = len(train)
        self.labels = np.array([class_idx[n.split('/')[2]] for n in train] + [class_idx[annotations[Path(n).name]] for n in val])
        assert self.ntrain == 100000 and len(val) == 10000
        assert np.all(np.bincount(self.labels[:self.ntrain]) == 500)
        assert np.all(np.bincount(self.labels[self.ntrain:]) == 50)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        if self.z is None:
            self.z = zipfile.ZipFile(self.archive)
        with Image.open(io.BytesIO(self.z.read(self.names[i]))) as im:
            arr = np.array(im.convert('RGB'))
        assert arr.shape == (64, 64, 3)
        digest = hashlib.sha256(arr.tobytes()).hexdigest()
        x = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.)
        return x, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    t = time.time()
    ds = ArchiveImages(a.archive)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(a.device)
    means = torch.tensor([.485, .456, .406], device=a.device).reshape(1,3,1,1)
    stds = torch.tensor([.229, .224, .225], device=a.device).reshape(1,3,1,1)
    loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers, shuffle=False)
    feature = np.empty((len(ds),512), dtype='float32')
    digests = []
    offset = 0
    with torch.inference_mode():
        for i, (x, digest) in enumerate(loader):
            y = model((x.to(a.device) - means) / stds).cpu().numpy()
            feature[offset:offset+len(x)] = y
            offset += len(x)
            digests.extend(digest)
            if i % 40 == 0:
                print(f'features {offset}/{len(ds)} elapsed={time.time()-t:.1f}s', flush=True)
    assert np.isfinite(feature).all()
    np.savez(a.out, train_features=feature[:ds.ntrain], train_labels=ds.labels[:ds.ntrain],
             eval_features=feature[ds.ntrain:], eval_labels=ds.labels[ds.ntrain:])
    tr_hash, ev_hash = digests[:ds.ntrain], digests[ds.ntrain:]
    overlap = sorted(set(tr_hash) & set(ev_hash))
    repeated_train = len(tr_hash) - len(set(tr_hash))
    repeated_eval = len(ev_hash) - len(set(ev_hash))
    manifest = {
        'source_url': 'https://cs231n.stanford.edu/tiny-imagenet-200.zip',
        'archive_sha256': sha256(a.archive), 'archive_bytes': a.archive.stat().st_size,
        'features_sha256': sha256(a.out), 'encoder': 'torchvision.resnet18',
        'weights': 'ResNet18_Weights.IMAGENET1K_V1',
        'preprocessing': 'RGB native 64x64, divide by255, ImageNet mean/std; no crop or resize',
        'encoder_training': False, 'pretraining_overlap_risk': True,
        'classes': ds.classes, 'train_files': ds.names[:ds.ntrain], 'eval_files': ds.names[ds.ntrain:],
        'train_pixel_sha256': tr_hash, 'eval_pixel_sha256': ev_hash,
        'pixel_duplicates': {'train_extra_rows': repeated_train, 'eval_extra_rows': repeated_eval,
                             'cross_split_unique_hashes': len(overlap), 'cross_split_hashes': overlap},
        'elapsed_seconds': time.time()-t, 'device': a.device, 'platform': platform.platform(),
        'torch': torch.__version__, 'torchvision': torchvision.__version__, 'numpy': np.__version__,
        'script_sha256': sha256(__file__),
    }
    a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if not k.endswith('files') and not k.endswith('sha256') and k not in ['classes']}, indent=2))


if __name__ == '__main__':
    main()
