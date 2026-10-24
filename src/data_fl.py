"""FL data partitioning: Dirichlet-based Non-IID split with per-client noise injection.

Usage:
    from src.data_fl import split_data_fl
    client_loaders, client_sizes = split_data_fl(options, processor)

Each client gets a DataInfo with its own dataloader and noise setting.
The noise indices are saved per client for reproducibility.
"""
import os
import csv
import hashlib
import torch
import logging
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _stable_hash_to_int(value):
    """Stable string hash independent of Python's randomized hash()."""
    if not isinstance(value, str):
        value = str(value)
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _derive_seed(base_seed, *parts):
    """Derive deterministic per-purpose seeds from a base seed."""
    token = "::".join([str(base_seed)] + [str(part) for part in parts])
    return _stable_hash_to_int(token) % (2 ** 32)


def _build_noise_indices(length, noise_ratio, seed):
    """Create a deterministic text-to-image permutation for one client."""
    indices = np.arange(length)
    if length == 0 or noise_ratio <= 0:
        return indices

    rng = np.random.default_rng(seed)
    shuffled_positions = rng.permutation(length)
    noise_length = int(noise_ratio * length)
    if noise_length <= 0:
        return indices

    selected = shuffled_positions[:noise_length]
    reassigned = indices[selected].copy()
    rng.shuffle(reassigned)
    indices[selected] = reassigned
    return indices


class DataInfo:
    """Wrap a client or reference-set loader and its sample counts."""
    def __init__(self, dataloader, sampler=None):
        self.dataloader = dataloader
        self.sampler = sampler

    @property
    def dataset(self):
        return self.dataloader.dataset

    @property
    def num_batches(self):
        return self.dataloader.num_batches

    @property
    def num_samples(self):
        return self.dataloader.num_samples

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def set_epoch(self, epoch):
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


class ImageRawCaptionDataset(Dataset):
    """Image-caption pairs with an explicit text-to-image noise permutation."""

    def __init__(self, images, captions, root, processor, noise_ratio=0.0,
                 noise_indices=None, data_split="train"):
        self.root = root
        self.images = list(images)
        self.captions = list(captions)
        self.processor = processor
        self.length = len(self.images)

        # Noise injection
        self.noise_ratio = noise_ratio
        if data_split == "train" and noise_ratio > 0:
            if noise_indices is not None and len(noise_indices) == self.length:
                self.t2i_index = noise_indices
            else:
                self.t2i_index = np.arange(self.length)
                idx = np.arange(self.length)
                np.random.shuffle(idx)
                noise_length = int(noise_ratio * self.length)
                shuffle_index = self.t2i_index[idx[:noise_length]]
                np.random.shuffle(shuffle_index)
                self.t2i_index[idx[:noise_length]] = shuffle_index
        else:
            self.t2i_index = np.arange(self.length)

        # Save clean/noisy labels (1=clean, 0=noisy)
        self._labels = np.ones(self.length, dtype=int)
        original_idx = np.arange(self.length)
        self._labels[self.t2i_index != original_idx] = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {}
        item["sample_idxs"] = idx
        item["raw_caps"] = self.captions[idx]
        img_path = os.path.join(self.root, self.images[self.t2i_index[idx]])
        item["pixel_values"] = self.processor.process_image(
            Image.open(img_path), train_mode=True
        )
        return item


def split_data_fl(options, processor):
    """Split train data into K clients with Dirichlet Non-IID distribution.

    Returns:
        client_data: list of DataInfo, one per client (with noise applied)
        client_sizes: list of int, number of samples per client
    """
    csv_path = options.train_data
    root = os.path.dirname(csv_path)
    df = pd.read_csv(csv_path, sep=options.delimiter)

    images = df[options.image_key].tolist()
    captions = df[options.caption_key].tolist()

    # Group by unique image to keep 5 captions per image together
    unique_images = {}
    for i, img_path in enumerate(images):
        if img_path not in unique_images:
            unique_images[img_path] = []
        unique_images[img_path].append((img_path, captions[i]))

    unique_img_paths = list(unique_images.keys())
    num_unique = len(unique_img_paths)
    num_clients = options.num_clients
    base_seed = int(getattr(options, "seed", 42))
    partition_rng = np.random.default_rng(_derive_seed(base_seed, "partition"))
    shared_rng = np.random.default_rng(_derive_seed(base_seed, "shared_set"))

    # Create pseudo-labels for Dirichlet partitioning
    # Use a stable hash of image path mod num_pseudo_classes.
    num_pseudo_classes = 50
    pseudo_labels = np.array([
        _stable_hash_to_int(p) % num_pseudo_classes for p in unique_img_paths
    ])

    # Dirichlet partition: for each pseudo-class, distribute across clients
    alpha = options.dirichlet_alpha
    client_img_indices = [[] for _ in range(num_clients)]

    for c in range(num_pseudo_classes):
        class_mask = np.where(pseudo_labels == c)[0]
        if len(class_mask) == 0:
            continue
        class_mask = partition_rng.permutation(class_mask)
        # Dirichlet proportions for this class
        proportions = partition_rng.dirichlet([alpha] * num_clients)
        proportions = proportions * len(class_mask)
        proportions = proportions.astype(int)
        # Adjust rounding
        proportions[0] += len(class_mask) - proportions.sum()

        start = 0
        for k in range(num_clients):
            count = max(0, proportions[k])
            end = min(start + count, len(class_mask))
            client_img_indices[k].extend(class_mask[start:end].tolist())
            start = end

    # Extract a public clean reference set, excluded from local training.
    # Sample ~M samples for cross-client consensus: exclude from client training
    shared_size = getattr(options, 'shared_set_size', 1000)
    shared_indices = set()
    shared_images = {}
    if shared_size > 0:
        all_indices = shared_rng.permutation(num_unique)
        shared_count = min(shared_size, num_unique // 10)  # max 10% of data
        for idx in all_indices[:shared_count]:
            img_path = unique_img_paths[idx]
            shared_images[img_path] = unique_images[img_path]  # copy all captions
            shared_indices.add(idx)
        logging.info(f"Shared set: {len(shared_images)} unique images "
                     f"({sum(len(v) for v in shared_images.values())} pairs)")

    # Parse per-client noise ratios
    noise_ratios = options.client_noise_ratios
    if isinstance(noise_ratios, str):
        noise_ratios = [float(x) for x in noise_ratios.split(",")]
    # Extend/repeat to match num_clients
    if len(noise_ratios) < num_clients:
        noise_ratios = noise_ratios * (num_clients // len(noise_ratios) + 1)
    noise_ratios = noise_ratios[:num_clients]

    # Build per-client datasets
    client_data = []
    client_sizes = []
    batch_size = options.batch_size

    noise_dir = os.path.join(root, "fl_noise")
    os.makedirs(noise_dir, exist_ok=True)

    shared_set = None
    if shared_size > 0 and len(shared_images) > 0:
        # Build shared DataLoader (no noise, no shuffle)
        shared_imgs, shared_caps = [], []
        for img_path, captions in shared_images.items():
            for path, cap in captions:
                shared_imgs.append(path)
                shared_caps.append(cap)
        shared_dataset = ImageRawCaptionDataset(
            shared_imgs, shared_caps, root, processor,
            noise_ratio=0.0, data_split="test"
        )
        shared_loader = DataLoader(
            shared_dataset, batch_size=batch_size, shuffle=False,
            num_workers=options.num_workers, pin_memory=True
        )
        shared_loader.num_batches = len(shared_loader)
        shared_set = DataInfo(shared_loader)

    for k in range(num_clients):
        # Expand image indices back to caption-level, excluding shared set
        client_images = []
        client_captions = []
        for idx in client_img_indices[k]:
            if idx in shared_indices:
                continue
            img_path = unique_img_paths[idx]
            for path, cap in unique_images[img_path]:
                client_images.append(path)
                client_captions.append(cap)

        if len(client_images) == 0:
            logging.warning(f"Client {k} has 0 samples, skipping")
            client_data.append(None)
            client_sizes.append(0)
            continue

        nr = noise_ratios[k]

        # Load or create noise indices
        noise_file = os.path.join(
            noise_dir, f"client_{k}_noise_{nr:.2f}_seed{base_seed}.npy"
        )
        noise_indices = None
        if nr > 0 and os.path.exists(noise_file):
            noise_indices = np.load(noise_file)
            if len(noise_indices) != len(client_images):
                logging.warning(
                    f"Noise file length mismatch for client {k} "
                    f"(expected {len(client_images)}, got {len(noise_indices)}); regenerating"
                )
                noise_indices = None
        if nr > 0 and noise_indices is None:
            noise_seed = _derive_seed(base_seed, "noise", k, f"{nr:.4f}", len(client_images))
            noise_indices = _build_noise_indices(len(client_images), nr, noise_seed)

        # Create Dataset with noise
        dataset = ImageRawCaptionDataset(
            client_images, client_captions, root, processor,
            noise_ratio=nr, noise_indices=noise_indices, data_split="train"
        )

        # Save noise indices for reproducibility
        if nr > 0 and not os.path.exists(noise_file):
            np.save(noise_file, dataset.t2i_index)

        dl_kwargs = dict(
            batch_size=batch_size, shuffle=True,
            num_workers=options.num_workers, pin_memory=True,
            drop_last=True
        )
        loader_seed = _derive_seed(base_seed, "loader", k)
        dl_kwargs["generator"] = torch.Generator().manual_seed(loader_seed)
        if options.num_workers > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = 4
        dataloader = DataLoader(dataset, **dl_kwargs)
        dataloader.num_samples = len(dataloader) * batch_size
        dataloader.num_batches = len(dataloader)

        client_data.append(DataInfo(dataloader))
        client_sizes.append(len(client_images))
        logging.info(f"Client {k}: {len(client_images)} samples, noise={nr:.2f}")

    return client_data, client_sizes, shared_set
