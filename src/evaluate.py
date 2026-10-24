"""Image-text retrieval evaluation using five captions per image."""
import logging
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class RetrievalEvalDataset(Dataset):
    """Lightweight dataset for retrieval evaluation: loads and preprocesses images."""
    def __init__(self, images, captions, root, processor):
        self.images = images
        self.captions = captions
        self.root = root
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        pixel_values = self.processor.process_image(Image.open(img_path).convert("RGB"))
        return pixel_values, self.captions[idx]


def _encode_retrieval_embeddings(umodel, processor, pixel_values, batch_caps, options):
    text_embed = umodel.get_text_features(
        raw_caps=list(batch_caps), processor=processor, device=options.device
    )
    text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
    image_embed = umodel.get_image_features(pixel_values)
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
    return image_embed, text_embed


def get_retrieval_metrics(model, processor, test_csv_path, options):
    """Evaluate image-text retrieval: Recall@K for I2T and T2I."""
    logging.info(f"Started retrieval evaluation on {test_csv_path}")

    model.eval()
    umodel = model.module if options.distributed else model

    root = os.path.dirname(test_csv_path)
    df = pd.read_csv(test_csv_path, sep=options.delimiter)
    images = df[options.image_key].tolist()
    captions = df[options.caption_key].tolist()

    # Build DataLoader for parallel image loading
    eval_dataset = RetrievalEvalDataset(images, captions, root, processor)
    loader_kwargs = dict(
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        pin_memory=True,
    )
    if options.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    eval_loader = DataLoader(eval_dataset, **loader_kwargs)

    text_embeddings = []
    image_embeddings = []

    with torch.no_grad():
        for pixel_values, batch_caps in tqdm(eval_loader, ascii=True):
            pixel_values = pixel_values.to(options.device, non_blocking=True)
            image_embed, text_embed = _encode_retrieval_embeddings(
                umodel, processor, pixel_values, batch_caps, options
            )
            image_embeddings.append(image_embed)
            text_embeddings.append(text_embed)

    text_embeddings = torch.cat(text_embeddings)  # [5N, d]
    image_embeddings = torch.cat(image_embeddings)  # [5N, d]

    n_images = len(image_embeddings) // 5
    n_texts = len(text_embeddings) // 5

    # I2T: Image -> Text (5 captions per image = 5 groups)
    ranks_i2t = np.zeros(n_images)
    for i in range(n_images):
        scores = image_embeddings[i * 5] @ text_embeddings.t()
        order = torch.argsort(scores, descending=True).cpu().numpy()
        rank = 1e20
        for j in range(i * 5, i * 5 + 5):
            pos = np.where(order == j)[0][0]
            if pos < rank:
                rank = pos
        ranks_i2t[i] = rank

    # T2I: Text -> Image (evaluate each caption separately, match to its image)
    ranks_t2i = np.zeros(n_texts * 5)
    for i in range(n_texts * 5):
        scores = text_embeddings[i] @ image_embeddings.t()
        # Compare against each image (using first caption embedding per image)
        img_scores = scores[::5].cpu().numpy()
        order = np.argsort(img_scores)[::-1]
        ranks_t2i[i] = np.where(order == i // 5)[0][0]

    i2t_r1 = 100.0 * np.mean(ranks_i2t < 1)
    i2t_r5 = 100.0 * np.mean(ranks_i2t < 5)
    i2t_r10 = 100.0 * np.mean(ranks_i2t < 10)

    t2i_r1 = 100.0 * np.mean(ranks_t2i < 1)
    t2i_r5 = 100.0 * np.mean(ranks_t2i < 5)
    t2i_r10 = 100.0 * np.mean(ranks_t2i < 10)

    r_mean = (i2t_r1 + i2t_r5 + i2t_r10 + t2i_r1 + t2i_r5 + t2i_r10) / 6

    results = {
        "retrieval_i2t_r1": i2t_r1, "retrieval_i2t_r5": i2t_r5, "retrieval_i2t_r10": i2t_r10,
        "retrieval_t2i_r1": t2i_r1, "retrieval_t2i_r5": t2i_r5, "retrieval_t2i_r10": t2i_r10,
        "retrieval_r_mean": r_mean,
    }

    logging.info("Finished retrieval evaluation")
    return results


def evaluate(epoch, model, processor, data, options):
    metrics = {}
    if options.master:
        logging.info("Base evaluation" if epoch == 0 else f"Epoch {epoch} evaluation")
        if options.retrieval_test_data is not None:
            metrics.update(get_retrieval_metrics(
                model, processor, options.retrieval_test_data, options,
            ))
        if metrics:
            logging.info("Results")
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.4f}")
    return metrics
