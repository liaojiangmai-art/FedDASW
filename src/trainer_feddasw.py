"""FedDASW-v5 trainer: decoupled weighting + momentum queue for FL negative mining.

Cross-Modal Momentum Negative Mining (fixed):
  - Warmup phase: queue fills with real features before being used as negatives
  - Only clean-sample features (FedDASW weight > median) enter the queue
  - Queue passed to loss only after warmup barrier (1 full cycle)
  - Feature-only storage → negligible memory overhead (~8MB for 4096x512)
"""
import time
import torch
import logging

from src.losses_feddasw import FedDASWLoss


def train_feddasw_epoch(epoch, model, processor, data, optimizer, scheduler, scaler, options,
                        text_queue=None, image_queue=None):
    """One epoch of FedDASW training with round-gradual queue enablement.

    Queue always accumulates gated features. Loss usage is gradual by round:
      Round 1-2: 0% (silent accumulation)
      Round 3:   25% (1024 features)
      Round 4-5: 50% (2048)
      Round 6-7: 75% (3072)
      Round 8+:  100% (4096)
    Queue is cleared each FL round after aggregation → no stale features.
    """
    dataloader = data["train"]
    model.train()
    umodel = model.module if options.distributed else model

    criterion = FedDASWLoss().to(options.device)
    modulo = max(1, int(dataloader.num_samples / options.batch_size / 10))
    start = time.time()

    queue_size = getattr(options, 'queue_size', 4096)
    dim = 512

    # Round-based gradual enablement
    current_round = getattr(options, 'current_fl_round', 1)
    queue_fraction = min(1.0, max(0.0, (current_round - 2) * 0.25))
    queue_use_count = int(queue_size * queue_fraction) if queue_size > 0 else 0

    if text_queue is None and queue_size > 0:
        text_queue = torch.zeros(queue_size, dim, device=options.device)
        text_queue_ptr = 0
    elif text_queue is not None:
        text_queue_ptr = getattr(options, '_tq_ptr', 0)
    else:
        text_queue_ptr = 0

    if image_queue is None and queue_size > 0:
        image_queue = torch.zeros(queue_size, dim, device=options.device)
        image_queue_ptr = 0
    elif image_queue is not None:
        image_queue_ptr = getattr(options, '_iq_ptr', 0)
    else:
        image_queue_ptr = 0

    num_batches = dataloader.num_batches

    for index, batch in enumerate(dataloader):
        step = dataloader.num_batches * (epoch - 1) + index
        raw_captions = batch["raw_caps"]
        pixel_values = batch["pixel_values"].to(options.device, non_blocking=True)

        scheduler(step)
        optimizer.zero_grad()

        image_embed = umodel.get_image_features(pixel_values)
        text_embed = umodel.get_text_features(raw_captions, processor, options.device)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        # Gradual enablement: pass first queue_use_count entries (truncated slice)
        cur_text_queue = (text_queue[:queue_use_count]
                          if text_queue is not None and queue_use_count > 0 else None)
        cur_image_queue = (image_queue[:queue_use_count]
                           if image_queue is not None and queue_use_count > 0 else None)

        loss = criterion(image_embed, text_embed, umodel.logit_scale, options,
                         cur_text_queue, cur_image_queue)

        loss.backward()
        optimizer.step()
        umodel.logit_scale.data = torch.clamp(umodel.logit_scale.data, 0, 4.6052)

        # ---- Enqueue clean features (always, even during warmup) ----
        if queue_size > 0:
            with torch.no_grad():
                # Compute FedDASW weights for clean sample selection
                scale = umodel.logit_scale.exp()
                logits_i2t = scale * image_embed @ text_embed.t()
                target = torch.arange(image_embed.shape[0], device=options.device).long()
                loss_i2t_per = torch.nn.functional.cross_entropy(
                    logits_i2t, target, reduction='none')
                loss_t2i_per = torch.nn.functional.cross_entropy(
                    logits_i2t.t(), target, reduction='none')
                anchor_i2t = loss_i2t_per.mean() + 1e-8
                anchor_t2i = loss_t2i_per.mean() + 1e-8
                w_i2t = torch.exp(-loss_i2t_per / anchor_i2t)
                w_t2i = torch.exp(-loss_t2i_per / anchor_t2i)

                # Top-50% cleanest → enqueue text
                text_clean_mask = w_i2t > w_i2t.median()
                clean_texts = text_embed[text_clean_mask]
                n_text = clean_texts.shape[0]
                for i in range(n_text):
                    text_queue[text_queue_ptr] = clean_texts[i]
                    text_queue_ptr = (text_queue_ptr + 1) % queue_size

                # Top-50% cleanest → enqueue image
                image_clean_mask = w_t2i > w_t2i.median()
                clean_images = image_embed[image_clean_mask]
                n_img = clean_images.shape[0]
                for i in range(n_img):
                    image_queue[image_queue_ptr] = clean_images[i]
                    image_queue_ptr = (image_queue_ptr + 1) % queue_size

        end = time.time()
        if options.master and (((index + 1) % modulo == 0) or (index == num_batches - 1)):
            num_samples = (index + 1) * len(raw_captions) * options.num_devices
            if queue_size > 0:
                q_tag = f", Q={queue_use_count}/{queue_size}(r{current_round})"
            else:
                q_tag = ""
            logging.info(
                f"Train Epoch: {epoch:02d} [{num_samples}/{dataloader.num_samples} "
                f"({100.0*(index+1)/num_batches:.0f}%)] "
                f"\tloss: {loss.item():.4f}{q_tag}\tTime taken {end - start:.3f}"
            )
            start = time.time()

    # Store queue state for next epoch
    if queue_size > 0:
        options._tq_ptr = text_queue_ptr
        options._iq_ptr = image_queue_ptr

    return text_queue, image_queue


@torch.no_grad()
def compute_client_confidence(model, processor, shared_loader, options):
    """Compute per-client decoupled confidence on shared set.

    Returns: (conf_i2t, conf_t2i)
    """
    model.eval()
    umodel = model.module if options.distributed else model
    total_i2t, total_t2i = 0.0, 0.0
    count = 0

    for batch in shared_loader:
        raw_captions = batch["raw_caps"]
        pixel_values = batch["pixel_values"].to(options.device, non_blocking=True)

        image_embed = umodel.get_image_features(pixel_values)
        text_embed = umodel.get_text_features(raw_captions, processor, options.device)
        image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        B = image_embed.shape[0]
        device = options.device
        logits_i2t = umodel.logit_scale.exp() * image_embed @ text_embed.t()
        target = torch.arange(B, device=device).long()

        loss_i2t = torch.nn.functional.cross_entropy(logits_i2t, target, reduction='none')
        loss_t2i = torch.nn.functional.cross_entropy(logits_i2t.t(), target, reduction='none')

        w_i2t = torch.exp(-loss_i2t / (loss_i2t.mean() + 1e-8))
        w_t2i = torch.exp(-loss_t2i / (loss_t2i.mean() + 1e-8))

        total_i2t += w_i2t.mean().item()
        total_t2i += w_t2i.mean().item()
        count += 1

    conf_i2t = total_i2t / max(count, 1)
    conf_t2i = total_t2i / max(count, 1)
    return conf_i2t, conf_t2i
