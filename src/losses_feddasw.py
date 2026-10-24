"""FedDASW-v5: Decoupled Asymmetric Soft-Weighting + Cross-Modal Momentum Queue.

Key innovations:
  1. FedDASW: per-sample decoupled I2T/T2I weighting
  2. CMA: Consensus-Maturity Annealing temperature
  3. Momentum Feature Queue: clean-sample text features cached for cross-batch
     negative mining, solving the FL small-batch negative scarcity problem.

Formula:
  w_i2t = exp(-L_i2t / L̄_i2t)    # I2T direction weight
  w_t2i = exp(-L_t2i / L̄_t2i)    # T2I direction weight
  With queue: logits extended from (B,B) to (B, B+Q)
  L_final = (1 - lambda_q) * L_batch + lambda_q * L_queue
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class FedDASWLoss(nn.Module):
    """Decoupled Asymmetric Soft-Weighting contrastive loss with momentum queue."""

    def __init__(self):
        super().__init__()

    def _directional_weighted_loss(
            self, logits_i2t, logits_t2i, target_i2t, target_t2i,
            temperature):
        loss_i2t_per = F.cross_entropy(logits_i2t, target_i2t, reduction='none')
        loss_t2i_per = F.cross_entropy(logits_t2i, target_t2i, reduction='none')

        with torch.no_grad():
            anchor_i2t = loss_i2t_per.mean() + 1e-8
            anchor_t2i = loss_t2i_per.mean() + 1e-8
            rho_i2t = loss_i2t_per / anchor_i2t
            rho_t2i = loss_t2i_per / anchor_t2i
            weight_i2t = torch.exp(-rho_i2t / temperature)
            weight_t2i = torch.exp(-rho_t2i / temperature)

            final_i2t = weight_i2t
            final_t2i = weight_t2i

        loss = (loss_i2t_per * final_i2t).mean() + (loss_t2i_per * final_t2i).mean()
        return loss / 2

    def forward(self, image_embeds, text_embeds, logit_scale, options=None,
                text_queue=None, image_queue=None):
        # CMA: Consensus-Maturity Annealing temperature
        temperature = getattr(options, 'dasw_temperature', 1.0)

        if options is not None and options.distributed:
            gathered_img = [torch.zeros_like(image_embeds) for _ in range(options.num_devices)]
            gathered_txt = [torch.zeros_like(text_embeds) for _ in range(options.num_devices)]
            dist.all_gather(gathered_img, image_embeds)
            dist.all_gather(gathered_txt, text_embeds)
            image_embeds = torch.cat(gathered_img[:options.the_rank] +
                                     [image_embeds] + gathered_img[options.the_rank + 1:])
            text_embeds = torch.cat(gathered_txt[:options.the_rank] +
                                    [text_embeds] + gathered_txt[options.the_rank + 1:])

        B = image_embeds.shape[0]
        device = image_embeds.device
        scale = logit_scale.exp()
        queue_loss_weight = float(getattr(options, 'queue_loss_weight', 0.001) if options is not None else 0.001)
        queue_loss_weight = min(max(queue_loss_weight, 0.0), 1.0)

        logits_i2t_base = scale * image_embeds @ text_embeds.t()
        logits_t2i_base = scale * text_embeds @ image_embeds.t()
        target_i2t = torch.arange(B, device=device).long()
        target_t2i = torch.arange(B, device=device).long()

        batch_loss = self._directional_weighted_loss(
            logits_i2t_base, logits_t2i_base, target_i2t, target_t2i,
            temperature
        )

        use_text_queue = text_queue is not None and text_queue.shape[0] > 0
        use_image_queue = image_queue is not None and image_queue.shape[0] > 0
        if queue_loss_weight <= 0.0 or (not use_text_queue and not use_image_queue):
            return batch_loss

        logits_i2t_queue = logits_i2t_base
        if use_text_queue:
            queued_text_logits = scale * image_embeds @ text_queue.t()
            logits_i2t_queue = torch.cat([logits_i2t_queue, queued_text_logits], dim=1)

        logits_t2i_queue = logits_t2i_base
        if use_image_queue:
            queued_image_logits = scale * text_embeds @ image_queue.t()
            logits_t2i_queue = torch.cat([logits_t2i_queue, queued_image_logits], dim=1)

        queue_loss = self._directional_weighted_loss(
            logits_i2t_queue, logits_t2i_queue, target_i2t, target_t2i,
            temperature
        )
        return (1.0 - queue_loss_weight) * batch_loss + queue_loss_weight * queue_loss
