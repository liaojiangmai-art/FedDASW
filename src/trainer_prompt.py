"""Prompt-only trainer: freeze CLIP backbone, only train prompt_ctx tokens.

Similar to CoOp (Context Optimization) but in FL setting.
Each client trains its own prompt tokens, CLIP weights are never updated.
"""
import torch
import logging


def freeze_clip_train_prompt(model, options):
    """Freeze all CLIP params, unfreeze only prompt_ctx."""
    umodel = model.module if options.distributed else model
    for param in umodel.parameters():
        param.requires_grad = False
    # Unfreeze prompt tokens
    if hasattr(umodel, 'prompt_ctx'):
        umodel.prompt_ctx.requires_grad = True
        logging.info("Prompt-only mode: prompt_ctx unfrozen, CLIP frozen")
    else:
        logging.warning("Model has no prompt_ctx!")


def get_prompt_parameters(model):
    """Return only prompt_ctx parameters."""
    umodel = model.module if hasattr(model, 'module') else model
    params = []
    for name, param in umodel.named_parameters():
        if 'prompt_ctx' in name:
            params.append(param)
    return params

