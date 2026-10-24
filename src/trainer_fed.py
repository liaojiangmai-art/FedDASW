"""Federated prompt training with FedDASW, CMA and feature queues."""
import copy
import logging
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler

from src.evaluate import evaluate
from src.scheduler import cosine_scheduler
from src.trainer_feddasw import train_feddasw_epoch, compute_client_confidence
from src.trainer_prompt import freeze_clip_train_prompt, get_prompt_parameters


def run_fl_training(model, processor, client_data, client_sizes, shared_set, options, logger):
    """Train local prompts and aggregate them using clean-reference confidence."""
    if shared_set is None:
        raise ValueError("FedDASW requires a nonempty public clean reference set.")

    device = options.device
    num_clients = options.num_clients
    fl_rounds = options.fl_rounds
    local_epochs = options.local_epochs

    logging.info(f"FL Training: method=feddasw_v2, {num_clients} clients, "
                 f"{fl_rounds} rounds, {local_epochs} local epochs/round")
    logging.info(f"Client sizes: {client_sizes}")
    logging.info(f"Total samples: {sum(client_sizes)}")

    eval_data = {"eval_test": None, "eval_train": None}
    evaluate(0, model, processor, eval_data, options)
    best_r_mean = 0.0

    ema_conf = [0.0] * num_clients
    ema_asym = [0.0] * num_clients
    ema_conf_prev = [0.0] * num_clients
    ema_conf_initial = [0.0] * num_clients
    dasw_temperature = 1.0

    for round_idx in range(1, fl_rounds + 1):
        if options.master:
            logging.info(f"======== FL Round {round_idx}/{fl_rounds} ========")

        round_start = time.time()
        client_models = []
        for k in range(num_clients):
            if client_data[k] is None or client_sizes[k] == 0:
                client_models.append(None)
                continue

            local_model = copy.deepcopy(model).to(device)
            freeze_clip_train_prompt(local_model, options)
            optimizer = optim.AdamW(
                get_prompt_parameters(local_model), lr=options.lr,
                betas=(options.beta1, options.beta2), eps=options.eps,
            )
            scheduler = cosine_scheduler(
                optimizer, options.lr, 0,
                client_data[k].num_batches * local_epochs,
            )
            scaler = GradScaler()
            local_options = copy.copy(options)
            local_options.epochs = local_epochs
            local_options.master = options.master
            local_options.dasw_temperature = dasw_temperature
            local_options.current_fl_round = round_idx
            local_options.current_client_id = k

            # Queues persist across local epochs and reset for each client/round.
            text_queue, image_queue = None, None
            for local_epoch in range(1, local_epochs + 1):
                text_queue, image_queue = train_feddasw_epoch(
                    local_epoch, local_model, processor,
                    {"train": client_data[k]},
                    optimizer, scheduler, scaler, local_options,
                    text_queue, image_queue,
                )
            client_models.append(local_model)

        active_models = [m for m in client_models if m is not None]
        if active_models:
            client_confidences = []
            client_asymmetries = []
            for k in range(num_clients):
                if client_models[k] is None or client_sizes[k] == 0:
                    client_confidences.append(0.0)
                    client_asymmetries.append(0.0)
                    continue
                conf_i2t, conf_t2i = compute_client_confidence(
                    client_models[k], processor,
                    shared_set.dataloader, options)
                conf = (conf_i2t + conf_t2i) / 2
                asym = abs(conf_i2t - conf_t2i)  # asymmetry = noise signal
                client_confidences.append(conf)
                client_asymmetries.append(asym)
                logging.info(f"  Client {k}: conf={conf:.4f} (i2t={conf_i2t:.4f}, t2i={conf_t2i:.4f}), asym={asym:.4f}")

            # Innovation 1: EMA confidence smoothing across rounds
            for k in range(num_clients):
                ema_conf[k] = 0.7 * ema_conf[k] + 0.3 * client_confidences[k]
                ema_asym[k] = 0.7 * ema_asym[k] + 0.3 * client_asymmetries[k]

            # Store initial EMA (round 1) for maturity baseline
            if round_idx == 1:
                ema_conf_initial = ema_conf.copy()

            # === CMA: Consensus-Maturity Annealing ===
            # maturity: how much the system has improved vs initial round
            # stability: how smoothly the EMA is converging
            eps = 1e-8
            maturity = np.mean([ema_conf[k] / max(ema_conf_initial[k], eps) for k in range(num_clients)])
            stability = np.mean([
                1.0 - abs(ema_conf[k] - ema_conf_prev[k]) / max(ema_conf[k], eps)
                for k in range(num_clients)
            ]) if round_idx > 1 else 0.0
            beta = float(getattr(options, 'dasw_cma_beta', 2.5))
            dasw_temperature = 1.0 / (1.0 + beta * max(maturity - 1.0, 0) * max(stability, 0))

            ema_conf_prev = ema_conf.copy()  # save for next round's stability

            logging.info(f"  EMA confidence: {[f'{c:.4f}' for c in ema_conf]}")
            logging.info(f"  CMA: maturity={maturity:.4f}, stability={stability:.4f}, T={dasw_temperature:.4f}")

            # Select top-2 by EMA confidence as verifiers
            sorted_idx = np.argsort(ema_conf)[::-1]
            verifier_idx = sorted_idx[:2].tolist()
            logging.info(f"  Verifiers: {verifier_idx} (EMA conf: {[f'{ema_conf[i]:.4f}' for i in verifier_idx]})")

            total_size = sum(client_sizes)
            global_state = model.state_dict()
            for key in global_state:
                global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)

            total_weight = 0.0
            for k in range(num_clients):
                if client_models[k] is None or client_sizes[k] == 0:
                    continue
                verif_score = 1.0
                if k not in verifier_idx:
                    verif_score = 1.0 - abs(ema_conf[k] - np.mean([ema_conf[i] for i in verifier_idx]))
                    verif_score = max(0.3, min(1.0, verif_score))
                asym_lambda = float(getattr(options, 'dasw_asym_lambda', 0.3))
                asym_penalty = 1.0 - asym_lambda * ema_asym[k]

                weight = (client_sizes[k] / total_size) * ema_conf[k] * verif_score * asym_penalty
                client_state = client_models[k].state_dict()
                for key in global_state:
                    global_state[key] += client_state[key].float() * weight
                total_weight += weight
                tag = "VERIFIER" if k in verifier_idx else ""
                logging.info(f"  Client {k} {tag}: weight={weight:.4f} "
                             f"(verif={verif_score:.3f}, asym_p={asym_penalty:.3f})")

            for key in global_state:
                global_state[key] /= max(total_weight, 1e-8)
            model.load_state_dict(global_state)
            logging.info(f"  Aggregated {len(active_models)} clients (FedDASW-v2 EMA+Asym)")

        round_time = time.time() - round_start
        if options.master:
            logging.info(f"  Round {round_idx} time: {round_time:.1f}s")

        if round_idx % options.fl_eval_freq == 0:
            metrics = evaluate(round_idx, model, processor, eval_data, options)
            if metrics and metrics.get("retrieval_r_mean", 0) > best_r_mean:
                best_r_mean = metrics["retrieval_r_mean"]
                checkpoint = {
                    "round": round_idx, "name": options.name,
                    "state_dict": model.state_dict(), "r_mean": best_r_mean,
                }
                torch.save(
                    checkpoint,
                    os.path.join(options.checkpoints_dir_path, "fl_best.pt"),
                )

    logging.info(f"FL Training complete. Best r_mean: {best_r_mean:.4f}")
    return model
