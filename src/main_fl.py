"""FL entry point: Federated Learning with Dirichlet Non-IID data split."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import logging
import warnings
import numpy as np
import random
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data_fl import split_data_fl
from src.trainer_fed import run_fl_training
from src.parser import parse_args
from src.logger import get_logger, set_logger

mp.set_start_method("spawn", force=True)
warnings.filterwarnings("ignore", category=FutureWarning)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(options):
    """Build pretrained CLIP model."""
    model_name_map = {
        "CLIP_VITB16": "ViT-B/16",
        "CLIP_VITB32": "ViT-B/32",
    }
    from pkgs.openai.clip import load as load_openai
    model, processor = load_openai(
        model_name_map.get(options.model_name, "ViT-B/16"), pretrained=True)
    return model, processor


def worker(rank, options, logger):
    options.the_rank = rank
    options.master = rank == 0

    set_logger(the_rank=rank, logger=logger, distributed=options.distributed)

    if options.device == "cuda":
        options.device += ":" + str(
            options.device_ids[options.the_rank] if options.distributed else options.device_id)

    if options.master:
        logging.info("Params:")
        with open(os.path.join(options.log_dir_path, "params.txt"), "w") as file:
            for key in sorted(vars(options)):
                value = getattr(options, key)
                logging.info(f"{key}: {value}")
                file.write(f"{key}: {value}\n")

    if options.distributed:
        dist.init_process_group(backend=options.distributed_backend,
                                init_method=options.distributed_init_method,
                                world_size=options.num_devices, rank=options.the_rank)

    options.batch_size = options.batch_size // options.num_devices

    # Build model
    model, processor = build_model(options)

    # If prompt-only mode, flag model to use prompt tokens in all text encoding
    if getattr(options, 'train_mode', 'full') == 'prompt':
        model.use_prompt = True
        logging.info("Model set to prompt mode (use_prompt=True)")

    if options.device.startswith("cuda"):
        torch.cuda.set_device(
            options.device_ids[options.the_rank] if options.distributed else options.device_id
        )
    model.to(options.device)

    # Split data for FL clients
    logging.info("Splitting data for FL clients...")
    client_data, client_sizes, shared_set = split_data_fl(options, processor)

    if options.distributed:
        model = DDP(model, device_ids=[options.device_ids[options.the_rank]])

    cudnn.benchmark = True
    cudnn.deterministic = False

    options.checkpoints_dir_path = os.path.join(options.log_dir_path, "checkpoints")
    os.makedirs(options.checkpoints_dir_path, exist_ok=True)

    # Run FL training
    run_fl_training(model, processor, client_data, client_sizes, shared_set, options, logger)

    if options.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    options = parse_args()
    seed_everything(options.seed)
    options.log_dir_path = os.path.join(options.logs, options.name)
    options.log_file_path = os.path.join(options.log_dir_path, "output.log")

    os.makedirs(options.log_dir_path, exist_ok=True)
    logger, listener = get_logger(options.log_file_path)

    listener.start()

    ngpus = torch.cuda.device_count()
    if ngpus == 0 or options.device == "cpu":
        options.device = "cpu"
        options.num_devices = 1
        options.distributed = False
        worker(0, options, logger)
    else:
        if ngpus == 1 or not options.distributed:
            options.device = "cuda"
            options.num_devices = 1
            options.distributed = False
            worker(0, options, logger)
        else:
            options.device = "cuda"
            if options.device_ids is None:
                options.device_ids = list(range(ngpus))
                options.num_devices = ngpus
            else:
                options.device_ids = list(map(int, options.device_ids))
                options.num_devices = len(options.device_ids)
            options.distributed = True
            os.environ["NCCL_P2P_DISABLE"] = "1"
            mp.spawn(worker, nprocs=options.num_devices, args=(options, logger))

    listener.stop()
