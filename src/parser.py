"""Command-line options for the formal FedDASW training pipeline."""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="FedDASW federated prompt training")

    parser.add_argument("--name", type=str, default="default", help="Experiment name")
    parser.add_argument("--logs", type=str, default="./logs", help="Logs directory path")
    parser.add_argument("--model_name", type=str, default="CLIP_VITB16",
                        choices=["CLIP_VITB16", "CLIP_VITB32"],
                        help="Model architecture")
    parser.add_argument("--train_data", type=str, default=None, help="Path to train data csv file")
    parser.add_argument("--delimiter", type=str, default=",", help="CSV delimiter")
    parser.add_argument("--image_key", type=str, default="image", help="CSV column for image paths")
    parser.add_argument("--caption_key", type=str, default="caption", help="CSV column for captions")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta2")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam eps")
    parser.add_argument("--num_workers", type=int, default=8, help="Data loader workers")
    parser.add_argument("--retrieval_test_data", type=str, default=None,
                        help="Path to retrieval test CSV (image, caption columns)")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "gpu"])
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--distributed_backend", type=str, default="nccl")
    parser.add_argument("--distributed_init_method", type=str, default="tcp://127.0.0.1:5432")
    parser.add_argument("--device_ids", nargs="+", default=None)
    parser.add_argument("--num_clients", type=int, default=10, help="Number of FL clients")
    parser.add_argument("--fl_rounds", type=int, default=50, help="Number of FL communication rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="Local training epochs per round")
    parser.add_argument("--fl_eval_freq", type=int, default=5, help="Evaluate every N rounds")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.5, help="Dirichlet concentration for Non-IID split")
    parser.add_argument("--client_noise_ratios", type=str, default="0.2,0.2,0.4,0.4,0.6,0.6,0.2,0.4,0.4,0.6",
                        help="Comma-separated noise ratios per client")
    parser.add_argument("--fed_method", type=str, default="feddasw_v2",
                        choices=["feddasw_v2"],
                        help="FL aggregation/training method")
    parser.add_argument("--shared_set_size", type=int, default=1000,
                        help="Number of unique images in the public clean reference set")
    parser.add_argument("--train_mode", type=str, default="prompt",
                        choices=["prompt"],
                        help="Train prompt tokens with the CLIP backbone frozen")
    parser.add_argument("--dasw_cma_beta", type=float, default=2.5,
                        help="FedDASW CMA annealing strength beta")
    parser.add_argument("--dasw_asym_lambda", type=float, default=0.3,
                        help="FedDASW asymmetry penalty coefficient lambda_a")
    parser.add_argument("--queue_size", type=int, default=4096,
                        help="Cross-modal momentum queue size for negative mining (0=disabled)")
    parser.add_argument("--queue_start_size", type=int, default=0,
                        help="Effective queue size threshold before momentum negatives are enabled (<=0 uses queue_size)")
    parser.add_argument("--queue_loss_weight", type=float, default=0.001,
                        help="Blend weight for queue-augmented loss (0=batch-only, 1=full queue loss)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()
