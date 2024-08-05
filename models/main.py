import argparse
import logging

from modules.autoencoder.inference import Inferencer as AutoencoderInferencer
from modules.autoencoder.train import Trainer as AutoencoderTrainer
from modules.lstm.inference import LSTMInferencer
from modules.lstm.train import Trainer as LSTMTrainer
from utils.log_setup import setup_logging

setup_logging()

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or infer an autoencoder model.")
    
    # Create mutually exclusive group
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mode", choices=["train", "infer"], help="Mode: train or infer")
    parser.add_argument("--model", choices=["autoencoder", "lstm"], help="Model: autoencoder or lstm")
    
    return parser.parse_args()

def train(model: str) -> None:
    if model == "autoencoder":
        logging.info("Training autoencoder")
        trainer = AutoencoderTrainer()
    elif model == "lstm":
        logging.info("Training LSTM autoencoder")
        trainer = LSTMTrainer()
    trainer.run()

def infer(model: str) -> None: 
    if model == "autoencoder":
        logging.info("Inferring autoencoder")
        inferer = AutoencoderInferencer()
    elif model == "lstm":
        logging.info("Inferring LSTM autoencoder")
        inferer = LSTMInferencer()
    inferer.run()

def run(mode: str, model: str) -> None:
    if mode == "train":
        train(model)
    elif mode == "infer":
        infer(model)
    else:
        raise ValueError(f"Invalid mode: {mode}")

def main() -> None:
    args = get_args()
    if not args.model:
        raise ValueError("Model must be specified when mode is train or infer")
    run(args.mode, args.model)

if __name__ == "__main__":
    main()