import argparse
from core.config import Config
from core.factory import build_training_components
from core.trainer import Trainer
from core.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.toml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()
    config = Config.load(args.config)
    set_seed(config.data.seed)
    components = build_training_components(config)
    trainer = Trainer(config, components)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted by user")


if __name__ == "__main__":
    main()
