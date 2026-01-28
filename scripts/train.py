import argparse
from core.config import Config
from core.factory import build_training_components
from core.trainer import Trainer


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
    components = build_training_components(config)
    trainer = Trainer(config, components)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted by user")


if __name__ == "__main__":
    main()
