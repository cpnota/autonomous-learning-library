import argparse
from all.experiments import plot_returns_100


def main():
    parser = argparse.ArgumentParser(description="Plots the results of experiments.")
    parser.add_argument("--logdir", help="Output directory", default='runs')
    parser.add_argument("--timesteps", type=int, default=-1, help="The final point will be fixed to this x-value")
    args = parser.parse_args()
    plot_returns_100(args.logdir, timesteps=args.timesteps)


if __name__ == "__main__":
    main()
