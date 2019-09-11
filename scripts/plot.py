import argparse
from all.experiments import plot_returns_100


def plot():
    parser = argparse.ArgumentParser(description="Plots the results of experiments.")
    parser.add_argument("dir", help="Output directory.")
    args = parser.parse_args()
    plot_returns_100(args.dir)

if __name__ == "__main__":
    plot()
