import argparse

from sim import DEFAULT_SIZE, Simulation
from ui import run


def main():
    parser = argparse.ArgumentParser(description="EMIT: continuous CA toy model")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    args = parser.parse_args()

    sim = Simulation(size=args.size)
    run(sim)


if __name__ == "__main__":
    main()
