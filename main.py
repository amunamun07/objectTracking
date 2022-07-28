import argparse
from centroid.single_object_tracking import SingleObjectTracking


def main(args):
    args = parse_args(args)
    if args.mode.lower() == "single":
        parser = SingleObjectTracking(args.tracker, args.video)
        parser.start_tracking()


def parse_args(args):
    parser = argparse.ArgumentParser(description='For Object Tracking')
    parser.add_argument('--mode', help='Single or Multiple')
    parser.add_argument('--tracker', help='The Tracking Algorithm')
    parser.add_argument('--video', help='Path to the Video')
    return parser.parse_args(args)


if __name__ == '__main__':
    import sys

    main(sys.argv[1:])
