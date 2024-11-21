import argparse


def init():
    parser = argparse.ArgumentParser(description="PyTorch")

    parser.add_argument('--llava_gpu_id', type=str, default='3', help='which gpu to use')

    args = parser.parse_args()
    return args
