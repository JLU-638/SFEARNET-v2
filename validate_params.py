#!/usr/bin/env python3
"""
Simple test to verify parameter defaults for the normalized multi-objective loss.
"""

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Train Change Detection Models')

    parser.add_argument('--lamda', default=1, type=float, help='edge loss weight for backward compatibility')
    parser.add_argument('--beta', default=0.3, type=float, help='balance factor for contrast loss')
    parser.add_argument('--gamma', default=0.3, type=float, help='edge loss convex weight')
    parser.add_argument('--alpha', default=0.5, type=float, help='segmentation blend weight between CE and IoU')
    parser.add_argument('--max_test_batches', default=50, type=int, help='max batches per epoch for quick test; 0 means use full loader')
    parser.add_argument('--num_epochs', default=5, type=int, help='train epoch number')
    parser.add_argument('--print_every_batches', default=10, type=int, help='print metrics every N batches during training')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    print("Parameter validation:")
    print(f"  beta: {args.beta}")
    print(f"  gamma: {args.gamma}")
    print(f"  alpha: {args.alpha}")
    print(f"  beta + gamma: {args.beta + args.gamma}")
    print(f"  max_test_batches: {args.max_test_batches}")
    print(f"  num_epochs: {args.num_epochs}")
    print(f"  print_every_batches: {args.print_every_batches}")

    # Check constraints
    valid = args.beta + args.gamma <= 1.0
    print(f"\nValidation:")
    print(f"  beta + gamma <= 1.0: {valid}")

    if valid:
        print("✓ Parameters are valid for normalized multi-objective loss")
        print("✓ Ready to run quick validation training (50 batches per epoch, 5 epochs)")
    else:
        print("✗ Parameters violate convex combination constraint")
        print("  Please ensure beta + gamma <= 1.0")