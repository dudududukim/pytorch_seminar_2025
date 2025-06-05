import argparse

def main():
    parser = argparse.ArgumentParser(description='Simple training script')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--dataset', type=str, default= 'ImageNet1K', help='Dataset name')
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')

    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epoch}")

if __name__ == '__main__':
    main()
