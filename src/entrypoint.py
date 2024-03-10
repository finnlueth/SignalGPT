import argparse


def main(args):
    # Here you can put the code to start the learning process of your ML model
    # using the parameters in args
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Launch the learning process of an ML model.')
    
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the model.')
    parser.add_argument('--model_path', type=str, default='./model',
                        help='Path to save the trained model.')

    args = parser.parse_args()
    main(args)