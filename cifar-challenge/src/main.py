import argparse
import logging

from cifa_challenge.cifar_model import CifarModel

if __name__ == "__main__":
    # Logging
    # create logger with 'spam_application'
    logger = logging.getLogger('cifar-challenge')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('main.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info('Starting main')

    parser = argparse.ArgumentParser(description='CIFAR-10 Challenge')
    parser.add_argument('--dataset-dir', help='root directory for the CIFAR-10 dataset', required=True)
    parser.add_argument('--samples', help='Number of samples to take from the dataset', type=int, default=10000)
    args = parser.parse_args()

    cifar_model = CifarModel(dataset_dir=args.dataset_dir, samples=args.samples)
    cifar_model.train_best_pipeline()

    logger.info('Done...')
