import logging

from my_pipeline import execute_pipeline

if (__name__ == "__main__"):
    # Logging
    # create logger with 'spam_application'
    logger = logging.getLogger('cifar-challenge')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('../../main.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info('Starting main')

    execute_pipeline()

    logger.info('Done...')
