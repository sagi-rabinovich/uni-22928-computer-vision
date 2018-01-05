from feature_extractor import FeatureExtractor

from image_dataset import ImageDataset
from pipeline.code_book import CodeBook
from progress_bar import ProgressBar

progressBar = ProgressBar()
imageDataset = ImageDataset()
featureExtractor = FeatureExtractor(progressBar)
imgs = imageDataset.load_training_data(10000)
featureExtractor.extractAndCompute(imgs)
codeBook = CodeBook(progressBar, imgs)
codeBook.computeCodeVector(imgs)
codeBook.printExampleCodes(imgs, 10, 20)
