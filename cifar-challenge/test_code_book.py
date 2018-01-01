from code_book import CodeBook
from feature_extractor import FeatureExtractor
from image_dataset import ImageDataset
from progress_bar import ProgressBar

progressBar = ProgressBar()
imageDataset = ImageDataset()
featureExtractor = FeatureExtractor(progressBar)
nimages = 5
imgs = imageDataset.load_training_data(nimages)
featureExtractor.extractAndCompute(imgs)
codeBook = CodeBook(progressBar, imgs)
codeBook.computeCodeVector(imgs)
codeBook.printExampleCodes(imgs, 10, 20)
