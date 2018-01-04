from sklearn.pipeline import Pipeline

from cifa_challenge.image_grid_plot import plot_image_grid
from cifa_challenge.pipeline.color_space_transformer import ColorSpaceTransformer
from cifa_challenge.progress_bar import ProgressBar
from image_dataset import ImageDataset


def execute_pipeline():
    image_dataset = ImageDataset()
    image_contexts = image_dataset.load_training_data()[:5]
    test_image_contexts = image_dataset.load_test_data()

    descriptor_extract_bar = ProgressBar()
    descriptor_extract_bar.prefix = 'Extracting descriptor'
    codeBookBar = ProgressBar()
    codeBookBar.prefix = 'Code Book'
    # pipeline = Pipeline([("dense_detector", DenseDetector(radiuses=[3, 6, 8, 12, 16], overlap=0.3)),
    #                      ("surf_descriptor", FeatureDescriptor(descriptor_extract_bar, cv2.xfeatures2d.SURF_create())),
    #                      ("code_book", CodeBook(codeBookBar)),
    #                      ("normalization", StandardScaler(copy=False)),
    #                      ("dim_reduction", PCA(0.75)),
    #                      ("classification", svm.SVC(decision_function_shape='ovr', cache_size=2000, verbose=True))])
    #
    # pipeline = pipeline.fit(image_contexts, [x.label for x in image_contexts])
    # score = pipeline.score(test_image_contexts, [x.label for x in test_image_contexts])
    # print('Score: ' + str(score))

    pipeline = Pipeline(
        [("color_space_transformer", ColorSpaceTransformer(transformation='transformed_color_distribution'))])
    images = pipeline.fit_transform(image_contexts, None)

    img_grid = [images]
    plot_image_grid(img_grid, (1, len(images)), '../../results/transformed_color_space.png', normalize=True)
    print('Done')
