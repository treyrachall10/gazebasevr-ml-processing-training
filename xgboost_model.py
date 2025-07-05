'''
IMPORTANT NOTE:
    - Good with tabular/structured data(one row = one example)
    - Good for ranking tasks(confidence ranking)
    - Not good for looking at sequences over time or multiple frames
    - Good for recommendation systems
    - Relatively lightweight
    - Good for small/medium datasets
        - Especially when dropout is included
    - Model expects input shape: [num_samples, num_features]
'''
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser
from preprocess_files import getXY, printLabelInfo, normalize_path, make_output_dirs

def createAndTrainModel(x, y):
    # Reshape Data
    x = np.array(x)
    x = np.reshape(x, (len(x), -1))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)

    num_classes = len(set(y))

    # Create model instance
    bst = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3, objective='multi:softmax', num_class=num_classes, tree_method='gpu_hist',
    predictor='gpu_predictor')

    print("Training on device: GPU" if "gpu" in bst.get_params()["tree_method"] else "Training on device: CPU")

    # Fit model
    bst.fit(X_train, y_train)

    #   Make predictions
    preds = bst.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test, preds))

if __name__ == "__main__":
    parser = ArgumentParser(description = "Preprocess round 1 files from GazeBaseVR data set")
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to existing directory containing GazeBaseVR data",
    )
    parser.add_argument(
        "--round_1_dir",
        type=str,
        required=True,
        help="Path to output directory for storing round 1 files"
    )
    parser.add_argument(
        "--norm_dir",
        type=str,
        required=True,
        help="path to output directory for storing normalized data"
    )
    args = parser.parse_args()

    input_dir = normalize_path(args.src)
    round_1_dir = normalize_path(args.round_1_dir)
    norm_dir = normalize_path(args.norm_dir)

    make_output_dirs(round_1_dir)
    make_output_dirs(norm_dir)
    x, y = getXY(input_dir, round_1_dir, norm_dir)
    printLabelInfo(y)
    createAndTrainModel(x, y)