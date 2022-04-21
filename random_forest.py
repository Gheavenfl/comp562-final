import tree
import random
from concurrent.futures import ProcessPoolExecutor as Pool

def prediction(predict_input):
    sample, trees = predict_input
    predictions = []
    for t in trees:
        predictions.append(tree.prediction((sample, t)))
    return max(set(predictions), key=predictions.count)

def predictions(predict_input):
    print("start prediction")
    data, trees = predict_input
    pred = []
    for i, sample in enumerate(data):
        pred.append(prediction((sample, trees)))
    print("finish prediction")
    return pred

def sub_sampling(dataset, sample_len):
    sample = []
    while len(sample) < sample_len:
        index = random.randrange(len(dataset))
        sample.append(dataset[index])
    return sample

def train_model(train_input):
    train_set, train_label, min_size, max_depth, num_trees, num_features, sample_size = train_input
    # train_set -> the training data
    # min_size -> the minimum number of samples in a non-terminal node
    # max_depth -> the maximize number of layers of the tree
    # num_trees -> number of sub-trees to train
    # num_features -> number of features to split on
    # sample_size -> the size of the sub-sample we create for each tree
    print("training forest")
    padded_train_set = [train_set[i] + [train_label[i]] for i in range(len(train_set))]
    trees = None
    sub_sample_list = []
    for i in range(num_trees):
        sub_sample_list.append(sub_sampling(padded_train_set, sample_size))
    args = ((sub_sample, min_size, max_depth, num_features) for sub_sample in sub_sample_list)
    with Pool() as p:
        trees = p.map(tree.train_model, args)
    trees = list(trees)
    print("finish training forest")
    return trees
