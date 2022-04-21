import random
import math

def gini_impurity(nodes, all_labels):
    # calculate the weighted sum of gini impurity for all the nodes
    total_number_of_samples = float(sum([len(node) for node in nodes]))
    gini_impurity = 0.0
    for node in nodes:
        node_size = len(node)
        if node_size == 0:
            continue
        score = 1.0
        node_labels = [n[-1] for n in node]
        for label in all_labels:
            possibility = node_labels.count(label) / node_size
            score -= possibility ** 2
        gini_impurity += score * (node_size / total_number_of_samples)
    return gini_impurity


def train_tree_helper(node, min_size, max_depth, features_to_split, current_depth):
    left, right = node['nodes']
    del (node['nodes'])
    if not left or not right:
        # no longer splittable
        node['left'] = node['right'] = dominating_label(left + right)
        return
    if current_depth > max_depth:
        node['left'] = dominating_label(left)
        node['right'] = dominating_label(right)
        return
    if len(left) <= min_size:
        node['left'] = dominating_label(left)
    else:
        node['left'] = choose_split(left, features_to_split)
        train_tree_helper(node['left'], min_size, max_depth, features_to_split, current_depth + 1)
    if len(right) <= min_size:
        node['right'] = dominating_label(right)
    else:
        node['right'] = choose_split(right, features_to_split)
        train_tree_helper(node['right'], min_size, max_depth, features_to_split, current_depth + 1)


def train_model(train_input):
    # train individual trees in the forst
    sample_set, min_size, max_depth, num_features = train_input
    print("training tree")
    all_features = [i for i in range(len(sample_set[0]) - 1)]
    features_to_split = random.sample(all_features, num_features)
    tree_root = choose_split(sample_set, features_to_split)
    train_tree_helper(tree_root, min_size, max_depth, features_to_split, 1)
    print("finish training tree")
    return tree_root


def split_data(sample_set, feature_ind, val):
    left = [sample for sample in sample_set if sample[feature_ind] < val]
    right = [sample for sample in sample_set if sample[feature_ind] >= val]
    return (left, right)


def choose_split(sample_set, features_to_split):
    all_labels = list(set([sample[-1] for sample in sample_set]))
    final_splitting_index = None
    final_splitting_val = None
    final_splitting_nodes = None
    best_score = math.inf
    for feature in features_to_split:
        splitting_vals = set([s[feature] for s in sample_set])
        for splitting_val in splitting_vals:
            nodes = split_data(sample_set, feature, splitting_val)
            gini_score = gini_impurity(nodes, all_labels)
            if gini_score < best_score:
                final_splitting_index = feature
                final_splitting_val = splitting_val
                final_splitting_nodes = nodes
                best_score = gini_score
    return {'index': final_splitting_index, 'val': final_splitting_val, 'nodes': final_splitting_nodes}


def dominating_label(sample_set):
    labels = [sample[-1] for sample in sample_set]
    return max(set(labels), key=labels.count)


def prediction(predict_input):
    sample, tree = predict_input
    if sample[tree['index']] < tree['val']:
        # left
        if isinstance(tree['left'], dict):
            return prediction((sample, tree['left']))
        else:
            return tree['left']
    else:
        if isinstance(tree['right'], dict):
            return prediction((sample, tree['right']))
        else:
            return tree['right']







