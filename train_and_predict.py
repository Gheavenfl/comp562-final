from concurrent.futures import ProcessPoolExecutor as Pool

def train_and_predict_by_fold(folds, train, predict, extra_params = None):
    # folds -> folds of data with each fold organized as [training_message, training_label, validation_message, validation_label]
    # train -> train function
    # predict -> predict function
    # extra_params -> extra parameters used by train function, organized as a tuple
    # returns predictions for folds organized as a list
    train_input_by_fold = ((fold[0], fold[1]) for fold in folds)
    if extra_params:
        train_input_by_fold = (inp + extra_params for inp in train_input_by_fold)
    models_by_fold = None
    with Pool() as p:
        models_by_fold = p.map(train, train_input_by_fold)
    models_by_fold = list(models_by_fold)
    predict_input_by_fold = ((folds[i][2], models_by_fold[i]) for i in range(len(folds)))
    predictions_by_fold = None
    with Pool() as p:
        predictions_by_fold = p.map(predict, predict_input_by_fold)
    predictions_by_fold = list(predictions_by_fold)
    return predictions_by_fold
