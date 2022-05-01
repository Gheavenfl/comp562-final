def fourFolds():
    # split data into four folds
    # each fold sturctures as [train_messages, train_labels, validation_messages, validation_label]
    import pandas as pd
    import random
    from sklearn.feature_extraction.text import CountVectorizer
    # read csv file, generate list and messages as seperate labels
    df = pd.read_csv('spam.csv', encoding="ISO-8859-1", usecols=[0, 1])
    labels = df['v1'].tolist()
    label_transform = {"ham": 0, 'spam': 1}
    labels = [label_transform[label] for label in labels]
    messages = df['v2'].tolist()

    # bag of words representation
    bow_vectorizer = CountVectorizer(stop_words='english')
    bow_messages = bow_vectorizer.fit_transform(messages).todense()
    bow_messages = [b.tolist()[0] for b in bow_messages]

    # use K-fold Validation
    folds = []

    for i in range(4):
        total_len = len(labels)
        validation_indexes = [n for n in range(i * int(total_len * 0.25), (i + 1) * int(total_len * 0.25))]
        training_indexes = [n for n in range(len(labels)) if
                            n < i * len(labels) * 0.25 or n >= (i + 1) * len(labels) * 0.25]
        training_labels = [labels[i] for i in training_indexes]
        training_messages = [bow_messages[i] for i in training_indexes]
        validation_labels = [labels[i] for i in validation_indexes]
        validation_messages = [bow_messages[i] for i in validation_indexes]
        folds.append((training_messages, training_labels, validation_messages, validation_labels))
    return folds