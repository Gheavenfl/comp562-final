import math


def trainModel(train_input):
    train_set, train_label = train_input
    print("training")
    spam_set = [train_set[i] for i in range(len(train_set)) if train_label[i] == 1]
    ham_set = [train_set[i] for i in range(len(train_set)) if train_label[i] == 0]
    spam_prob = len(spam_set) / float(len(train_set))
    ham_prob = 1.0 - spam_prob
    spam_set_mean, spam_set_stdev = trainSet(spam_set)
    ham_set_mean, ham_set_stdev = trainSet(ham_set)
    print("finish training")
    return (spam_set_mean, spam_set_stdev, ham_set_mean, ham_set_stdev, spam_prob, ham_prob)


def mean(data):
    return sum(data) / float(len(data))


def stdev(data):
    avg = mean(data)
    variance = sum([(x - avg) ** 2 for x in data]) / float(len(data) - 1)
    return math.sqrt(variance)


def logGaussianProbability(x, mean, stdev):
    if stdev == .0:
        stdev = 0.01
    temp = -((x - mean) ** 2 / (2 * stdev ** 2))
    return temp - math.log(math.sqrt(2 * math.pi) * stdev)


def trainSet(s):
    set_mean = []
    set_stdev = []
    for i in range(len(s[0])):
        # note that the last column is label
        column = [d[i] for d in s]
        me = mean(column)
        st = stdev(column)
        set_mean.append(me)
        set_stdev.append(st)
    return (set_mean, set_stdev)


def probability(d, set_mean, set_stdev, prob):
    # make a single prediction
    prob = math.log(prob)
    for i in range(len(d)):
        mean = set_mean[i]
        stdev = set_stdev[i]
        prob += logGaussianProbability(d[i], mean, stdev)
    return prob


def predictions(predict_input):
    data, model = predict_input
    print("predicting")
    spam_set_mean, spam_set_stdev, ham_set_mean, ham_set_stdev, spam_prob, ham_prob = model
    # assume each row is a set of data
    re = []  # 0->ham, 1->spam
    for d in data:
        prob1 = probability(d, spam_set_mean, spam_set_stdev, spam_prob)
        prob2 = probability(d, ham_set_mean, ham_set_stdev, ham_prob)
        if prob1 >= prob2:
            re.append(1)
        else:
            re.append(0)
    print("finish predicting")
    return re