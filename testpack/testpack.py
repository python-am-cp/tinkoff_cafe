import numpy as np


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


class TestingPackage:
    def __init__(self, features, labels, model):
        self.features = features
        self.labels = labels
        self.model = model

    def f1Metrics(self, predicted, trueAnswer):
        if len(predicted) == 0:
            return 0
        countedPred = {}
        for dish in predicted:
            if dish in countedPred:
                countedPred[dish] += 1
            else:
                countedPred[dish] = 1
        intersectSize = 0
        for dish in trueAnswer:
            if dish in countedPred and countedPred[dish] > 0:
                countedPred[dish] -= 1
                intersectSize += 1
        parameterP = intersectSize / len(predicted)
        parameterZ = intersectSize / len(trueAnswer)
        return 0 if intersectSize == 0 else 2 * (parameterP * parameterZ) / (parameterP + parameterZ)

    def getTheMetric(self):
        result = .0
        count = min(len(self.features), len(self.labels))
        # printProgressBar(0, 100, prefix='Progress:', suffix='Complete', length=50)
        for index in range(count):
            if index % (count // 100) == 0:
                print(str(index // (count // 100)) + "%")
                # printProgressBar(index // (count // 100), 100, prefix='Progress:', suffix='Complete', length=50)
            predicted = self.model.predict(self.features[index])
            #print(self.features[index])
            #print("My: ", predicted, "True: ", self.labels[index])
            result += self.f1Metrics(predicted, self.labels[index])
        return 0 if count == 0 else result / count
