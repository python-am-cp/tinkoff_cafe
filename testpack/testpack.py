class TestingPackage:
    def __init__(self, features, labels, model):
        self.features = features
        self.labels = labels
        self.model = model

    def f1Metrics(self, predicted, trueAnswer):
        if len(predicted) == 0:
            return (0, 0)
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
        return (parameterP, parameterZ)

    def getTheMetric(self, prints = True):
        result, midP, midZ = .0, .0, .0
        count = min(len(self.features), len(self.labels))
        for index in range(count):
            if index % (count // 100) == 0:
                if prints:
                    print(str(index // (count // 100)) + "%")
            predicted = self.model.predict(self.features[index])
            p, z = self.f1Metrics(predicted, self.labels[index])
            midP += p
            midZ += z
            if p + z != 0:
                result += 2 * (p * z) / (p + z)
        return (result / count, midP / count, midZ / count)
