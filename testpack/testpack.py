import  numpy as np
class TestingPackage:
    def __init__(self, features, labels, model):
        self.features = features
        self.labels = labels
        self.model = model
    def f1Metrics(self, predicted, trueAnswer):
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
        return 0 if parameterP**2 + parameterZ**2 == 0 else 2 * (parameterP * parameterZ) / (parameterP + parameterZ)
    def getTheMetric(self):
        result = .0
        count = min(len(self.features), len(self.labels))
        for index in range(count):
            predicted = self.model.predict(self.features[index])
            result += self.f1Metrics(predicted, self.labels[index])
        return 0 if count == 0 else result / count
