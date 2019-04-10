import numpy as np
import math


class Model:
    def __init__(self, data):
        self.data = data
        self.NUM_TYPES = 8
        self.NUM_TAGS = 19
        self.quantedPreds = {}
        self.preferences = {}
        self.precalc()

    def getPreferences(self, humanId):
        NUM_TAGS = 9
        checks = self.data.getChecksList(humanId)
        preferences = np.zeros(shape=(self.NUM_TYPES, self.NUM_TAGS))
        typeCounters = np.zeros(self.NUM_TYPES)
        for check in checks:
            for dish in self.data.getDishesList(check[0]):
                tags = self.data.getTagsList(dish)
                types = self.getTypes(tags)
                for type in types:
                    typeCounters[type] += 1
                    preferences[type] = preferences[type] + tags[self.NUM_TYPES:]
        for type in range(self.NUM_TYPES):
            if typeCounters[type] != 0:
                preferences[type] /= typeCounters[type]
        if len(checks):
            typeCounters /= len(checks)
        return (typeCounters, preferences)

    def getTypes(self, tags):
        return np.array([i for i in range(self.NUM_TYPES) if tags[i] != 0])

    def getQuantedPred(self, typeCounter):
        answer = np.zeros(self.NUM_TYPES)
        for type in range(self.NUM_TYPES):
            bottom, upper = math.floor(typeCounter[type]), math.ceil(typeCounter[type])
            upperChance = upper - typeCounter[type]
            answer[type] = upper if np.random.rand() > upperChance else bottom
        # print("A: ", answer)
        return answer

    def getThirsts(self, quantedPred, preferences, day, month):
        dailyMenu = self.data.getTodayMenu(day, month)
        thirsts = []
        assert len(dailyMenu) != 0
        for dish in dailyMenu:
            tags = self.data.getTagsList(dish)[self.NUM_TYPES:]
            types = self.getTypes(tags)
            thirst = 0
            for type in types:
                thirst = max(thirst, np.dot(tags, preferences[type]))
            thirsts.append([thirst, dish])
        thirsts.sort(reverse=True)
        return thirsts

    def precalc(self):
        for human in self.data.getPeopleIds():
            typeCounters, preference = self.getPreferences(human)
            quantedPred = self.getQuantedPred(typeCounters)
            self.preferences[human] = preference
            self.quantedPreds[human] = quantedPred

    def predict(self, features):
        humanId, day, month = features
        #print("D:", self.quantedPreds)
        quantedPred, preferences = self.quantedPreds[humanId].copy(), self.preferences[humanId]
        thirsts = self.getThirsts(quantedPred, preferences, day, month)
        #print("Q: ", quantedPred)
        labels = []
        sorted = [[], [], [], [], [], [], [], []]
        for thirst, dish in thirsts:
            types = self.getTypes(self.data.getTagsList(dish))
            for type in types:
                sorted[type].append(dish)
        for type in range(self.NUM_TYPES):
            for dish in sorted[type]:
                if quantedPred[type] == 0:
                    break
                labels.append(dish)
                quantedPred[type] -= 1
        return labels
