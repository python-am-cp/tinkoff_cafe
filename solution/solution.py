import pickle
import math
import pandas as pd
import numpy as np
from collections import defaultdict


class Data:
    def __init__(self, trainPath, menuTrainPath, menuTaggedPath):
        self.__loadTrainData(trainPath)
        self.__loadMenuData(menuTaggedPath, menuTrainPath)

    def getPeopleIds(self):
        return np.fromiter(self.checksByPeople.keys(), dtype=int)

    def getChecksList(self, humanId):
        return np.array(list(self.checksByPeople[humanId]), dtype=int)

    def getDishesList(self, checkId, day=0, month=0):
        return np.array(self.dishesByChecks[checkId])

    def getTagsList(self, dish):
        return np.array(self.tagedDishes[dish])

    def getTodayMenu(self, day, month):
        return self.menuTrain[(self.menuTrain["day"] == day) & (self.menuTrain["month"] == month)][
            "good_id"].values

    def __loadTrainData(self, trainPath):
        train = pd.read_csv(trainPath)
        self.checksByPeople = {}  # key: person id; value: (check id, day, month)
        self.dishesByChecks = {}  # key: check id;  value: list of dish ids...
        CHECK_ID, PERSON_ID, MONTH, DAY, GOOD_ID = 0, 1, 2, 3, 5
        for row in train.values:
            self.checksByPeople.setdefault(row[PERSON_ID], set()).add((int(row[CHECK_ID][2:]), row[DAY], row[MONTH]))
            self.dishesByChecks.setdefault(int(row[CHECK_ID][2:]), []).append(row[GOOD_ID])

    def __loadMenuData(self, menuTaggedPath, menuTrainPath):
        menu = pd.read_csv(menuTaggedPath)
        self.menuTrain = pd.read_csv(menuTrainPath)
        self.tagedDishes = {}
        for row in menu.values:
            self.tagedDishes[row[0]] = row[2:]


class Model:
    def __init__(self):
        self.prefsByHuman = defaultdict(dict)  # key - human_id | value - dict of dishes counted in all checks
        self.allPeoplePrefs = defaultdict(int)  # key - dish_id, value - count of occurrences in checks
        self.dailyMenu = defaultdict(list)  # menu for every day
        self.quantPreds = defaultdict(list)  # quantitative prediction
        self.dishTypes = defaultdict(list)  # types of dishes
        self.paramsIsLoaded = False

    def train(self, train, menu_train, goods):
        entriesNum = defaultdict(int)
        dayChecker = defaultdict(bool)
        self.taggedMenu = pd.read_csv(goods)  # for load params
        self.data = Data(train, menu_train, goods)
        for human in self.data.getPeopleIds():
            self.prefsByHuman[int(human)] = defaultdict(int)
            typeCounters = np.zeros(8)
            countOfChecks = 0
            for dish in self.taggedMenu.values:
                typeTags = dish[2:10].tolist()
                self.dishTypes[int(dish[0])] = [i for i in range(len(typeTags)) if typeTags[i]]
            for check in self.data.getChecksList(human):
                countOfChecks += 1
                day, month = check[1], check[2]  # [1] - day, [2] - month
                if not dayChecker[(day, month)]:
                    dayChecker[(day, month)] = True
                    dailyMenu = self.data.getTodayMenu(day, month)
                    for dish in dailyMenu:
                        entriesNum[dish] += 1
                for dish in self.data.getDishesList(check[0]):  # [0] - checkId
                    self.prefsByHuman[int(human)][
                        int(dish)] += 1  # value - the number of dish occurrences in this person's checks
                    self.allPeoplePrefs[int(dish)] += 1
                    types = self.dishTypes[dish]
                    for type in types:
                        typeCounters[type] += 1
            self.quantPreds[int(human)] = (typeCounters / countOfChecks).tolist()
        for dish in entriesNum:
            count = entriesNum[dish]
            assert count != 0
            self.allPeoplePrefs[int(dish)] /= float(count)
            for human in self.data.getPeopleIds():
                self.prefsByHuman[int(human)][int(dish)] /= float(count)

        file = open("preferencesByHuman", "wb")
        pickle.dump(self.prefsByHuman, file)
        file = open("allPeoplePreferences", "wb")
        pickle.dump(self.allPeoplePrefs, file)
        file = open("quantativePredictions", "wb")
        pickle.dump(self.quantPreds, file)
        print("Was trained")

    def load_params(self, menuTestFName, taggedMenuFName):
        menuTest = pd.read_csv(menuTestFName)
        self.taggedMenu = pd.read_csv(taggedMenuFName)
        for dish in self.taggedMenu.values:
            typeTags = dish[2:10].tolist()
            self.dishTypes[int(dish[0])] = [i for i in range(len(typeTags)) if typeTags[i]]
        for row in menuTest.values:  # Creatring daily menu:
            month, day, goodId = [int(row[i]) for i in range(3)]
            self.dailyMenu[(day, month)].append(goodId)
        file = open("preferencesByHuman", "rb")  # Load model params (blaclbox):
        self.prefsByHuman = pickle.load(file)
        file = open("allPeoplePreferences", "rb")
        self.allPeoplePrefs = pickle.load(file)
        file = open("quantativePredictions", "rb")
        self.quantPreds = pickle.load(file)
        self.paramsIsLoaded = True

    def getValueByQuant(self, quantativePred):
        answer = [0] * 8
        for type in range(len(quantativePred)):
            bottom, upper = math.floor(quantativePred[type]), math.ceil(quantativePred[type])
            upperChance = upper - quantativePred[type]
            answer[type] = upper if np.random.rand() - 0.05 > upperChance else bottom
        return answer

    def predict(self, features):
        assert self.paramsIsLoaded  # Please call .load_params(...)
        humanId, day, month = features
        preferences = []
        if humanId not in self.prefsByHuman:
            preferences = self.allPeoplePrefs
        else:
            preferences = self.prefsByHuman[humanId]

        assert (day, month) in self.dailyMenu, (day, month, features)  # No information on today's menu in menuTestFName
        todayMenu = self.dailyMenu[(day, month)]
        sortedDishes = defaultdict(list)
        for dish in todayMenu:
            types = self.dishTypes[dish]
            for type in types:
                # if preferences[dish] == 0: #PROVERIT ETU TEORIO
                #    preferences[dish] = self.allPeoplePrefs[dish] #PROVERITb
                sortedDishes[type].append((preferences[dish], dish))
        labels = []
        quantPreds = self.getValueByQuant(self.quantPreds[humanId])
        for type in range(8):
            sortedDishes[type].sort(reverse=True)
            for counter in range(quantPreds[type]):
                if len(sortedDishes[type]):
                    labels.append(sortedDishes[type][0][1])
        return labels
