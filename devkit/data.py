import pandas as pd
import numpy as np


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
        CHECK_ID, PERSON_ID, DAY, MONTH, GOOD_ID = 0, 1, 3, 2, 5
        for row in train.values:
            self.checksByPeople.setdefault(row[PERSON_ID], set())\
                .add((int(row[CHECK_ID][2:]), row[DAY], row[MONTH]))
            self.dishesByChecks.setdefault(int(row[CHECK_ID][2:]), []).append(row[GOOD_ID])

    def __loadMenuData(self, menuTaggedPath, menuTrainPath):
        menu = pd.read_csv(menuTaggedPath)
        self.menuTrain = pd.read_csv(menuTrainPath)
        self.tagedDishes = {}
        for row in menu.values:
            self.tagedDishes[row[0]] = row[2:]
