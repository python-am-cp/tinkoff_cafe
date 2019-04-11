import pandas as pd
import solution as Solver
import testpack.testpack as tp

allData = pd.read_csv("data/train.csv")
trainSize = int(allData.shape[0] * 0.8)
train = allData.iloc[:trainSize, :]
test = allData.iloc[trainSize:, :]

train.to_csv("solution/train.csv", index=False)
test.to_csv("solution/test.csv", index=False)

model = Solver.Model()
model.train("solution/train.csv", "data/menu_train.csv", "data/menu_tagged.csv")
model.load_params("data/menu_train.csv", "data/menu_tagged.csv")

data = Solver.Data("solution/test.csv", "data/menu_train.csv", "data/menu_tagged.csv")
features = []
labels = []
for human in data.getPeopleIds():
    checks = data.getChecksList(human)
    testChecks = checks[int(len(checks) * 0.8):]
    for check in testChecks:
        features.append([human, check[1], check[2]])
        labels.append(data.getDishesList(check[0]))

print("Features and labels was created")
testpack = tp.TestingPackage(features, labels, model)
print(testpack.getTheMetric())
