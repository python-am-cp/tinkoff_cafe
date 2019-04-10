import devkit.data as datapack
import devkit.mymodel as model
import testpack.testpack as tp

data = datapack.Data("data/train.csv", "data/menu_train.csv", "data/menu_tagged.csv")
model = model.Model(data)
print("Data was readed and precalculated")
features = []
labels = []
for human in data.getPeopleIds():
    for check in data.getChecksList(human):
        features.append([human, check[1], check[2]])
        labels.append(data.getDishesList(check[0]))
print("Features and labels was created")
testpack = tp.TestingPackage(features, labels, model)
# print("Testing...")
print(testpack.getTheMetric())
