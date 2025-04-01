import json

def saveJson(data):
    with open("data\data.json","w") as file:
        json.dump(data,file)

def loadData():
    with open("data\data.json","r") as file:
        loadData = json.load(file)
        return loadData