# Imports here
import itertools
import os
import sys

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from skorch import NeuralNetClassifier

datasetPath = "./Dataset"
testImagesPath = "./Dataset/test/"
maskDatasetPath = datasetPath + "/masked dataset"
humanDatasetPath = datasetPath + "/human dataset"
nonHumanDatasetPath = datasetPath + "/non-human dataset"
preprocessedDataPath = datasetPath + "/data.npy"
resultsPath = "./Results/"
modelName = "kFoldModel.pkl"


class Data:
    def __init__(self):
        self.data = []
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []
        self.normalizedWeights = []
        self.labelsDict = {0: "Masked Human", 1: "Human", 2: "Non-Human"}

    def buildData(self):
        # Mask Dataset
        for path in os.listdir(maskDatasetPath):
            print(maskDatasetPath + "/" + path)
            img = cv2.imread(maskDatasetPath + "/" + path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (100, 100))
            self.data.append([np.array(img), 0])

        # Human Dataset
        for path in os.listdir(humanDatasetPath):
            print(humanDatasetPath + "/" + path)
            img = cv2.imread(humanDatasetPath + "/" + path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (100, 100))
            self.data.append([np.array(img), 1])

        # Non Human Dataset
        for path in os.listdir(nonHumanDatasetPath):
            print(nonHumanDatasetPath + "/" + path)
            img = cv2.imread(nonHumanDatasetPath + "/" + path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (100, 100))
            self.data.append([np.array(img), 2])

        np.random.shuffle(self.data)
        np.save(preprocessedDataPath, self.data)

    def loadPreprocessedData(self):
        return np.load(preprocessedDataPath, allow_pickle=True)

    def buildDataLoader(self, build=False):
        print("Build DataLoader")
        if build:
            self.buildData()
        loadedData = self.loadPreprocessedData()
        data = np.zeros((loadedData.shape[0], 3, 100, 100), dtype=np.float32)
        labels = np.zeros((loadedData.shape[0]), dtype=np.int64)
        sampleData = loadedData[:, 0]
        sampleLabels = loadedData[:, 1]
        i = 0
        for image, label in zip(sampleData, sampleLabels):
            data[i, :, :, :] = image.reshape(3, 100, 100)
            labels[i] = label
            i = i + 1

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data, labels, test_size=0.05,
                                                                                random_state=3)


class CNN(nn.Module):
    def __init__(self):
        print("Building CNN")
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=3, padding=1),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.Dropout2d(p=0.05),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # output: 512 x 4 x 4
            nn.Dropout2d(p=0.05),

            nn.Flatten(),
            nn.Linear(320000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Linear(256, 3))

    def forward(self, xb):
        return self.network(xb)


class TrainTest:

    def __init__(self, data):
        self.data = data
        self.model = {}

    def getDevice(self):
        isCudaAvailable = torch.cuda.is_available()
        print("IsCudaAvailable:", isCudaAvailable)
        device = torch.device('cuda') if isCudaAvailable else torch.device('cpu')
        return device

    def trainAndSaveModel(self):
        cnnModel = NeuralNetClassifier(CNN, lr=0.0001, optimizer=torch.optim.Adam, criterion=torch.nn.CrossEntropyLoss,
                                       device=self.getDevice())
        cnnModel.set_params(train_split=False, verbose=0)
        params = {"max_epochs": [10, 15]}
        gs = GridSearchCV(cnnModel, params, cv=10, scoring="accuracy", verbose=10, return_train_score=True)

        gs.fit(self.data.X_train, self.data.y_train)
        print("Best Score: " + gs.best_score_)
        print("CV Results " + gs.cv_results_)
        self.model = gs.best_estimator_
        joblib.dump(self.model, modelName, compress=1)

    def loadModel(self):
        self.model = joblib.load(modelName)

    def evaluate(self, testData=True):
        if testData:
            data, labels = self.data.X_test, self.data.y_test
        else:
            data, labels = self.data.X_train, self.data.y_train

        y_pred = self.model.predict(data)
        return y_pred

    def printClassificationReportAndPlotConfusionMatrix(self):
        classes = ["Masked Human", "Human", "Non-Human"]
        print("Test Classification Report")
        testPredictionLabels = self.evaluate()
        print(classification_report(self.data.y_test, testPredictionLabels))
        self.plot_confusion_matrix(confusion_matrix(self.data.y_test, testPredictionLabels), classes,
                                   title="Test Confusion Matrix")

        print("Train Classification Report")
        trainPredictionLabels = self.evaluate(testData=False)
        print(classification_report(self.data.y_train, trainPredictionLabels))
        self.plot_confusion_matrix(confusion_matrix(self.data.y_train, trainPredictionLabels), classes,
                                   title="Train Confusion Matrix")

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes, rotation=-60)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.axis('scaled')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(resultsPath + title)
        plt.show(block=True)

    def plotTestPredictions(self, imageName):
        img = cv2.imread(testImagesPath + imageName, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (100, 100))
        floatImg = img.astype(np.float32)
        y_pred = self.model.predict(floatImg.reshape(-1, 3, 100, 100))
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig = plt.figure()
        plt.title("Predicted Label:{}".format(y_pred))
        plt.imshow(image)
        plt.savefig(resultsPath + imageName)
        plt.close(fig)
    
    def predictProbabilities(self, imageName):
        img = cv2.imread(testImagesPath + imageName, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (100, 100))
        floatImg = img.astype(np.float32)
        y_pred = self.model.predict_proba(floatImg.reshape(-1, 3, 100, 100))
        return y_pred[0]



buildData = sys.argv[1] == "True"
trainModel = sys.argv[2] == "True"

dataObject = Data()
dataObject.buildDataLoader(buildData)

trainTest = TrainTest(dataObject)
if trainModel:
    trainTest.trainAndSaveModel()

trainTest.loadModel()

#trainTest.printClassificationReportAndPlotConfusionMatrix()

#for fileName in os.listdir(testImagesPath):
#    trainTest.plotTestPredictions(fileName)

probabilities = trainTest.predictProbabilities("2.jpg")
print(probabilities)
