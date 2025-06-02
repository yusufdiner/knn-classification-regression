import pandas as pd
import numpy as np
import random
from math import sqrt
dataFrame = pd.read_csv("subset_16P.csv")
dataFrame["Personality"]=dataFrame["Personality"].map({"ESTJ":0,"ENTJ":1,"ESFJ":2,"ENFJ":3,"ISTJ":4,"ISFJ":5,"INTJ":6,"INFJ":7,"ESTP":8,"ESFP":9,"ENTP":10,"ENFP":11,
         "ISTP":12,"ISFP":13,"INTP":14,"INFP":15}).astype(int)

dataFrame2 = pd.read_csv("energy_efficiency_data.csv")
#dataFrame  # for jupyter
#dataFrame2  # for jupyter
class Classification:
    def __init__(self, kNN, fold, data, accuracyTab, precisionTab, recallTab, weighted=False, featureScale=False):
        self.kNN = kNN
        self.fold = fold
        self.weighted = weighted
        self.data = data
        self.featureScale = featureScale
        self.accuracyTab = accuracyTab
        self.precisionTab = precisionTab
        self.recallTab = recallTab

    def featureScaling(self, dataSet):  # feature normalization function
        maxVal = np.max(dataSet)
        minVal = np.min(dataSet)
        array = np.zeros(len(dataSet))
        for i in range(len(dataSet)):
            array[i] = (dataSet[i] - minVal) / (maxVal - minVal)
        return array

    def calculatePrecision(self, matrix):  # calculate precision of predict
        sm = 0.0
        sumMatrix = np.sum(matrix, axis=0)
        for a in range(len(matrix)):

            tp = matrix[a][a]
            if sumMatrix[a] != 0 or tp != 0:
                sm += (tp / sumMatrix[a])

        return float("{:.6f}".format(sm / (len(matrix))))

    def calculateRecall(self, matrix):  # calculate recall of predict
        sm = 0.0
        sumMatrix = np.sum(matrix, axis=1)
        for a in range(len(matrix)):
            tp = matrix[a][a]
            if sumMatrix[a] != 0 or tp != 0:
                sm += (tp / sumMatrix[a])

        return float("{:.6f}".format(sm / (len(matrix))))

    def calculateAccuracy(self, matrix):  # calculate accuracy of predict
        diagsm = np.trace(matrix)
        allSum = np.sum(matrix)

        return float("{:.6f}".format(diagsm / allSum))

    def shuffle(self):  # shuffle dataset
        rangex = 10000
        ls = random.sample(range(rangex), rangex)
        shf = np.zeros((rangex, 62))
        cnt = 0

        for a in ls:
            shf[cnt] = self.data[a]
            cnt += 1
        return shf

    def predict(self):
        shuffled = self.data
        if (self.featureScale == True):  # check whether to do feature normalization
            for a in range(1, 61):
                shuffled[:, a] = self.featureScaling(shuffled[:, a])
        datasize = len(shuffled) // self.fold
        splitted_x = shuffled[:, 0:]

        accArr = self.accuracyTab
        preArr = self.precisionTab
        recArr = self.recallTab
        countPredictMistake = 1
        for k in range(self.fold):
            # split dataset
            x_train = np.concatenate((splitted_x[:(k * datasize)], splitted_x[((k + 1) * datasize):]))
            x_valid = splitted_x[(k * datasize):][:datasize]
            matrix = np.zeros((16, 16))
            for d in x_valid:
                nw = np.zeros((self.kNN, 63), dtype=float)
                count = 0
                for c in x_train:
                    length = self.euclidian_distance(c[1:-1], d[1:-1])
                    newArray = np.array([0.0])
                    newArray[0] = length
                    j = np.append(c, newArray)  # add distance to the list

                    if count < self.kNN:
                        nw[count] = j
                        if (count == self.kNN - 1):
                            nw = nw[nw[:, -1].argsort()]
                    else:
                        if nw[-1][-1] > length:
                            nw[-1] = j
                            nw = nw[nw[:, -1].argsort()]

                    count += 1
                nw = nw[nw[:, -1].argsort()]

                skill = np.zeros(16, dtype=float)

                if (not self.weighted):  # check whether to do weight
                    for z in nw:
                        skill[int(z[-2])] += 1
                        predicted = np.where(skill == np.max(skill))
                else:
                    for z in nw:
                        skill[int(z[-2])] += (1 / (z[-1]))
                        predicted = np.where(skill == np.max(skill))

                matrix[int(d[-1])][predicted[0][0]] += 1
            isWeighted = 0
            isScale = 0
            if self.weighted:
                isWeighted = 0
            else:
                isWeighted = 2
            if self.featureScale:
                isScale = 0
            else:
                isScale = 1
            strfold = "Fold " + str(k + 1)

            # update tables
            accArr[k][((self.kNN * 2) - 2) + isWeighted + isScale] = self.calculateAccuracy(matrix)
            recArr[k][((self.kNN * 2) - 2) + isWeighted + isScale] = self.calculateRecall(matrix)
            preArr[k][((self.kNN * 2) - 2) + isWeighted + isScale] = self.calculatePrecision(matrix)

    def euclidian_distance(self, a, b):  # calculate euclidian distance between 2 points
        result = np.float64(0)
        for i in range(len(a)):
            result = result + np.power((a[i] - b[i]), 2, dtype=np.float64)
        return sqrt(result)


class Regression():
    def __init__(self, kNN, fold, data, maeTab1, maeTab2, weighted=False, featureScale=False):
        self.kNN = kNN
        self.fold = fold
        self.weighted = weighted
        self.data = data
        self.maeTab1 = maeTab1
        self.maeTab2 = maeTab2
        self.weighted = weighted
        self.featureScale = featureScale

    def featureScaling(self, dataSet):
        maxVal = np.max(dataSet)
        minVal = np.min(dataSet)
        array = np.zeros(len(dataSet))
        for i in range(len(dataSet)):
            array[i] = (dataSet[i] - minVal) / (maxVal - minVal)
        return array

    def shuffle(self):
        rangex = 768
        ls = random.sample(range(rangex), rangex)
        shf = np.zeros((rangex, 10))
        cnt = 0

        for a in ls:
            shf[cnt] = self.data[a]
            cnt += 1
        return shf

    def euclidian_distance(self, a, b):
        result = np.float64(0)
        for i in range(len(a)):
            result = result + np.power((a[i] - b[i]), 2, dtype=np.float64)

        return sqrt(result)

    def meanAbsoluteError(self, predicted, gt):  # calculate MAE
        sm = 0
        for a in range(len(predicted)):
            sm += abs(gt[a] - predicted[a])

        return float("{:.6f}".format((1 / len(predicted)) * sm))

    def weightedFunc(self, matrix):
        for a in range(len(matrix)):
            if not (a + 1 >= len(matrix)):
                cross = matrix[a][0] * matrix[a + 1][1] + matrix[a][1] * matrix[a + 1][0]
                p = matrix[a][1] + matrix[a + 1][1]
                matrix[a + 1][0] = cross / p
                matrix[a + 1][1] = p

        return matrix[len(matrix) - 1][0]

    def predict(self):

        shuffled = self.data
        if (self.featureScale == True):
            for a in range(0, 10):
                shuffled[:, a] = self.featureScaling(shuffled[:, a])
        datasize = len(shuffled) // self.fold
        splitted_x = shuffled[:, 0:]
        mae1Arr = self.maeTab1
        mae2Arr = self.maeTab2
        for k in range(self.fold):
            # split dataset
            x_train = np.concatenate((splitted_x[:(k * datasize)], splitted_x[((k + 1) * datasize):]))  # train set
            x_valid = splitted_x[(k * datasize):][:datasize]  # test set
            cooling = np.zeros(len(x_valid))
            heating = np.zeros(len(x_valid))
            ct = 0
            for d in x_valid:
                nw = np.zeros((self.kNN, 11), dtype=float)
                count = 0
                for c in x_train:
                    length = self.euclidian_distance(c[0:-2], d[0:-2])
                    newArray = np.array([0.0])
                    newArray[0] = length
                    j = np.append(c, newArray)

                    if count < self.kNN:
                        nw[count] = j
                        if (count == self.kNN - 1):
                            nw = nw[nw[:, -1].argsort()]

                    else:
                        if float(nw[-1][-1]) > float(length):
                            nw[-1] = j
                            nw = nw[nw[:, -1].argsort()]

                    count += 1

                nw = nw[nw[:, -1].argsort()]

                if (self.weighted == False):
                    heatingLoad = np.mean(nw, axis=0)[-3]
                    coolingLoad = np.mean(nw, axis=0)[-2]
                    cooling[ct] = coolingLoad
                    heating[ct] = heatingLoad
                    ct += 1
                else:
                    cooling[ct] = self.weightedFunc(nw[:, -2:])  # cooling
                    heating[ct] = self.weightedFunc(nw[:, -3::2])  #
                    ct += 1

            meanAbsoluteErrorCooling = self.meanAbsoluteError(cooling, x_valid[:, -1])
            meanAbsoluteErrorHeating = self.meanAbsoluteError(heating, x_valid[:, -2])

            isWeighted = 0
            isScale = 0
            if self.weighted:
                isWeighted = 0
            else:
                isWeighted = 2
            if self.featureScale:
                isScale = 0
            else:
                isScale = 1
            strfold = "Fold " + str(k + 1)

            # update tables
            mae1Arr[k][((self.kNN * 2) - 2) + isWeighted + isScale] = meanAbsoluteErrorHeating

            mae2Arr[k][((self.kNN * 2) - 2) + isWeighted + isScale] = meanAbsoluteErrorCooling


# PART I
df = np.array(dataFrame[0:]).astype(np.float64)

rangex = 10000
ls = random.sample(range(rangex), rangex)
dfx = np.zeros((rangex, 62))
cnt = 0

for a in ls:
    dfx[cnt] = df[a]
    cnt += 1

accuracyTab = np.zeros((5, 20))
precisionTab = np.zeros((5, 20))
recallTab = np.zeros((5, 20))
kNNList = [1, 3, 5, 7, 9]
for a in kNNList:
    clf1 = Classification(a, 5, df, accuracyTab, precisionTab, recallTab, True, True)
    clf1.predict()
    clf2 = Classification(a, 5, df, accuracyTab, precisionTab, recallTab, True, False)
    clf2.predict()
    clf3 = Classification(a, 5, df, accuracyTab, precisionTab, recallTab, False, True)
    clf3.predict()
    clf4 = Classification(a, 5, df, accuracyTab, precisionTab, recallTab, False, False)
    clf4.predict()
average1 = np.zeros((1, 20))
for i in range(len(np.mean(accuracyTab, axis=0))):
    average1[0][i] = float("{:.6f}".format(np.mean(accuracyTab, axis=0)[i]))

average2 = np.zeros((1, 20))
for i in range(len(np.mean(precisionTab, axis=0))):
    average2[0][i] = float("{:.6f}".format(np.mean(precisionTab, axis=0)[i]))

average3 = np.zeros((1, 20))
for i in range(len(np.mean(recallTab, axis=0))):
    average3[0][i] = float("{:.6f}".format(np.mean(recallTab, axis=0)[i]))

accuracyTab = np.vstack([accuracyTab, average1])
precisionTab = np.vstack([precisionTab, average2])
recallTab = np.vstack([recallTab, average3])
tableAccuracy = pd.DataFrame(accuracyTab,
                             columns=pd.MultiIndex.from_product([[1, 3, 5, 7, 9], ['Yes', 'No'], ["Yes", "No"]],
                                                                names=["Nearest Neighbor", "Weighted KNN",
                                                                       "Feature Normalization"]),
                             index=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average"])
tablePrecision = pd.DataFrame(precisionTab,
                              columns=pd.MultiIndex.from_product([[1, 3, 5, 7, 9], ['Yes', 'No'], ["Yes", "No"]],
                                                                 names=["Nearest Neighbor", "Weighted KNN",
                                                                        "Feature Normalization"]),
                              index=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average"])
tableRecall = pd.DataFrame(recallTab,
                           columns=pd.MultiIndex.from_product([[1, 3, 5, 7, 9], ['Yes', 'No'], ["Yes", "No"]],
                                                              names=["Nearest Neighbor", "Weighted KNN",
                                                                     "Feature Normalization"]),
                           index=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average"])
tableAccuracy.to_csv("accSave.csv")
tablePrecision.to_csv("preSave.csv")
tableRecall.to_csv("recSave.csv")

# PART II


maeTab1 = np.zeros((5, 20))  # heating
maeTab2 = np.zeros((5, 20))

df3 = np.array(dataFrame2[:])
rangex = 768
ls = random.sample(range(rangex), rangex)
df2 = np.zeros((rangex, 10))
cnt = 0

for a in ls:
    df2[cnt] = df3[a]
    cnt += 1

kNNList = [1, 3, 5, 7, 9]
for a in kNNList:
    clf1 = Regression(a, 5, df2, maeTab1, maeTab2, True, True)
    clf1.predict()
    clf2 = Regression(a, 5, df2, maeTab1, maeTab2, True, False)
    clf2.predict()
    clf3 = Regression(a, 5, df2, maeTab1, maeTab2, False, True)
    clf3.predict()
    clf4 = Regression(a, 5, df2, maeTab1, maeTab2, False, False)
    clf4.predict()
average1 = np.zeros((1, 20))
for i in range(len(np.mean(maeTab1, axis=0))):
    average1[0][i] = float("{:.6f}".format(np.mean(maeTab1, axis=0)[i]))

average2 = np.zeros((1, 20))
for i in range(len(np.mean(maeTab2, axis=0))):
    average2[0][i] = float("{:.6f}".format(np.mean(maeTab2, axis=0)[i]))

maeTab1 = np.vstack([maeTab1, average1])
maeTab2 = np.vstack([maeTab2, average2])

tableHeating = pd.DataFrame(maeTab1, columns=pd.MultiIndex.from_product([[1, 3, 5, 7, 9], ['Yes', 'No'], ["Yes", "No"]],
                                                                        names=["Nearest Neighbor", "Weighted KNN",
                                                                               "Feature Normalization"]),
                            index=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average"])
tableCooling = pd.DataFrame(maeTab2, columns=pd.MultiIndex.from_product([[1, 3, 5, 7, 9], ['Yes', 'No'], ["Yes", "No"]],
                                                                        names=["Nearest Neighbor", "Weighted KNN",
                                                                               "Feature Normalization"]),
                            index=["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Average"])

tableHeating.to_csv("heating.csv")
tableCooling.to_csv("cooling.csv")

