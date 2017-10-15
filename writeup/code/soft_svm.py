import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import random
import matplotlib.pyplot as plt
import itertools
import time
import datetime
from visualize import VtkPointCloud
import vtk

if __name__ == "__main__":

	dataPath1 = "data/oakland_part3_am_rf.node_features"
	dataPath2 = "data/oakland_part3_an_rf.node_features"

	# NOTE:
	# format: x y z node_id node_label [features]

	data1 = pd.read_csv(dataPath1, header=None, delim_whitespace=True, comment="#")
	data2 = pd.read_csv(dataPath2, header=None, delim_whitespace=True, comment="#")
	data = pd.concat([data1, data2])

	## Might use later...
	#scaler = MinMaxScaler()
	#scaler.fit_transform( data[[5,6,7,8,9,10,11,12,13]])

	for i in range(5,14):
		data[i] = data[i]/max(data[i])

	dataY = data[4]

	trainAll, testAll, trainY, testY = train_test_split(data, dataY, test_size=0.2)
	trainX = np.array(trainAll[range(5,14)])
	testX = np.array(testAll[range(5,14)])
	trainY = np.array(trainY)
	testY = np.array(testY)

	classes = [1004, 1100, 1103, 1200, 1400]
	classNames = ['Veg', 'Wire', 'Pole', 'Ground', 'Facade']
	trainY = label_binarize(trainY, classes=classes, neg_label=-1, pos_label=1)
	testY = label_binarize(testY, classes=classes, neg_label=-1, pos_label=1)

	lambdaHyper = 1
	tHyper = 250000

	trainingStartTime = time.time()
	avgWeights = np.zeros([len(classes), trainX.shape[1]])

	for currClass in xrange(len(classes)):

		currTrainY = trainY[:, currClass]
		weights = np.zeros([tHyper, trainX.shape[1]])
		thetas = np.zeros(trainX.shape[1])

		for t in xrange(tHyper):

			weights[t] = np.multiply((1/lambdaHyper), thetas)

			m = random.randint(0, trainX.shape[0] - 1)
			currX = trainX[m]
			currY = currTrainY[m]

			if (currY * np.dot(weights[t], currX)) < 1:

				thetas = thetas + np.multiply(currX, currY)

		avgWeights[currClass] = np.mean(weights, axis=0)

	trainingEndTime = time.time()

	testingStartTime = time.time()
	results = np.dot(testX,avgWeights.transpose())
	predictions = []
	groundTruth = []
	confusionMatrix = np.zeros([len(classes), len(classes)])
	numCorrect = 0

	for i in xrange(results.shape[0]):
		predictions.append(np.argmax(results[i]))
		groundTruth.append(np.argmax(testY[i]))
		confusionMatrix[groundTruth[-1]][predictions[-1]] = confusionMatrix[groundTruth[-1]][predictions[-1]] + 1
		if groundTruth[-1] == predictions[-1]:
			numCorrect += 1
	testingEndTime = time.time()

	print("")
	print("It took " + str(trainingEndTime - trainingStartTime) + " seconds to train over " + str(trainX.shape[0])  + " datapoints...")
	print("It took " + str(testingEndTime - testingStartTime) + " seconds to test over " + str(testX.shape[0])  + " datapoints...")
	print("")
	print("Accuracy = " + str(float(numCorrect)/results.shape[0]))

	print("")
	print("Confusion Matrix = ")
	print("")
	for x in xrange(len(classes)):
		for y in xrange(len(classes)):
			print("%5s" % str(int(confusionMatrix[x, y]))),
		print("")

	# Write new classifications to file
	ts = time.time()
	st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
	outputFilePath = "data/classified" + st + ".node_features"
	outputFile = open(outputFilePath, "w")

	for i in xrange(testAll.shape[0]):
		currLine = str(testAll.iloc[i, 0]) + " " + str(testAll.iloc[i, 1]) + " " + str(testAll.iloc[i, 2]) + " "
		currLine += str(testAll.iloc[i, 3]) + " " + str(classes[predictions[i]]) + " " + str(testAll.iloc[i, 5]) + " "
		currLine += str(testAll.iloc[i, 6]) + " " + str(testAll.iloc[i, 7]) + " " + str(testAll.iloc[i, 8]) + " "
		currLine += str(testAll.iloc[i, 9]) + " " + str(testAll.iloc[i, 10]) + " " + str(testAll.iloc[i, 11]) + " "
		currLine += str(testAll.iloc[i, 12]) + " " + str(testAll.iloc[i, 13]) + " " + str(testAll.iloc[i, 14]) + "\n"
		outputFile.write(currLine)

	outputFile.close()
	print("")
	print("Wrote classified point cloud data to " + outputFilePath)

	pointCloud = VtkPointCloud()
	pointCloud.readFile(outputFilePath)

	# Renderer
	renderer = vtk.vtkRenderer()
	renderer.AddActor(pointCloud.vtkActor)
	renderer.SetBackground(.1, .1, .1)
	renderer.ResetCamera()

	# Render Window
	renderWindow = vtk.vtkRenderWindow()
	renderWindow.AddRenderer(renderer)

	# Interactor
	renderWindowInteractor = vtk.vtkRenderWindowInteractor()
	renderWindowInteractor.SetRenderWindow(renderWindow)

	# Begin Interaction
	renderWindow.Render()
	renderWindowInteractor.Start()
