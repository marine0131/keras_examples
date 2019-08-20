import autokeras as ak
from tensorflow import keras
from sklearn.metrics import classification_report


def main():
    ak.constant.Constant.MAX_BATCH_SIZE=16
    ak.constant.Constant.MAX_LAYERS=5
    ((trainX, trainY), (testX, testY)) = keras.datasets.cifar10.load_data()
    trainX = trainX.astype("float")/255.0
    testX =  testX.astype("float")/255.0

    labels = ["airplane", "automobile", "bird", "cat", "deer",
                  "dog", "frog", "horse", "ship", "truck"]
    seconds = 3600

    model = ak.ImageClassifier(verbose=True)
    model.fit(trainX, trainY, time_limit=seconds)
    model.final_fit(trainX, trainY, testX, testY, retrain=True)

    #evaluate the model
    score = model.evaluate(testX, testY)
    predictions = model.predict(testX)
    report = classification_report(testY, predictions,
                                   target_names=labels)

    print(report)

if __name__ == "__main__":
    main()
