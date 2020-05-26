import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import shutil

with open("config.yaml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

POSITIVE = cfg['model'].get("path_positive")
NEGATIVE = cfg['model'].get("path_negative")
LABELS = {NEGATIVE: 0, POSITIVE: 1}
checkpoint_dir = "model/checkpoint/"
model_dir = "model/checkpoint/best/"

class Net(nn.Module):
    def __init__(self, MAKE_TRAINING_DATA = False, VALIDATION_PROPORTION = 0.1, device="cpu"):
        super().__init__()
        self.device = device
        self.training = False
        print("Loading net, device: ", self.device)
        self.tracker = 10
        self.feature_size = (cfg['feature_vector'].get("border_x") * 2)+1

        self.fc1 = nn.Linear(self.feature_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        #self.fc4 = nn.Linear(512, 512)
        #self.fc6 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(128, 2)


        if MAKE_TRAINING_DATA == True:
            self.makeTrainingData(self.feature_size, VALIDATION_PROPORTION = 0.1)

    def forward(self, x, TRAINING = False):
        if TRAINING == False:
            if self.device != "cpu":
                x = self.dataTransform(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        #x = F.relu(self.fc6(x))
        x = self.fc5(x)

        return F.softmax(x, dim=1)

    def makeTrainingData(self, feature_size, VALIDATION_PROPORTION = 0.1):


        negative, positive = [], []

        for label in LABELS:
            for file in tqdm(os.listdir(label)):

                file_path = os.path.join(label, file)
                npArr = np.load(file_path, allow_pickle=True)
                for data in npArr:
                    if len(data[0]) == feature_size:
                        if label == POSITIVE:
                            positive.append(data)
                        elif label == NEGATIVE:
                            negative.append(data)

        positive = np.array(positive)
        np.random.shuffle(positive)
        negative = np.array(negative)
        np.random.shuffle(negative)

        val_size = int(((len(positive)+len(negative))*VALIDATION_PROPORTION)*0.5)

        train_positive = positive[:-val_size]
        train_negative = negative[:-val_size]
        validation_positive = positive[-val_size:]
        validation_negative = negative[-val_size:]

        training = np.concatenate((train_positive, train_negative))
        np.random.shuffle(training)
        validation = np.concatenate((validation_positive, validation_negative))
        np.random.shuffle(validation)

        np.save("model/training_data.npy", training)
        np.save("model/validation_data.npy", validation)

        # Make data into tensors
        train_X = torch.Tensor([i[0] for i in training]).view(-1,feature_size,1)
        train_y = torch.Tensor([i[1] for i in training])

        test_X = torch.Tensor([i[0] for i in validation]).view(-1,feature_size,1)
        test_y = torch.Tensor([i[1] for i in validation])

        torch.save(train_X, "model/train_x.pt")
        torch.save(train_y, "model/train_y.pt")
        torch.save(test_X, "model/test_x.pt")
        torch.save(test_y, "model/test_y.pt")

    def train(self, patience=3):

        self.training = True

        self.val_best = 100000000
        #load training and validation data
        train_X = torch.load("model/train_x.pt")
        train_y = torch.load("model/train_y.pt")
        test_X = torch.load("model/test_x.pt")
        test_y = torch.load("model/test_y.pt")

        if self.device != "cpu":
            train_X = self.dataTransform(train_X)
            train_y = self.dataTransform(train_y)
            test_X = self.dataTransform(test_X)
            test_y = self.dataTransform(test_y)
        print("Loaded ", train_X.shape[0], " training samples.")
        print("Loaded ", test_X.shape[0], " test samples.")
            # Initialize optimizer and loss function
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        loss_function = nn.BCELoss()

        #track metrics
        self.train_loss, self.train_acc, self.train_precision, self.train_recall, self.val_loss, self.val_acc, self.val_precision, self.val_recall = [], [], [], [], [], [], [], []

        # train the classifier
        #for epoch in tqdm(range(EPOCHS)):
        epoch = 0
        print("Training...")
        while self.early_stopping(patience=patience) is not True:
            for i in range(len(train_X)):

                self.zero_grad()
                output = self.forward(train_X[i].view(-1,self.feature_size), TRAINING =True)[0]

                loss = loss_function(output, train_y[i])
                loss.backward()
                optimizer.step()    # Does the update

            for i in range(len(test_X)):
                output_val = self.forward(test_X[i].view(-1,self.feature_size), TRAINING = True)[0]
                loss_val = loss_function(output_val, test_y[i])
            self.val_loss.append(loss_val.item())
            self.val_acc.append(self.getAccuracy(test_X, test_y))
            self.val_recall.append(self.getRecall(test_X, test_y))
            self.val_precision.append(self.getPrecision(test_X, test_y))
            self.train_loss.append(loss.item())
            self.train_acc.append(self.getAccuracy(train_X, train_y))
            self.train_recall.append(self.getRecall(train_X, train_y))
            self.train_precision.append(self.getPrecision(train_X, train_y))

            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': self.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            is_best = False
            if self.val_loss[epoch] < self.val_best:
                is_best = True
            self.save_checkpoint(checkpoint, is_best, checkpoint_dir, model_dir)
            self.save_training_results()
            print("Epoch: ", epoch+1, "tracker: ", self.tracker, " validation loss:", loss_val.item(), " validation accuracy: ", self.val_acc[-1])
            #print("Epoch: ", epoch+1, "tracker: ", self.tracker, " validation loss:", loss_val.item())
            epoch += 1

        self.getROCCurve(test_X, test_y)
        auc_score = self.getAUCScore(test_X, test_y)
        print("AUC score: ", auc_score)

    def getAUCScore(self, testX, testY):

        predictions, labels = [], []
        with torch.no_grad():
            for j in range(len(testX)):
                label = torch.argmax(testY[j])
                label = label.item()
                labels.append(np.array([label]))
                predictions.append(np.array([self.predict_confidence(testX[j])]))


        labels = np.array(labels)
        predictions = np.array(predictions)
        auc_score = metrics.roc_auc_score(labels, predictions)

        return auc_score

    def getROCCurve(self, testX, testY):


        predictions, labels = [], []
        with torch.no_grad():
            for j in range(len(testX)):
                label = torch.argmax(testY[j])
                label = label.item()
                labels.append(np.array([label]))
                predictions.append(np.array([self.predict_confidence(testX[j])]))


        labels = np.array(labels)
        predictions = np.array(predictions)
        fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)

        fig = plt.plot()
        plt.plot(fpr, tpr)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.savefig('model/figures/ROC_curve.png')
        plt.savefig('model/figures/ROC_curve.pdf', format='pdf')
        plt.show()


    def getAccuracy(self, testX, testY):
        correct = 0
        total = 0

        with torch.no_grad():
            for j in range(len(testX)):
                real_class = torch.argmax(testY[j])
                predicted_class = self.predict(testX[j])

                if predicted_class == real_class:
                    correct += 1
                total += 1
        return round(correct/total,3)

    def getPrecision(self, testX, testY):
        true_positives = 0
        false_positives = 0

        with torch.no_grad():
            for j in range(len(testX)):
                real_class = torch.argmax(testY[j])
                predicted_class = self.predict(testX[j])

                if predicted_class == 1:
                    if predicted_class == real_class:
                        true_positives += 1
                    else:
                        false_positives += 1
        return round(true_positives/(false_positives+true_positives),3)

    def getRecall(self, testX, testY):
        true_positives = 0
        false_negatives = 0

        with torch.no_grad():
            for j in range(len(testX)):
                real_class = torch.argmax(testY[j])
                predicted_class = self.predict(testX[j])

                if predicted_class == 1:
                    if predicted_class == real_class:
                        true_positives += 1
                elif predicted_class == 0:
                    if real_class == 1:
                        false_negatives += 1
        return round(true_positives/(false_negatives+true_positives),3)


    def save_training_results(self, plot=False):

        np.save((cfg['model'].get("path") + "train_loss.npy"), self.train_loss)
        np.save((cfg['model'].get("path") + "val_loss.npy"), self.val_loss)
        np.save((cfg['model'].get("path") + "train_acc.npy"), self.train_acc)
        np.save((cfg['model'].get("path") + "train_prec.npy"), self.train_precision)
        np.save((cfg['model'].get("path") + "train_recall.npy"), self.train_recall)
        np.save((cfg['model'].get("path") + "val_acc.npy"), self.val_acc)
        np.save((cfg['model'].get("path") + "val_prec.npy"), self.val_precision)
        np.save((cfg['model'].get("path") + "val_recall.npy"), self.val_recall)

    def plot_training_results(self):
        valMin = np.amin(self.val_loss)
        valMinIndx = np.where(self.val_loss == valMin)
        valMinIndx = valMinIndx[0][0]
        fig, ax = plt.subplots(2, 2)
        #ax[0, 0].plot(self.train_loss)
        ax[0, 0].plot(self.val_loss, label="Validation loss")
        ax[0, 0].axvline(x = valMinIndx, color='r')
        ax[0, 0].set_ylabel('Loss')
        ax[0, 0].set_xlabel('Epochs')
        ax[0, 0].set_title("Training Loss")
        ax[0, 0].legend(loc='best')

        ax[1, 0].plot(self.train_acc, label="Training accuracy")
        ax[1, 0].plot(self.val_acc, label="Validation accuracy")
        ax[1, 0].axvline(x = valMinIndx, color='r')
        ax[1, 0].set_ylabel('Accuracy')
        ax[1, 0].set_xlabel('Epochs')
        ax[1, 0].set_title("Training Accuracy")
        ax[1, 0].legend(loc='best')

        ax[0, 1].plot(self.train_precision, label="Training precision")
        ax[0, 1].plot(self.val_precision, label="Validation precision")
        ax[0, 1].axvline(x = valMinIndx, color='r')
        ax[0, 1].set_ylabel('Precision')
        ax[0, 1].set_xlabel('Epochs')
        ax[0, 1].set_title("Training Precision")
        ax[0, 1].legend(loc='best')

        ax[1, 1].plot(self.train_recall, label="Training recall")
        ax[1, 1].plot(self.val_recall, label="Validation recall")
        ax[1, 1].axvline(x = valMinIndx, color='r')
        ax[1, 1].set_ylabel('Recall rate')
        ax[1, 1].set_xlabel('Epochs')
        ax[1, 1].set_title("Training Recall rate")
        ax[1, 1].legend(loc='best')

        plt.tight_layout()
        plt.savefig('model/figures/training_data.png')
        plt.savefig('model/figures/training_data.pdf', format='pdf')
        plt.show()

    def save_checkpoint(self, state, is_best, checkpoint_dir, best_model_dir):

        f_path = (checkpoint_dir + "checkpoint.pth")
        torch.save(state, f_path)
        if is_best:
            best_fpath = (best_model_dir + "model.pth")
            shutil.copyfile(f_path, best_fpath)
            self.save()

    def load_checkpoint(self, checkpoint_fpath, model, optimizer):
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch']

    def early_stopping(self, patience):

        best_val_old = self.val_best
        if len(self.val_loss) is not 0:
            if self.val_best > self.val_loss[-1]:
                self.val_best = self.val_loss[-1]

            if best_val_old == self.val_best:
                self.tracker += -1
            else:
                self.tracker = patience

            if self.tracker <= 0:
                return True #stop the training
            else:
                return False



    def predict(self, x):

        output = self.forward(x.view(-1,self.feature_size))[0] #maybe remove [0]
        y = torch.argmax(output)
        return y

    def predict_confidence(self, x):
        """
        Returns the probability that the given input is class positive. [0, 1]
        """
        output = self.forward(x.view(-1,self.feature_size))[0] #maybe remove [0]
        y = output
        y = y[1].item()
        return y

    def predict_threshold(self, x, threshold):
        """
        Accepts a threshold value between 0 and 1. Returns true if predicted
        value is above threshold.
        """
        output = self.forward(x.view(-1,self.feature_size))[0] #maybe remove [0]
        y = output
        y = y[1].item()

        if y >= threshold:
            return 1
        else:
            return 0

    def save(self):
        torch.save(self.state_dict(), "model/classifier.pth")

    def loadTestData(self):
        test_X = torch.load("model/test_x.pt")
        test_y = torch.load("model/test_y.pt")
        return test_X, test_y

    def dataTransform(self, data):
        if self.device != 'cpu':
            return data.to(self.device)
        else:
            return data
