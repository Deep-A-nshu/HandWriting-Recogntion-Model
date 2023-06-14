import argparse
import random
import cv2
import editdistance
import numpy as np
# import tensorflow as tf

# from DataLoader import Batch, DataLoader, FilePaths
from Preprosessing import preprocessor, wer
from Model import DecoderType, Model
# from SpellChecker import correct_sentence



# class FilePaths:
#     """ Filenames and paths to data """
#     fnCharList = '../model/charList.txt'
#     fnWordCharList = '../model/wordCharList.txt'
#     fnCorpus = '../data/corpus.txt'
#     fnAccuracy = '../model/accuracy.txt'
#     fnTrain = '../data/'
#     fnInfer = '../data/testImage1.png'  ## path to recognize the single image

from autocorrect import spell


def correct_sentence(line):
    lines = line.strip().split(' ')
    new_line = ""
    # similar_word = {}
    for l in lines:
        new_line += spell(l) + " "
    # similar_word[l]=spell.candidates(l)
    return new_line



class Sample:
    """ Sample from the dataset """

    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath


class Batch:
    """ Batch containing images and ground truth texts """

    def __init__(self, gtTexts, imgs):
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts


class DataLoader:

    def __init__(self, filePath, batchSize, imgSize, maxTextLen, load_aug=True):
        "loader for dataset at given location, preprocess images and text according to parameters"

        # assert filePath[-1] == '/'

        self.dataAugmentation = True # False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []
        self.corpus = ""
        self.WordCharList = ""
        f = open("../data/" + 'lines.txt')
        chars = set()
        for line in f:
            # ignore comment line
            if not line or line[0] == '#':
                continue

            lineSplit = line.strip().split(' ')  ## remove the space and split with ' '
            # assert len(lineSplit) >= 9

            # filename: part1-part2-part3 --> part1/part1-part2/part1-part2-part3.png
            fileNameSplit = lineSplit[0].split('-')
            #print(fileNameSplit)
            fileName = filePath + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '/' +\
                       lineSplit[0] + '.png'

            # GT text are columns starting at 10
            # see the lines.txt and check where the GT text starts, in this case it is 10
            gtText_list = lineSplit[8].split('|')
            gtText = self.truncateLabel(' '.join(gtText_list), maxTextLen)
            chars = chars.union(set(list(gtText)))  ## taking the unique characters present
            self.corpus += gtText
            # check if image is not empty
            # put sample into list
            self.samples.append(Sample(gtText, fileName))



        # split into training and validation set: 95% - 10%
        splitIdx = int(0.95 * len(self.samples))
        self.trainSamples = self.samples[:splitIdx]
        self.validationSamples = self.samples[splitIdx:]
        print("\n\n\n<----------------- TRAINING-TEST SPLIT OF THE INPUT DATA -------------------->\n\n\n")
        print("Total: {} Train: {}, Validation: {}".format(len(self.samples) , len(self.trainSamples), len(self.validationSamples)))
        print("\n\n\n<------------------------------------------------------------------------------------->\n\n\n")
        # put lines into lists
        self.trainLines = [x.gtText for x in self.trainSamples]
        self.validationLines = [x.gtText for x in self.validationSamples]

        # number of randomly chosen samples per epoch for training
        self.numTrainSamplesPerEpoch = 9500

        # start with train set
        self.trainSet()

        # list of all chars in dataset
        self.charList = sorted(list(chars))
        for i in self.charList:
            if(i.isalpha()):
                self.WordCharList += i
        

    def truncateLabel(self, text, maxTextLen):
        # ctc_loss can't compute loss if it cannot find a mapping between text label and input
        # labels. Repeat letters cost double because of the blank symbol needing to be inserted.
        # If a too-long label is provided, ctc_loss returns an infinite gradient
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text


    def trainSet(self):
        "switch to randomly chosen subset of training set"
        self.dataAugmentation = True
        self.currIdx = 0
        random.shuffle(self.trainSamples) # shuffle the samples in each epoch
        self.samples = self.trainSamples #[:self.numTrainSamplesPerEpoch]

    def validationSet(self):
        "switch to validation set"
        self.dataAugmentation = False
        self.currIdx = 0
        self.samples = self.validationSamples

    def getIteratorInfo(self):
        "current batch index and overall number of batches"
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)

    def hasNext(self):
        "iterator"
        return self.currIdx + self.batchSize <= len(self.samples)

    def getNext(self):
        "iterator"
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        gtTexts = [self.samples[i].gtText for i in batchRange]
        imgs = [preprocessor(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize)
            for i in batchRange]
        self.currIdx += self.batchSize
        return Batch(gtTexts, imgs)

def splitParaToLines(img):
    #binary
    ret ,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        #dilation
    kernel = np.ones((10,1000), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
        #find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    sentences = []
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = img[y:y+h, x:x+w]
        sentences.append(roi)
        # show ROI
    #     cv2.imshow('segment no:'+str(i),roi)
        cv2.rectangle(img,(x,y),( x + w, y + h ),(255,0,0),2)
    #     cv2.waitKey(0)
    sentences = sentences[::-1]
    for img in sentences:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        (wid , ht) = (800 , 64)
        (h , w ) = img.shape
        fx = w / wid
        fy = h / ht
        f = max(fx , fy)
        newSize = (int(max(min(wid , w // f) , 1)) ,
                    int(max(min(ht , h // f) , 1)))
        
        img = cv2.resize(img , newSize , interpolation=cv2.INTER_CUBIC)

        target = np.ones([ht , wid]) * 255
        target[0:newSize[1] , 0:newSize[0]] = img

        img = cv2.transpose(target)
        (m, s) = cv2.meanStdDev(img)
        m = m[0][0]
        s = s[0][0]
        img = img - m
        img = img / s if s>0 else img
    return sentences


def train(model, loader):
    """ Train the neural network """
    epoch = 0  # Number of training epochs since start
    bestCharErrorRate = float('inf')  # Best valdiation character error rate
    noImprovementSince = 0  # Number of epochs no improvement of character error rate occured
    earlyStopping = 25  # Stop training after this number of epochs without improvement
    batchNum = 0

    totalEpoch = len(loader.trainSamples)//Model.batchSize 

    while True:
        epoch += 1
        print('Epoch:', epoch, '/', totalEpoch)

        # Train
        print('Train neural network')
        loader.trainSet()
        while loader.hasNext():
            batchNum += 1
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch, batchNum)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        # Validate
        charErrorRate, addressAccuracy, wordErrorRate = validate(model, loader)
        # cer_summary = tf.Summary(value=[tf.Summary.Value(
        #     tag='charErrorRate', simple_value=charErrorRate)])  # Tensorboard: Track charErrorRate
        # # Tensorboard: Add cer_summary to writer
        # model.writer.add_summary(cer_summary, epoch)
        # address_summary = tf.Summary(value=[tf.Summary.Value(
        #     tag='addressAccuracy', simple_value=addressAccuracy)])  # Tensorboard: Track addressAccuracy
        # # Tensorboard: Add address_summary to writer
        # model.writer.add_summary(address_summary, epoch)
        # wer_summary = tf.Summary(value=[tf.Summary.Value(
        #     tag='wordErrorRate', simple_value=wordErrorRate)])  # Tensorboard: Track wordErrorRate
        # # Tensorboard: Add wer_summary to writer
        # model.writer.add_summary(wer_summary, epoch)

        # If best validation accuracy so far, save model parameters
        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved, save model')
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            print("\n\n\n<---------------   VALIDATION CHARACTER ERROR RATE   ------------>\n\n\n")
            print(charErrorRate * 100.0)
            print("\n\n\n<---------------------------------------------------------------------->\n\n\n")
            # open(FilePaths.fnAccuracy, 'w').write(
            #     'Validation character error rate of saved model: %f%%' % (charErrorRate*100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        # Stop training if no more improvement in the last x epochs
        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' %
                  earlyStopping)
            break


def validate(model, loader):
    """ Validate neural network """
    print('\n\n\n<----------------------------------------->')
    print('Validate neural network')
    print("<-------------------------------------------->\n\n")
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0

    totalCER = []
    totalWER = []
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        batch = loader.getNext()
        recognized = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            ## editdistance
            currCER = dist/max(len(recognized[i]), len(batch.gtTexts[i]))
            totalCER.append(currCER)

            currWER = wer(recognized[i].split(), batch.gtTexts[i].split())
            totalWER.append(currWER)

            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' +
                  batch.gtTexts[i] + '"', '->', '"' + recognized[i] + '"')

    # Print validation result
    charErrorRate = sum(totalCER)/len(totalCER)
    addressAccuracy = numWordOK / numWordTotal
    wordErrorRate = sum(totalWER)/len(totalWER)
    print('Character error rate: %f%%. Address accuracy: %f%%. Word error rate: %f%%' %
          (charErrorRate*100.0, addressAccuracy*100.0, wordErrorRate*100.0))
    return charErrorRate, addressAccuracy, wordErrorRate


# def load_different_image():
#     imgs = []
#     for i in range(1, Model.batchSize):
#        imgs.append(preprocessor(cv2.imread("../data/check_image/a ({}).png".format(i), cv2.IMREAD_GRAYSCALE), Model.imgSize, enhance=False))
#     return imgs


# def generate_random_images():
#     return np.random.random((Model.batchSize, Model.imgSize[0], Model.imgSize[1]))


def infer(model, fnImg):
    """ Recognize text in image provided by file path """
    img = preprocessor(cv2.imread(fnImg, cv2.IMREAD_GRAYSCALE), imgSize=Model.imgSize)
    if img is None:
        print("Image not found")

    imgs = splitParaToLines(img)
    batch = Batch(None, imgs)
    recognized = model.inferBatch(batch)  # recognize text
    print("Predicted text by the model: ")
    for i in range(recognized):
        print(recognized[i])
    
    print("Corrected text: ")
    for i in range(recognized):
        print(correct_sentence(recognized[i]))
    return recognized
    # print("Without Correction", recognized[0])
    # print("With Correction", correct_sentence(recognized[0]))
    # return recognized[0]



def main():
    """ Main function """
    # Opptional command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", help="train the neural network", action="store_true")
    parser.add_argument(
        "--validate", help="validate the neural network", action="store_true")
    parser.add_argument(
        "--wordbeamsearch", help="use word beam search instead of best path decoding", action="store_true")
    args = parser.parse_args()

    decoderType = DecoderType.BestPath
    input_label_file_path = "../data/"
    # print("<------------------------------ LOADING TRAINING DATA -------------------------->\n\n\n")
    # loader = DataLoader(input_label_file_path , Model.batchSize , Model.imgSize , Model.maxTextLen)
    # print("\n\n\n<------------------------------------- DATA LOADED -------------------------------->\n\n\n")
    # model = Model(loader.charList , loader.WordCharList , loader.corpus , decoderType)
    # train(model , loader)
    # Train or validate on Cinnamon dataset
    if args.train or args.validate:
        # Load training data, create TF model
        loader = DataLoader(input_label_file_path, Model.batchSize,
                            Model.imgSize, Model.maxTextLen, load_aug=True)

        # Execute training or validation
        if args.train:
            model = Model(loader.charList, loader.wordCharList , loader.corpus , decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, loader.wordCharList , loader.corpus , decoderType, mustRestore=False)
            validate(model, loader)

    # Infer text on test image
    else:
        print("<---------- THIS IS THE CUSTOM TEST PART.WILL BE UPDATED LATERRRR.--------->")
        # print(open(FilePaths.fnAccuracy).read())
        # model = Model(open(FilePaths.fnCharList).read(),
        #               decoderType, mustRestore=False)
        # infer(model, FilePaths.fnInfer)


# def infer_by_web(path, option):
#     decoderType = DecoderType.BestPath
#     print(open(FilePaths.fnAccuracy).read())
#     model = Model(open(FilePaths.fnCharList).read(),
#                   decoderType, mustRestore=False)
#     recognized = infer(model, path)

#     return recognized


if __name__ == '__main__':
    main()
