"""
This script loads pre-saved model fit param. data and plots acc, val_acc, loss, val_loss vs epochs
Text Based Information Retrieval, KULeuven 2018
Matija Kljun r0725870 
Tomás Pereira r0725869
"""

# Dependencies
import matplotlib.pyplot as plt
import pickle

# Imports model fit param data into history (acc, val_acc, loss, val_loss)
with open('modelHistory', 'rb') as handle:
    history = pickle.loads(handle.read())

def fit_plot(history):
    plt.ion()

    # summarize history for accuracy
    plt.figure()
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()

    # summarize history for loss
    plt.figure()
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()

    plt.show(block=False)

fit_plot(history)
