"""
This script contains auxiliary functions that save models, plot info, compare sequences and recreate phrases from sequences
Text Based Information Retrieval, KULeuven 2018
Matija Kljun r0725870 
Tomás Pereira r0725869
"""


# Dependencies
import time
import matplotlib.pyplot as plt
import numpy as np
import wups
from keras.utils import plot_model


def model_save(model, model_name="", visual=False, epochs=0, path='models/', structure=False):
    if model_name == "":
        model_name = time.strftime("%d-%m-%Y-%H:%M:%S", time.gmtime()) + '_model_'
        if visual:
            model_name = model_name + "visual_"
        else:
            model_name = model_name + "textual_"
        if epochs != 0:
            model_name = model_name + "ep_" + str(epochs)
    model.save(path + model_name +'.h5')
    print("Model: " + model_name + " saved")

    # Save model structure
    if structure:
        plot_model(model, to_file=(model_name+'.png'), show_shapes=True, show_layer_names=True)

def fit_plot(history):
    plt.ion()

    # summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()

    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid()

    plt.show(block=False)

def argmax_onehot(onehot):
    return np.array(list(map(lambda sentence: np.array(list(map(lambda oh_word: np.argmax(oh_word), sentence))), onehot)))

def compare_predictions_sets(pred_seq, y_seq, word_map, inv_map, do_wups=True):
    matched_words = 0
    correct_answers = 0
    wups_score = 0

    for i in range(pred_seq.shape[0]):
        y_set =  set()
        pred_set = set()
        for j in range(pred_seq.shape[1]):
            if pred_seq[i][j] not in {word_map['<go>'], word_map['<eos>'], 0}:
                pred_set.add(pred_seq[i][j])
            if y_seq[i][j] not in {word_map['<go>'], word_map['<eos>'], 0}:
                y_set.add(y_seq[i][j])
        matched_words += len(y_set & pred_set)
        if len(y_set & pred_set) == len(y_set):
            correct_answers += 1
        if do_wups:
            wups_score += wups.wup_measure_sequences(list(y_set), list(pred_set), inv_map)

    if do_wups:
        return  matched_words / y_seq.shape[0], correct_answers / y_seq.shape[0], wups_score / y_seq.shape[0]
    else:
        return matched_words / y_seq.shape[0], correct_answers / y_seq.shape[0]

def to_words_list(seq, inv_map):
    return list(map(lambda i: inv_map[i] if i != 0 else "", seq))


def reconstruct_batch(input_seq, max_len, word_map, inv_map, model):
    decoder_input = np.zeros(shape=(input_seq.shape))
    decoder_input[:, 0] = word_map['<go>']

    for i in range(1,max_len):
        print("predicting---char " + str(i))
        out = model.predict([input_seq, decoder_input])
        decoder_input[:, i] = out.argmax(axis=2)[:,i-1]

    decoded_sentences = []
    for i in range(decoder_input.shape[0]):
        decoded_sentences.append(to_words_list(decoder_input[i], inv_map))
    return decoded_sentences, decoder_input
