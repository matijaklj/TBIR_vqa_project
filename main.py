"""
This is the main script for the project. It enables the user to switch between question representation and question
answering with or without visual features. It also allows the user to train new network models or to use existing ones
Text Based Information Retrieval, KULeuven 2018
Matija Kljun r0725870
Tomás Pereira r0725869
"""

from future.moves import pickle
from keras.utils.vis_utils import plot_model

import util
from data import *
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, concatenate
from keras.layers.core import Dense, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping

# Script params
QA = True  # Choose between question answering (True) or question reconstruction (False)
VISUAL = True # Choose between using visual features for qa (True) or only texutal (False)
TRAIN = False # Choose between training the model (True) or loading the model from file (False)

# Network params
LATENT_DIM = 512
EPOCHS = 30
VERBOSE = False

# Get questions and answers from train and test sets
q_raw, a_raw = read_data('data/qa.894.raw.train.txt')
#q_test_raw, a_test_raw = read_data('data/qa.894.raw.test.txt')
q_test_raw, a_test_raw = read_data('data/new_questions.txt')

# Data processing
if QA:  # Question answering
    # train data
    q, a = process_data(q_raw, a_raw)
    t, vocab_size = build_vocab(q + a)
    q_seq, x, x_onehot, max_len = prepare_data(q, t, [], vocab_size, postpad=False, shift=False)
    a_seq, y, y_onehot, max_len_ans = prepare_data(a, t, [], vocab_size, postpad=True, shift=True)

    # test data
    q_test, a_test = process_data(q_test_raw, a_test_raw)
    q_test_seq, x_test, x_test_onehot, _ = prepare_data(q_test, t, max_len, vocab_size, postpad=False, shift=False)
    a_test_seq, y_test, y_test_onehot, _ = prepare_data(a_test, t, max_len_ans, vocab_size, postpad=True, shift=True)
else:  # Question representation
    # train data
    q, q_tar = process_data(q_raw, q_raw, reverse=False)
    t, vocab_size = build_vocab(q+q_tar)
    q_seq, x, x_onehot, max_len = prepare_data(q, t, [], vocab_size, postpad=False, shift=False)
    a_seq, y, y_onehot, max_len_ans = prepare_data(q_tar, t, [], vocab_size, postpad=True, shift=True)

    # test data
    q_test, q_tar_test = process_data(q_test_raw, q_test_raw, reverse=False)
    q_test_seq, x_test, x_test_onehot, _ = prepare_data(q_test, t, max_len, vocab_size, postpad=False, shift=False)
    a_test_seq, y_test, y_test_onehot, _ = prepare_data(q_tar_test, t, max_len_ans, vocab_size, postpad=True, shift=True)

# Word mapping
inv_map = {v: k for k, v in t.word_index.items()}
word_map = {k: v for k, v in t.word_index.items()}

# Get visual train and test data
visual_data = get_visual_data('data/img_features.csv', q_raw)
visual_data_test = get_visual_data('data/img_features.csv', q_test_raw)

# Network model builder
def define_model(latent_dim, vocab_size, max_len, max_len_out, visual, qa, verbose=False):

    if qa:
        if visual:
            question_inputs = Input(shape=(max_len,), name="encoder_input")
            encoder_inputs = Embedding(input_dim=vocab_size, output_dim=latent_dim, mask_zero=False)(question_inputs)

            _, h, c = LSTM(latent_dim, return_sequences=False, return_state=True, name="encoder_lstm", dropout=0.4)(encoder_inputs)

            answers_inputs = Input(shape=(max_len_out,), name="decoder_input")
            decoder_inputs = Embedding(input_dim=vocab_size, output_dim=latent_dim, mask_zero=False)(answers_inputs)

            decoder_lstm = LSTM(latent_dim, return_sequences=True, name="decoder_lstm", dropout=0.4)
            decoder_output = decoder_lstm(decoder_inputs, initial_state=[h,c])

            question_dense = Dense(1024, activation='tanh', name="dense_1")(decoder_output)

            image_inputs = Input(shape=(2048,), name="image_input")
            image_dense = Dense(1024, activation="tanh", name="image_dense")(image_inputs)
            image_repeat = RepeatVector(n=max_len_out)(image_dense)

            con = concatenate([question_dense, image_repeat])

            mul_dense = Dense(1024, activation='tanh', name="dense_out")(con)

            output = Dense(vocab_size, activation='softmax', name="dense_softmax")(mul_dense)

            model = Model([question_inputs, answers_inputs, image_inputs], output)

            model.summary()
        else:
            encoder_input = Input(shape=(max_len,), name="encoder_input")
            encoder_inputs = Embedding(input_dim=vocab_size, output_dim=latent_dim, mask_zero=False)(
                encoder_input)

            _, state_h, state_c = LSTM(latent_dim, return_state=True, name="encoder_3_lstm", dropout=0.4)(
                encoder_inputs)
            encoder_states = [state_h, state_c]

            decoder_input = Input(shape=(max_len_out,), name="decoder_input")
            decoder_inputs = Embedding(input_dim=vocab_size, output_dim=latent_dim, mask_zero=False)(
                decoder_input)

            decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm", dropout=0.4)
            decoder_output, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

            decoder_dense = Dense(vocab_size, activation='softmax')
            output = decoder_dense(decoder_output)

            model = Model([encoder_input, decoder_input], output)

    else:
        encoder_inputs = Input(shape=(None,), name="encoder_input")
        embedded_encoder_input = Embedding(input_dim=vocab_size, output_dim=latent_dim, mask_zero=False)(encoder_inputs)
        _, state_h, state_c = LSTM(latent_dim, return_state=True, name="encoder_lstm", dropout=0.4)(embedded_encoder_input)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,), name="decoder_input")
        embedded_decoder_input = Embedding(input_dim=vocab_size, output_dim=latent_dim, name="decoder_embedding")(
            decoder_inputs)

        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm", dropout=0.4)
        decoder_output, _, _ = decoder_lstm(embedded_decoder_input, initial_state=encoder_states)

        decoder_dense = Dense(vocab_size, activation='softmax')
        output = decoder_dense(decoder_output)

        model = Model([encoder_inputs, decoder_inputs], output)

    if verbose: model.summary()

    return model

if TRAIN:
    model = define_model(LATENT_DIM, vocab_size, max_len, max_len_ans, VISUAL, QA, VERBOSE)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=4)

    if QA:
        if VISUAL:
            history = model.fit([x, y, visual_data], y_onehot,
                                epochs=EPOCHS,
                                validation_split=0.2,
                                callbacks=[early_stopping])
        else:
            history = model.fit([x, y], y_onehot,
                                epochs=EPOCHS,
                                validation_split=0.2,
                                callbacks=[early_stopping])
    else:
        history = model.fit([x, y], y_onehot,
                            epochs=EPOCHS,
                            validation_split=0.2,
                            callbacks=[early_stopping])

    util.model_save(model, model_name="model_new_name", path='models/')

    if VERBOSE:
        try:
            util.fit_plot(history)
        except:
            print("fit_plot function failed")

        try:
            plot_model(model)
        except:
            print("plot_model function failed")


else:
    model = load_model('models/' + 'model_visual_qa' + '.h5') # q reconstruction model: model_q_rec

# test accuracy and wups score
if QA:
    if TRAIN:
        # test accuracy
        if VISUAL:
            start_input = np.zeros(y.shape)
            start_input[:, 0] = word_map['<go>']
            pred = model.predict([x, start_input, visual_data])
        else:
            start_input = np.zeros(y.shape)
            start_input[:, 0] = word_map['<go>']
            pred = model.predict([x, start_input])

        print()
        print("Train q/a accuracy")
        word_acc, acc, wups_score = util.compare_predictions_sets(util.argmax_onehot(pred),
                                                                  util.argmax_onehot(y_onehot), word_map, inv_map)
        print()
        print("Train word accuracy: " + str(word_acc))
        print("Train accuracy: " + str(acc))
        print("Test WUPS: " + str(wups_score))

    # train accuracy
    if VISUAL:
        start_input = np.zeros(y_test.shape)
        start_input[:, 0] = word_map['<go>']
        pred = model.predict([x_test, start_input, visual_data_test])
    else:
        start_input = np.zeros(y_test.shape)
        start_input[:,0] = word_map['<go>']
        pred = model.predict([x_test, start_input])

    print()
    print("Predictions for test set with %d questions" % len(q_test_raw))
    test_word_acc, test_acc, test_wups_score = util.compare_predictions_sets(util.argmax_onehot(pred), util.argmax_onehot(y_test_onehot), word_map, inv_map)
    print()
    print("Test word accuracy: " + str(test_word_acc))
    print("Test accuracy: " + str(test_acc))
    print("Test WUPS: " + str(test_wups_score))

    print()
    print("Predictions for test set with %d questions" % len(q_test_raw))
    print("[# Question raw]")
    print("[# Answer raw]")
    print("[# Answer prediction]")
    print()

    for i in range(len(q_test_raw)):  # print sentence reconstructions
        print(("[%d]: " + q_test_raw[i]) % i)
        print(("[%d]: " + a_test_raw[i]) % i)

        count = 0
        for p in pred[i]:
            key = np.argmax(p)
            if key not in [0, 3, 4]:
                if count == 0:
                    print(("[%d]: " + inv_map[key]) % i, end='')
                    count += 1
                else:
                    print(", " + inv_map[key], end='')

        print()
        print()

else:
    print("Question reconstruction")

    if TRAIN:
        print()
        print("Train accuracy")
        pred, pred_seq = util.reconstruct_batch(x, max_len, word_map, inv_map, model)
        word_acc, acc = util.compare_predictions_sets(pred_seq, x, word_map, inv_map, do_wups=False)
        print()
        print("Train word accuracy: " + str(word_acc))
        print("Train accuracy: " + str(acc))

    print()
    print("Test accuracy")
    pred_test, pred_test_seq = util.reconstruct_batch(x_test, max_len, word_map, inv_map, model)
    test_word_acc, test_acc = util.compare_predictions_sets(pred_test_seq, x_test, word_map, inv_map, do_wups=False)
    print()
    print("Test word accuracy: " + str(test_word_acc))
    print("Test accuracy: " + str(test_acc))

    print()
    print("Predictions for test set with %d questions" % len(q_test_raw))
    print("[# Raw]")
    print("[# Prediction]")
    print()
    for i in range(len(q_test_raw)):  # print sentence reconstructions
        print(("[%d]: " + q_test_raw[i]) % i)

        prediction = list(filter(None, pred_test[i]))
        print(("[%d]: " + " ".join(prediction[1:-1]) + " ?") % i)
        print()

