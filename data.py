"""
This script aggregates functions dedicated to data processing for model training and testing
Text Based Information Retrieval, KULeuven 2018
Matija Kljun r0725870 
Tomás Pereira r0725869
"""

# Dependencies
import re
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# Boolean inputs: reverse, tokens, question marks, post-pad, shift sequences, one-hot encoding

def read_data(file):
    """ Read question and answer pairs from designated file """

    f = open(file, mode='r')

    data = f.read().split('\n')

    # Pop trailing end
    while data[-1] == "":
        data.pop()

    q = data[::2]
    a = data[1::2]
    f.close()

    return q, a

def read_visual_dict(file):
    f = open(file, mode='r')

    data = f.read().split('\n')

    image_data = {}
    for d in data:
        d = d.split(',')
        image_data[d[0]] = np.array(d[1:])

    return image_data

def get_visual_data(file, questions):
    visual_dict = read_visual_dict(file)
    visual_data = []

    regex = re.compile('(image\d*)')

    for q in questions:
        m = regex.search(q)
        if m != None:
            visual_data.append(visual_dict[m.group()])

    return np.array(visual_data)

def process_data(q, a, reverse=True, tokens=True):
    """
    :return: processed data according to values associated with
    'rev' and 'tokens' parameters
    """

    # Output (answers) <GO> and <EOS> tokens = True
    if tokens:
        a_tokens = []
        for line in a:
            a_tokens.append('<go> ' + line + ' <eos>')
        a = a_tokens

        #q_tokens = []
        #for line in q:
        #    q_tokens.append('<go> ' + line + ' <eos>')
        #q = q_tokens

    # Input (questions) reverse = True
    if reverse:
        q_rev = []
        for line in q:
            items = line.split()
            q_rev.append(' '.join(items[::-1]))
        q = q_rev

    return q, a


def build_vocab(data, qmark=True):
    """ Builds the vocabulary from the read data and formats it with
    or without question marks """

    # With data = total q + a
    # Check necessity of <UNK> tokens

    # Tokenizer
    # Learning - t.word_counts, t.document_count, t.word_index, t.word_docs

    # Choose whether question marks are filtered out when building the vocabulary
    if qmark:
        t = Tokenizer(lower=True, filters='!"#$%&()*+,-./:;=@[\]^_`{|}~?')
    else:
        t = Tokenizer(lower=True, filters='!"#$%&()*+,-./:;=@[\]^_`{|}~')
    t.fit_on_texts(data)  # Was q_train only
    vocab_size = len(
        t.word_index) + 1  # The integer for the largest encoded word needs to be specified as an array index

    # Integer encode documents (binary) - Fixed size array for direct network input
    # encoded_docs = t.texts_to_matrix(q_train)

    return t, vocab_size


def shift_sequences(seq_array):
    """ Shifts sequences to the right """

    # Sequence based on t.word_index for embedding input
    shifted_seq = []
    for sequence in seq_array:
        shifted_seq.append(sequence[1:])  # [:-1]
    return shifted_seq

def prepare_data(data, t, pad_lenght, vocab_size, postpad=True, shift=False):
    data_seq = t.texts_to_sequences(data)

    if not pad_lenght:
        pad_lenght = len(max(data_seq, key=len))

    if postpad:
        x = pad_sequences(data_seq, maxlen=pad_lenght, padding='post')
    else:
        x = pad_sequences(data_seq, maxlen=pad_lenght)

    if shift:
        if postpad:
            x_onehot = to_categorical(pad_sequences(shift_sequences(data_seq), maxlen=pad_lenght, padding='post'),
                                      num_classes=vocab_size)
        else:
            x_onehot = to_categorical(pad_sequences(shift_sequences(data_seq), maxlen=pad_lenght),
                                      num_classes=vocab_size)
    else:
        x_onehot = to_categorical(x, num_classes=vocab_size)

    return data_seq, x, x_onehot, pad_lenght

""" DEPRECATED 
def sequence_gen(q, a, t, max_len, vocab_size, postpad=True, shift=False):

    q_seq = t.texts_to_sequences(q)
    a_seq = t.texts_to_sequences(a)
    full_seq = q_seq + a_seq

    if not max_len:
        max_len = len(max(full_seq, key=len))

    x = pad_sequences(q_seq, maxlen=max_len)
    if postpad:
        y = pad_sequences(a_seq, maxlen=max_len, padding='post')
        y_to_onehot = pad_sequences(shift_sequences(a_seq), maxlen=max_len, padding='post')
    else:
        y = pad_sequences(a_seq, maxlen=max_len)
        y_to_onehot = pad_sequences(shift_sequences(a_seq), maxlen=max_len)

    if shift:
        x_onehot = to_categorical(x, num_classes=vocab_size)
        y_onehot = to_categorical(y_to_onehot, num_classes=vocab_size)
    else:
        x_onehot = to_categorical(x, num_classes=vocab_size)
        y_onehot = to_categorical(y, num_classes=vocab_size)

    return q_seq, a_seq, x, y, x_onehot, y_onehot, max_len
"""
