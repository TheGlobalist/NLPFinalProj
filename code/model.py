import tensorflow as tf
from tensorflow import keras as keras
from typing import List
import tensorflow_hub as hub
from tokenizer import FullTokenizer
import os


def convert_sentence_to_features(sentence, tokenizer, max_seq_len):
    tokens = ['[CLS]']
    tokens.extend(tokenizer.tokenize(sentence))
    if len(tokens) > max_seq_len - 1:
        tokens = tokens[:max_seq_len - 1]
    tokens.append('[SEP]')

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    # Zero Mask till seq_length
    zero_mask = [0] * (max_seq_len - len(tokens))
    input_ids.extend(zero_mask)
    input_mask.extend(zero_mask)
    segment_ids.extend(zero_mask)

    return input_ids, input_mask, segment_ids


def convert_sentences_to_features(sentences, tokenizer, max_seq_len=20):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)

    return all_input_ids, all_input_mask, all_segment_ids

def create_tokenizer(vocab_file, do_lower_case=False):
    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)



class WSD:
    def __init__(self,hidden_size: int = 256, input_length: int = None,dropout: float = 0.0,recurrent_dropout: float = 0.0,
                 learning_rate: float = None,vocab_size: int = None,outputs_size: List = None):
        input_word_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="segment_ids")
        #BERt = BERtLayer()([input_word_ids, input_mask, segment_ids])
        bert = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1", trainable=True)
        LSTM = keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=hidden_size,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        ))(bert[1])
        lstm_attention = self.attention_layer(LSTM)
        #We need to perform multitask learning, so we need 3 outputs...
        babelnet = keras.layers.Dense(outputs_size[0], activation="softmax", name="bn")(lstm_attention)
        domain = keras.layers.Dense(outputs_size[1], activation="softmax", name="dom")(lstm_attention)
        lexicus = keras.layers.Dense(outputs_size[2], activation="softmax", name="lex")(lstm_attention)
        #Usage of AdamOptimizer in order to have a better train results
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        sess = keras.backend.get_session()
        init = tf.global_variables_initializer()
        sess.run(init)
        self.model = keras.models.Model(inputs=[input_word_ids,input_mask, segment_ids], outputs=[babelnet,domain,lexicus])
        self.model.compile(
            loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["acc"]
        )
        self.model.summary()


    def attention_layer(self, lstm):
        """
        Produces an Attention Layer like the one mentioned in the Raganato et al. Neural Sequence Learning Models for Word Sense Disambiguation,
        chapter 3.2
        :param lstm: The LSTM that will be used in the task
        :return: The LSTM that was previously given in input with the enhancement of the Attention Layer
        """
        hidden_state = keras.layers.Concatenate()([lstm[1], lstm[3]]) #Layer that concatenates a list of inputs.
        hidden_state = keras.layers.RepeatVector(keras.backend.shape(lstm[0])[1])(hidden_state)
        u = keras.layers.Dense(1, activation="tanh")(hidden_state)
        a = keras.layers.Activation("softmax")(u)
        context_vector = keras.layers.Lambda(lambda x: keras.backend.sum(x[0] * x[1], axis=1))([lstm[0], a])
        return keras.layers.Multiply()([lstm[0], context_vector])


if __name__ == "__main__":
    testing = WSD()
    print("ciao")

