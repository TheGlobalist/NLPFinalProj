import tensorflow as tf
from typing import List
import tensorflow_hub as hub
from pathlib import Path
from typing import Dict, List, Set
from tokenizer import FullTokenizer
import os
import time
import matplotlib.pyplot as plt
from prova import convert_sentences_to_features, convert_y
from sklearn.model_selection import train_test_split
from data_preprocessing import load_dataset
from tokenizer import FullTokenizer


class WSD:
    def __init__(self,hidden_size: int = 256, dropout: float = 0.0, recurrent_dropout: float = 0.0,
                 learning_rate: float = 0.01, outputs_size: List = None):
        input_word_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="segment_ids")
        #BERt = BERtLayer()([input_word_ids, input_mask, segment_ids])
        bert = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1", trainable=True)
        pooled_output, sequence_output = bert([input_word_ids, input_mask, segment_ids])
        #self.vocab_file = bert.resolved_object.vocab_file.asset_path.numpy()
        #self.do_lower_case = bert.resolved_object.do_lower_case.numpy()
        LSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=hidden_size,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True,
                return_state=False
            )
        )(sequence_output)
        #lstm_attention = self.attention_layer(LSTM)
        #We need to perform multitask learning, so we need 3 outputs...
        babelnet_output = tf.keras.layers.Dense(outputs_size[0], activation="softmax", name="babelnet")(LSTM)
        domain_output = tf.keras.layers.Dense(outputs_size[1], activation="softmax", name="domain")(LSTM)
        lexicon_output = tf.keras.layers.Dense(outputs_size[2], activation="softmax", name="lexicon")(LSTM)
        #Usage of AdamOptimizer in order to have a better train results
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.model = tf.keras.models.Model(inputs=[input_word_ids,input_mask, segment_ids], outputs=[babelnet_output,domain_output,lexicon_output])
        self.model.compile(
            loss="sparse_categorical_crossentropy", optimizer=optimizer
        )
        self.model.summary()

    def train(self, train_data, label, vocab_label_bn: Dict, vocab_label_wndmn:Dict , vocab_label_lex: Dict, train_dev: Dict, label_dev: Dict):
        """
        WIP...
        :return:
        """
        early_stopper = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, mode="min", verbose=1, restore_best_weights=True
        )

        if not os.path.exists("../saved_model"):
            os.makedirs("../saved_model")
        path_to_checkpoint = "../saved_model/model_{epoch:02d}_{val_loss:.2f}.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            path_to_checkpoint,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        )
        print("enter in train...")
        tokenizatore = FullTokenizer('/var/folders/rw/1qf5yn6960jbwyvc1rmlrcn40000gn/T/tfhub_modules/a7f4eb577e5eeec24c73b9dace49639b7c8193ed/assets/vocab.txt',do_lower_case=False)

        label_bn_conv, label_wndmn_conv, label_lex_conv = convert_y(label, vocab_label_bn, vocab_label_wndmn, vocab_label_lex)


        train_1,train_2,train_3 = convert_sentences_to_features(train_data, tokenizatore, max_seq_len=128)
        print("Done train preparation...")
        #train_dev_1,train_dev_2,train_dev_3 = convert_sentences_to_features(train_dev, tokenizatore, max_seq_len=20)

        print("Done label preparatiomn")


        print("ciao")

        start = time.process_time()
        history = self.model.fit(
            x={'input_word_ids': train_1,'input_mask': train_2, 'segment_ids':train_3},
            y= {'babelnet': label_bn_conv,'domain': label_wndmn_conv, 'lexicon': label_lex_conv},
            epochs=4,
            batch_size=128,
            verbose=1,
            validation_split=0.2,
            callbacks = [checkpoint, early_stopper],
        )
        end = time.process_time()
        print("Done in " + str(end-time))






    def attention_layer(self, lstm):
        """
        Produces an Attention Layer like the one mentioned in the Raganato et al. Neural Sequence Learning Models for Word Sense Disambiguation,
        chapter 3.2
        :param lstm: The LSTM that will be used in the task
        :return: The LSTM that was previously given in input with the enhancement of the Attention Layer
        """
        hidden_state = tf.keras.layers.Concatenate()([lstm[1], lstm[3]]) #Layer that concatenates a list of inputs.
        hidden_state = tf.keras.layers.RepeatVector(tf.keras.backend.shape(lstm[0])[1])(hidden_state)
        u = tf.keras.layers.Dense(1, activation="tanh")(hidden_state)
        a = tf.keras.layers.Activation("softmax")(u)
        context_vector = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x[0] * x[1], axis=1))([lstm[0], a])
        return tf.keras.layers.Multiply()([lstm[0], context_vector])


if __name__ == "__main__":
    from data_preprocessing import load_dataset, calculate_train_output_size
    train, label = load_dataset("../dataset/SemCor/semcor.data.xml", "../dataset/SemCor/semcor.gold.key.txt")
    train = [dato for dato in train if dato and dato]
    vocab_train = calculate_train_output_size(train)
    vocab_label_bn = calculate_train_output_size(label, mode='bn')
    vocab_label_wndmn = calculate_train_output_size(label, mode='wndmn')
    vocab_label_lex = calculate_train_output_size(label, mode='lex')
    modello = WSD(dropout = 0.0001, recurrent_dropout  =  0.0001,
                  outputs_size = [len(vocab_label_bn),len(vocab_label_wndmn), len(vocab_label_lex)])
    train_dev, label_dev = load_dataset("../dataset/dev/semeval2007.data.xml", "../dataset/dev/semeval2007.gold.key.txt", dev_parsing=True)
    modello.train(train,label,vocab_label_bn,vocab_label_wndmn,vocab_label_lex, train_dev, label_dev)
    print("ciao")

