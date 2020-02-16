import tensorflow as tf
from typing import List
import tensorflow_hub as hub
from tokenizer import FullTokenizer
import os
import time
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
        self.vocab_file = bert.resolved_object.vocab_file.asset_path.numpy()
        self.do_lower_case = bert.resolved_object.do_lower_case.numpy()
        LSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=hidden_size,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True,
                return_state=True,
            )
        )(sequence_output)
        lstm_attention = self.attention_layer(LSTM)
        #We need to perform multitask learning, so we need 3 outputs...
        babelnet = tf.keras.layers.Dense(outputs_size[0], activation="softmax", name="bn")(lstm_attention)
        domain = tf.keras.layers.Dense(outputs_size[1], activation="softmax", name="dom")(lstm_attention)
        lexicus = tf.keras.layers.Dense(outputs_size[2], activation="softmax", name="lex")(lstm_attention)
        #Usage of AdamOptimizer in order to have a better train results
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        sess = tf.keras.backend.get_session()
        init = tf.global_variables_initializer()
        sess.run(init)
        self.model = tf.keras.models.Model(inputs=[input_word_ids,input_mask, segment_ids], outputs=[babelnet,domain,lexicus])
        self.model.compile(
            loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["acc"]
        )
        self.model.summary()

    def train(self):
        """
        WIP...
        :return:
        """
        early_stopper = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, mode="min", verbose=1, restore_best_weights=True
        )

        if not os.path.exists("../saved_model"):
            os.makedirs("../saved_model")
        cp_path = "../saved_model/model_{epoch:02d}_{val_loss:.2f}.h5"
        cp = tf.keras.callbacks.ModelCheckpoint(
            cp_path,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        )

        train_data, label_data = load_dataset("../dataset/SemCor/semcor.data.xml","../dataset/SemCor/semcor.gold.key.txt")

        train_data_es, label_data_es = load_dataset("../dataset/eval_dataset/semeval2015.es.data.xml","../dataset/eval_dataset/semeval2015.es.gold.key.txt")

        tokenizatore = FullTokenizer(self.vocab_file,do_lower_case=self.do_lower_case)
        test = tokenizatore.tokenize(train_data[0])
        print("ciao")



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
    testing = WSD()
    testing.train()
    print("ciao")

