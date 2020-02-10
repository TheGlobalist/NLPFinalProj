import tensorflow as tf
from tensorflow import keras as keras
from typing import List
import tensorflow_hub as hub


class BERtLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        """
        The class's constructor that will host BERt's representation
        :param kwargs: eventual input parameters
        """
        self.trainable = True
        super(BERtLayer, self).__init__(**kwargs)
        #TODO rimuovere questa chiamata?
        #self.build(1)

    def build(self, input_shape):
        """
        Builds the actual Layer
        :param input_shape: the shape of the Layer
        """
        self.bert = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1", trainable=self.trainable)
        super(BERtLayer, self).build(input_shape)

    def call(self,input_shape):
        """
        Builds the representation of the model via Keras's Functional API
        :param input_shape: the shape of the Layer
        :return: the actual (expected) output
        """
        input_word_ids = tf.keras.layers.Input(shape=(input_shape,), dtype=tf.int32,name = "input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(input_shape,), dtype=tf.int32,name = "input_mask")
        segment_ids = tf.keras.layers.Input(shape=(input_shape,), dtype=tf.int32,name = "segment_ids")
        pooled_output, sequence_output = self.bert([input_word_ids, input_mask, segment_ids])
        return pooled_output


class WSD:
    def __init__(self,hidden_size: int = 256, input_length: int = None,dropout: float = 0.0,recurrent_dropout: float = 0.0,
                 learning_rate: float = None,vocab_size: int = None,outputs_size: List = None):
        input_word_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name="segment_ids")
        BERt = BERtLayer()([input_word_ids, input_mask, segment_ids])
        LSTM = keras.layers.Bidirectional(
        keras.layers.LSTM(
            units=hidden_size,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True,
            return_state=True,
        ))(BERt)
        lstm_a = self.attention_layer(LSTM)

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

