import tensorflow as tf
from typing import List
import tensorflow_hub as hub
from typing import Dict, List
import os
from AttentionLayer import MyLayer
import time
from prova import convert_sentences_to_features, convert_y
from tokenizer import FullTokenizer


class WSD:
    def __init__(self,bert_path: str, outputs_size: List, hidden_size: int = 256, dropout: float = 0.1, recurrent_dropout: float = 0.1,learning_rate: float = 0.01):
        """
        This class encapsules a WSD system
        :param bert_path: the path to use for loading the BERT's dictionary
        :param outputs_size: a list that represent how much big should the output units be
        :param hidden_size: default to 256. Represent how many hidden units should be in the network
        :param dropout: default to 0.1. The default value for dropout
        :param recurrent_dropout: default to 0.1. The default value for recurrent dropout.
        :param learning_rate: default to 0.01. The default value for the learning rate.
        """
        self.tokenizatore = FullTokenizer(bert_path,do_lower_case=False)
        input_word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="segment_ids")
        print("dopwnloading BERT...")
        bert = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1", trainable=False)
        print("BERT downloaded")
        pooled_output, sequence_output = bert([input_word_ids, input_mask, segment_ids])
        #self.vocab_file = bert.resolved_object.vocab_file.asset_path.numpy()
        #self.do_lower_case = bert.resolved_object.do_lower_case.numpy()
        LSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                units=hidden_size,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                return_sequences=True,
                return_state=True
            )
        )(sequence_output)
        LSTM = self.produce_attention_layer(LSTM)
        LSTM = tf.keras.layers.BatchNormalization()(LSTM)
        LSTM = tf.keras.layers.Dropout(0.5)(LSTM)

        #We need to perform multitask learning, so we need 3 outputs...
        babelnet_output = tf.keras.layers.Dense(outputs_size[0], activation="softmax", name="babelnet")(LSTM)
        domain_output = tf.keras.layers.Dense(outputs_size[1], activation="softmax", name="domain")(LSTM)
        lexicon_output = tf.keras.layers.Dense(outputs_size[2], activation="softmax", name="lexicon")(LSTM)
        #Usage of AdamOptimizer in order to have a better train results
        optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
        self.model = tf.keras.models.Model(inputs=[input_word_ids,input_mask, segment_ids], outputs=[babelnet_output,domain_output,lexicon_output])

        self.model.compile(
            loss="sparse_categorical_crossentropy", optimizer=optimizer, experimental_run_tf_function=False
        )

        self.model.summary()

    def train(self, train_data, label, vocab_label_bn: Dict, vocab_label_wndmn:Dict , vocab_label_lex: Dict, train_dev: Dict, label_dev: Dict):
        """
        Trains the model
        :param train_data: the features that will be fed to the model for training
        :param label: the truth values of the features
        :param vocab_label_bn: the vocabulary of the first task to perform
        :param vocab_label_wndmn: the vocabulary of the second task to perform
        :param vocab_label_lex: the vocabulary of the third task to perform
        :param train_dev: the features that will be used per validation purpopes
        :param label_dev: the truth values that will be used per validation purpopes
        :return: returns the keras.history object of the train
        """
        early_stopper = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, mode="min", verbose=1, restore_best_weights=True
        )

        if not os.path.exists("../resources/saved_model"):
            os.makedirs("../saved_model")
        path_to_checkpoint = "../resources/saved_model/model_{epoch:02d}_{val_loss:.2f}.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            path_to_checkpoint,
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode="min",
        )
        print("enter in train...")
        train_1, train_2, train_3 = convert_sentences_to_features(train_data, self.tokenizatore, max_seq_len=64)
        train_dev_conv_1, train_dev_conv_2, train_dev_conv_3 = convert_sentences_to_features(train_dev, self.tokenizatore,
                                                                                             max_seq_len=64)
        print("train_1", train_1.shape)
        print("train_2", train_2.shape)
        print("train_3", train_3.shape)
        print("Done train preparation...")
        # train_dev_1,train_dev_2,train_dev_3 = convert_sentences_to_features(train_dev, tokenizatore, max_seq_len=20)


        label_bn_conv, label_wndmn_conv, label_lex_conv = convert_y(label, vocab_label_bn, vocab_label_wndmn,
                                                                    vocab_label_lex)
        label_bn_dev_conv, label_wndmn_dev_conv, label_lex_dev_conv = convert_y(label_dev, vocab_label_bn,
                                                                                vocab_label_wndmn, vocab_label_lex)
        print("Done label preparatiomn")

        print("label_bn_conv",label_bn_conv.shape)
        print("label_wndmn_conv", label_wndmn_conv.shape)
        print("label_lex_conv", label_lex_conv.shape)

        print("ciao")

        start = time.process_time()
        self.history = self.model.fit(
            x={'input_word_ids': train_1, 'input_mask': train_2, 'segment_ids': train_3},
            y={'babelnet': label_bn_conv, 'domain': label_wndmn_conv, 'lexicon': label_lex_conv},
            epochs=20,
            batch_size=64,
            verbose=1,
            validation_data=([train_dev_conv_1, train_dev_conv_2, train_dev_conv_3],
                             [label_bn_dev_conv, label_wndmn_dev_conv, label_lex_dev_conv]),
            callbacks=[checkpoint, early_stopper],
        )
        return self.history

    def produce_attention_layer(self, LSTM):
        """
        Produces an Attention Layer like the one mentioned in the Raganato et al. Neural Sequence Learning Models for Word Sense Disambiguation,
        chapter 3.2
        :param lstm: The LSTM that will be used in the task
        :return: The LSTM that was previously given in input with the enhancement of the Attention Layer
        """
        conc1 = tf.keras.layers.Concatenate()([LSTM[1], LSTM[3]])
        conc2 = tf.keras.layers.Concatenate()([LSTM[2], LSTM[4]])
        hidden_states = tf.keras.layers.Multiply()([conc1, conc2])
        u = tf.keras.layers.Dense(1, activation="tanh")(hidden_states)
        a = tf.keras.layers.Activation("softmax")(u)
        context_vector = tf.keras.layers.Multiply()([LSTM[0], a])
        print(context_vector.shape)
        to_return = tf.keras.layers.Multiply()([LSTM[0], context_vector])
        print(to_return.shape)
        return to_return



if __name__ == "__main__":
    from data_preprocessing import load_dataset, load_gold_key_file, create_mapping_dictionary
    train,etree_file = load_dataset("../dataset/SemCor/semcor.data.xml")
    label = load_gold_key_file("../dataset/SemCor/semcor.gold.key.txt", etree_file)
    train = [dato for dato in train if dato and dato]
    vocab_train = create_mapping_dictionary("../resources",data = None)
    vocab_label_bn = create_mapping_dictionary("../resources", data = None, mode='bn')
    vocab_label_wndmn = create_mapping_dictionary("../resources", data = None, mode='wndmn')
    vocab_label_lex = create_mapping_dictionary("../resources", data = None, mode='lex')
    modello = WSD("../resources/vocabularies/bert_vocab.txt",[len(vocab_label_bn),len(vocab_label_wndmn), len(vocab_label_lex)],dropout = 0.5, recurrent_dropout = 0.1,learning_rate = 0.0003)
    train_dev,etree_file_dev = load_dataset("../dataset/dev/semeval2007.data.xml")
    label_dev = load_gold_key_file("../dataset/dev/semeval2007.gold.key.txt", etree_file_dev)
    modello.train(train,label,vocab_label_bn,vocab_label_wndmn,vocab_label_lex, train_dev, label_dev)
    print("ciao")

