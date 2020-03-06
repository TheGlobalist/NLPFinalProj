import numpy as np
from typing import List, Dict, Tuple
from tokenizer import FullTokenizer


def convert_sentence_to_features(sentence, tokenizer, max_seq_len):
    tokens = ['[CLS]']
    tokens += tokenizer.tokenize(sentence)
    tokens = tokens[:max_seq_len - 1] if len(tokens) > max_seq_len -1 else tokens
    tokens.append('[SEP]')

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    zero_mask = [0] * (max_seq_len - len(tokens))
    input_ids += zero_mask
    input_mask += zero_mask
    segment_ids += zero_mask

    return input_ids, input_mask, segment_ids


def strategy_for_rebuilding_word(sentence:str, output: np.ndarray, tokenizatore: FullTokenizer):
    tokens = ['CLS'] + tokenizatore.tokenize(sentence) + ['SEP']
    check_if_subwords = [i for i, x in enumerate(sentence) if "#" in x]
    effective_list_of_indexes = check_if_subwords.copy()
    if check_if_subwords:
        for index in check_if_subwords:
            first_piece = tokens[index]
            if '#' in first_piece:
                word = first_piece[2:]
                decreaser = 1
                while True:
                    next_piece = tokens[index-decreaser]
                    word = (next_piece if '#' not in next_piece else next_piece[2:]) + word
                    if '#' in next_piece:
                        effective_list_of_indexes.append(index-decreaser)
                        decreaser += 1
                    else:
                        break







def convert_sentences_to_features(sentences, tokenizer, max_seq_len=64):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)

    return np.asarray(all_input_ids), np.asarray(all_input_mask), np.asarray(all_segment_ids)


def convert_sentence_to_features_with_no_padding(sentence, tokenizer) -> Tuple:
    """
    Works the same as the convert_sentences_to_features() function, but it has to be used for creating a sentence to be predicted by the network
    :param sentence: the sentence to convert
    :param tokenizer: the BERTs's tokenizer object to use
    :return: the converted sentence
    """
    tokens = ['[CLS]']
    tokens += tokenizer.tokenize(sentence)
    tokens.append('[SEP]')

    segment_ids = [0] * len(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    return input_ids, input_mask, segment_ids

def convert_sentence_to_features_no_padding(sentence, tokenizer):
    input_ids, input_mask, segment_ids = convert_sentence_to_features_with_no_padding(sentence, tokenizer)

    return np.asarray([input_ids]), np.asarray([input_mask]), np.asarray([segment_ids])

def convert_y(label, label_bn, label_wndmn, label_lex, max_seq_len:int = 64):
    label_bn_to_return = []
    label_wndmn_to_return = []
    label_lex_to_return = []
    for sent_arr in label:
        sent1 = __consult_dict(sent_arr[0],label_wndmn, max_seq_len)
        sent2 = __consult_dict(sent_arr[1], label_bn, max_seq_len)
        sent3 = __consult_dict(sent_arr[2], label_lex,max_seq_len)
        label_wndmn_to_return.append(sent1)
        label_bn_to_return.append(sent2)
        label_lex_to_return.append(sent3)
    return np.asarray(label_bn_to_return), np.array(label_wndmn_to_return), np.array(label_lex_to_return)



def __consult_dict(sentence: str, vocab: Dict, max_seq_len: int) -> List:
    toReturn = []
    for parola in sentence.split(" "):
        to_use = None
        try:
            to_use = int(vocab[parola][0])
        except:
            to_use = int(vocab['<UNK>'][0])
        toReturn.append(to_use)
    if len(toReturn) > max_seq_len - 1:
        toReturn = toReturn[:max_seq_len - 1]
    zero_mask = [0] * (max_seq_len - len(toReturn))
    toReturn.extend(zero_mask)
    return toReturn
