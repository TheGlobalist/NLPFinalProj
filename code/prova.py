import numpy as np
from typing import List, Dict


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


def convert_sentences_to_features(sentences, tokenizer, max_seq_len=128):
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []

    for sentence in sentences:
        input_ids, input_mask, segment_ids = convert_sentence_to_features(sentence, tokenizer, max_seq_len)
        all_input_ids.append(input_ids)
        all_input_mask.append(input_mask)
        all_segment_ids.append(segment_ids)

    return np.asarray(all_input_ids), np.asarray(all_input_mask), np.asarray(all_segment_ids)


def convert_y(label, label_bn, label_wndmn, label_lex, max_seq_len:int = 128):
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
            to_use = vocab[parola][0]
        except:
            to_use = vocab['<UNK>'][0]
        toReturn.append(to_use)
    if not toReturn:
        print("caio")
    if len(toReturn) > max_seq_len - 1:
        toReturn = toReturn[:max_seq_len - 1]
    zero_mask = [0] * (max_seq_len - len(toReturn))
    toReturn.extend(zero_mask)
    return toReturn