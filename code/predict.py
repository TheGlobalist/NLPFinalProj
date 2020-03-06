from model import WSD
from data_preprocessing import load_dataset, create_mapping_dictionary, reload_word_mapping,get_bn2wn,get_bn2wndomains, get_bn2lex
from typing import List, Dict, Tuple
from prova import convert_sentence_to_features_no_padding
import numpy as np
import os
from nltk.corpus import wordnet


mfs_counter = 0


def predict_babelnet(input_path : str, output_path : str, resources_path : str) -> None:
    global mfs_counter
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <BABELSynset>" format (e.g. "d000.s000.t000 bn:01234567n").
    
    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.
    
    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    print(">>>> BABELNET PREDICTION")
    prediction_results, sentences_xml_elements = __predict(input_path,resources_path)
    vocab_label_bn = create_mapping_dictionary(resources_path, mode='bn')
    correctly_saved = 0
    filename = os.path.normpath(input_path)
    filename = filename.split(os.sep)[-1]
    filename = filename[:-3]+"babelnet.gold.key.txt"
    for index in range(len(prediction_results)):

        correctly_saved += __write_result(filename,
                                          sentences_xml_elements[index],
                                          resources_path, output_path,
                                          prediction_results[index][0][0],
                                          vocab=vocab_label_bn,
                                          enable_coarse_grained=1,
                                          vocab_for_coarse=None)

    print("Successfully saved {} out of {}".format(correctly_saved, len(prediction_results)))
    del prediction_results
    print("Of these, {} were MFS".format(mfs_counter))
    mfs_counter = 0
    return


def predict_wordnet_domains(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <wordnetDomain>" format (e.g. "d000.s000.t000 sport").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    global mfs_counter
    print(">>>> WORDNET DOMAINS  PREDICTION")
    prediction_results, sentences_xml_elements = __predict(input_path,resources_path)
    vocab_label_wndmn = create_mapping_dictionary(resources_path, mode='wndmn')
    correctly_saved = 0
    bn2wndom = get_bn2wndomains()
    filename = os.path.normpath(input_path)
    filename = filename.split(os.sep)[-1]
    filename = filename[:-3]+"wndomains.gold.key.txt"
    for index in range(len(prediction_results)):

        correctly_saved += __write_result(filename,
                                          sentences_xml_elements[index],
                                          resources_path, output_path,
                                          prediction_results[index][1][0],
                                          vocab=vocab_label_wndmn,
                                          enable_coarse_grained=2,
                                          vocab_for_coarse=bn2wndom)

    print("Successfully saved {} out of {}".format(correctly_saved, len(prediction_results)))
    del prediction_results
    print("Of these, {} were MFS".format(mfs_counter))
    mfs_counter = 0
    return


def predict_lexicographer(input_path : str, output_path : str, resources_path : str) -> None:
    """
    DO NOT MODIFY THE SIGNATURE!
    This is the skeleton of the prediction function.
    The predict function will build your model, load the weights from the checkpoint and write a new file (output_path)
    with your predictions in the "<id> <lexicographerId>" format (e.g. "d000.s000.t000 noun.animal").

    The resources folder should contain everything you need to make the predictions. It is the "resources" folder in your submission.

    N.B. DO NOT HARD CODE PATHS IN HERE. Use resource_path instead, otherwise we will not be able to run the code.
    If you don't know what HARD CODING means see: https://en.wikipedia.org/wiki/Hard_coding

    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: None
    """
    global mfs_counter
    print(">>>> LEXICOGRAPHER PREDICTION")
    prediction_results, sentences_xml_elements = __predict(input_path, resources_path)
    vocab_label_lex = create_mapping_dictionary(resources_path, mode='lex')
    correctly_saved = 0
    filename = os.path.normpath(input_path)
    filename = filename.split(os.sep)[-1]
    bn2lex = get_bn2lex()
    filename = filename[:-3] + "lexicon.gold.key.txt"
    for index in range(len(prediction_results)):
        correctly_saved += __write_result(filename,
                                          sentences_xml_elements[index],
                                          resources_path,output_path,
                                          prediction_results[index][2][0],
                                          vocab= vocab_label_lex,
                                          enable_coarse_grained=3,
                                          vocab_for_coarse=bn2lex)

    print("Successfully saved {} out of {}".format(correctly_saved, len(prediction_results)))
    del prediction_results
    print("Of these, {} were MFS".format(mfs_counter))
    mfs_counter = 0
    return


def __predict(input_path : str, resources_path : str) -> Tuple:
    """
    Actually predicts a sentence and returns the predictions in the requested formats
    :param input_path: the path of the input file to predict in the same format as Raganato's framework (XML files you downloaded).
    :param output_path: the path of the output file (where you save your predictions)
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :return: The actual prediction by the network
    """
    train, etree_data = load_dataset(input_path)
    train = [dato for dato in train if dato]
    vocab_label_wndmn = create_mapping_dictionary(resources_path, mode='wndmn')
    vocab_label_bn = create_mapping_dictionary(resources_path, mode='bn')
    vocab_label_lex = create_mapping_dictionary(resources_path, mode='lex')
    modello = WSD(resources_path+"/vocabularies/bert_vocab.txt", [len(vocab_label_bn), len(vocab_label_wndmn), len(vocab_label_lex)], dropout=0.1, recurrent_dropout=0.1,learning_rate=0.0003)
    tokenizatore = modello.tokenizatore
    modello.model.load_weights(resources_path+"/saved_model/model_20_2.14.h5")
    to_return = []
    sentences_xml_elements = etree_data.xpath("/*/*/*")
    for sentence in train:
        feature_1, feature_2, feature_3 = convert_sentence_to_features_no_padding(sentence,tokenizatore)
        results = modello.model.predict(
            {'input_word_ids': feature_1, 'input_mask': feature_2, 'segment_ids': feature_3},
            verbose=1
        )
        to_return.append(results)
    del vocab_label_lex
    del vocab_label_wndmn
    del vocab_label_bn
    return to_return, sentences_xml_elements


def __write_result(filename: str,
                   frase,
                   resources_path: str,
                   outputh_path: str,
                   predictions,
                   vocab = None,
                   enable_coarse_grained: int = 1,
                   vocab_for_coarse = None) -> int:
    """
    Write results in the file system
    :param filename: the name of the file to save
    :param frase: the object from which recover the sentence
    :param resources_path: the path of the resources folder containing your model and stuff you might need.
    :param output_path: the path of the output file (where you save your predictions)
    :param predictions: the predictions made by the system
    :param vocab: the vocab needed for giving a sense
    :param enable_coarse_grained: changes the flow of the function from fine-grained to coarse-grained. Default to 1. Possible values:
        1 --> Means I'm predicting with Babelnet. No extra precautions needed
        2 --> Means I'm predicting with WordNet Domains. Need to consult the vocab. If I don't find anything, the empty class "factotum" is returned instead
        3 --> Means I'm predicting with Lexicon. Need to consult the vocab.
    :param vocab_for_coarse: The vocab in support of mode 2 or 3
    :return: 1 if succeeds
    """
    global mfs_counter
    bn2wn = get_bn2wn()
    lemma2wn = reload_word_mapping(resources_path+"/mapping/lemma2wn.txt")
    to_write = []
    for index, parola in enumerate(frase):
        name = parola.xpath('name()')
        if name == 'instance':
            id = parola.get('id')
            list_of_possible_senses_first_step = lemma2wn.get(parola.text)
            if not list_of_possible_senses_first_step:
                # MFS
                the_actual_meaning = MFS(parola,
                                         bn2wn,
                                         vocab2=vocab_for_coarse,
                                         pred_case=enable_coarse_grained)
                mfs_counter += 1
                to_write.append((id, the_actual_meaning))
                continue
            list_of_possible_senses_bn_version = convert_from_wnlist_2_bnlist(list_of_possible_senses_first_step, bn2wn)

            candidates,list_of_possible_senses_bn_version = create_custom_label(list_of_possible_senses_bn_version,
                                                                                parola.text,
                                                                                vocab,
                                                                                predictions[index],
                                                                                enable_coarse_grained=enable_coarse_grained)
            the_actual_meaning = None
            if candidates:
                argmax = np.argmax(candidates)
                the_actual_meaning = list_of_possible_senses_bn_version[argmax]
            else:
                #MFS
                mfs_counter += 1
                the_actual_meaning = MFS(parola,
                                         bn2wn,
                                         vocab2=vocab_for_coarse,
                                         pred_case=enable_coarse_grained)
            to_write.append((id, the_actual_meaning))
    with open(outputh_path + "/"+filename, "a") as test_saving:
        for tupla in to_write:
            test_saving.write(tupla[0] + " " + tupla[1]+"\n")
    del to_write
    del lemma2wn
    del bn2wn
    return 1


def MFS(parola, vocab: Dict, vocab2:Dict = None, pred_case: int = 1) -> str:
    """
    Returns the sense by applying the Most Frequent Sense (MFS) strategy
    :param parola: the Element object to which associate a sense
    :param vocab: the vocab needed for giving a sense
    :param vocab2: default to None. The other vocabulary to use if coarse-grained mode is enabled. Has to be populated if enable_coarse_grained
    :param pred_case: whether to adopt a "rollback" strategy such as MFS or not. Possible values:
        1 --> Means I'm predicting with Babelnet. No extra precautions needed
        2 --> Means I'm predicting with WordNet Domains. Need to consult the vocab. If I don't find anything, the empty class "factotum" is returned instead
        3 --> Means I'm predicting with Lexicon. Need to consult the vocab.
    :return: the chosen sense with the MFS technique
    """
    pos = parola.get('pos')
    pos_input = __decide_pos(pos)
    wordnet_object = wordnet.synsets(parola.get('lemma'), pos=pos_input)
    try:
        wordnet_object = wordnet_object[0]
    except:
        print(wordnet_object)
        print(parola.text)
    wn_synset = "wn:" + str(wordnet_object.offset()).zfill(8) + wordnet_object.pos()
    the_actual_meaning = next(key for key, value in vocab.items() if wn_synset in value)
    to_return = __extrapolate_value_for_MFS(the_actual_meaning,vocab=vocab2, pred_case=pred_case)
    return to_return


def __extrapolate_value_for_MFS(value: object, pred_case: int = 1, vocab: Dict = None) -> str:
    """
    Taking either a List or String in input, that represents the found Babelnet ID, this function handles it and return a string that contains the value of the prediction
    :param value: The Value from which to extrapolate the actual meaning found
    :param pred_case: whether to adopt a "rollback" strategy such as MFS or not. Possible values:
        1 --> Means I'm predicting with Babelnet. No extra precautions needed
        2 --> Means I'm predicting with WordNet Domains. Need to consult the vocab. If I don't find anything, the empty class "factotum" is returned instead
        3 --> Means I'm predicting with Lexicon. Need to consult the vocab.
    :param vocab: The vocab in support of mode 2 or 3.
    :return: the actual meaning found with MFS
    """
    the_meaning_to_explot = __type_checker(value)
    if pred_case == 1:
        return the_meaning_to_explot
    if pred_case == 2:
        to_return = vocab.get(the_meaning_to_explot)
        return to_return[0] if to_return else "factotum"
    if pred_case == 3:
        to_return = vocab.get(the_meaning_to_explot)
        return to_return[0]

def __type_checker(value: object) -> str:
    """
    Checks the type of the object and, accordingly, returns it
    :param value: the value to examinate
    :return: a string that is the value expected
    """
    if type(value) == str:
        return value
    if type(value) == list:
        return value[0]

def __decide_pos(pos: str) -> str:
    """
    Decides the WN representation of the given pos in input
    :param pos: the pos to interpret with WordNet
    :return: the WN representation of the given pos
    """
    to_return = None
    if pos == 'NOUN':
        to_return = "n"
    if pos == 'VERB':
        to_return = 'v'
    if pos == 'ADJ':
        to_return = 'a'
    if pos == 'ADV':
        to_return = 'r'
    return to_return


def convert_from_wnlist_2_bnlist(list_of_bn: List, vocab: Dict) -> List:
    """
    Cast the given list (which contains only WN ids) to Babelnet IDs
    :param list_of_bn: the list to cast
    :param vocab: the vocabulary to use to perform the conversion
    :return: the converted list
    """
    list_of_possible_senses_bn_version = []
    for candidate in list_of_bn:
        is_it_here = next(key for key, value in vocab.items() if candidate in value)
        if is_it_here:
            list_of_possible_senses_bn_version.append(is_it_here if type(is_it_here) == str else is_it_here[0])
    return list_of_possible_senses_bn_version

def create_custom_label(list_of_possible_senses: List, word: str, vocab: Dict, predictions, enable_coarse_grained: int = 1) -> List:
    """
    Converts the list of babelnet IDS to a number and outputs the converted list
    :param list_of_possible_senses: the list that contains all the babelnet's IDs
    :param word: the word for which we are predicting the sense in a specific moment
    :param vocab: the vocabulary Word -> Serial to exploit for the conversion
    :param predictions: the predictions made by the system
    :param enable_coarse_grained: changes the flow of the function from fine-grained to coarse-grained. Default to None. Possible values:
        1 --> The flow will still be the same
        2,3 -> Flow will change, triggering the first step for the coarse-grained approach.
    :return: a List with the IDs converted
    """
    to_return = []
    list_of_indices_to_delete = []
    for indice in range(len(list_of_possible_senses)):
        new_string = word + "_" + list_of_possible_senses[indice] if enable_coarse_grained == 1 else list_of_possible_senses[indice]
        conversion = None
        try:
            conversion = int(vocab[new_string])
            to_return.append(predictions[conversion])
        except:
            list_of_indices_to_delete.append(indice)
            continue
    if list_of_indices_to_delete:
        list_of_possible_senses = [list_of_possible_senses[prov_index] for prov_index in range(len(list_of_possible_senses)) if prov_index not in list_of_indices_to_delete]
    return to_return, list_of_possible_senses



if __name__ == "__main__":
    predict_babelnet("/Users/gimmi/Desktop/Università/MAGISTRALE/NLP/nlp-finalproject/dataset/test/senseval3.data.xml", "../output", "/Users/gimmi/Desktop/Università/MAGISTRALE/NLP/nlp-finalproject/resources")
    #predict_wordnet_domains("/Users/gimmi/Desktop/Università/MAGISTRALE/NLP/nlp-finalproject/dataset/test/senseval3.data.xml", "../output", "/Users/gimmi/Desktop/Università/MAGISTRALE/NLP/nlp-finalproject/resources")
    #predict_lexicographer("/Users/gimmi/Desktop/Università/MAGISTRALE/NLP/nlp-finalproject/dataset/test/senseval3.data.xml", "../output", "/Users/gimmi/Desktop/Università/MAGISTRALE/NLP/nlp-finalproject/resources")
