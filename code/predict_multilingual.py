from data_preprocessing import load_multilingual_mapping, create_mapping_dictionary, get_bn2wn
from predict import __predict, __decide_pos
from typing import List, Dict, Tuple
from nltk.corpus import wordnet
import numpy as np
import os


mfs_counter = 0

def predict_multilingual(input_path : str, output_path : str, resources_path : str, lang : str) -> None:
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
    :param lang: the language of the dataset specified in input_path
    :return: None
    """
    global mfs_counter
    print(">>>> BABELNET MULTILANG PREDICTION")
    prediction_results, sentences_xml_elements = __predict(input_path,resources_path)
    vocab_label_bn = create_mapping_dictionary(resources_path, mode='bn')
    correctly_saved = 0
    filename = os.path.normpath(input_path)
    filename = filename.split(os.sep)[-1]
    filename = filename[:-3]+"babelnet_"+lang+".gold.key.txt"
    lang_vocab = load_multilingual_mapping(resources_path+ "/mapping/lemma2synsets4.0.xx.wn.ALL.txt",lang=lang[0:2].upper())
    for index in range(len(prediction_results)):

        correctly_saved += __write_result_multilang(filename,
                                          sentences_xml_elements[index],
                                          output_path,
                                          prediction_results[index][0][0],
                                          lang_vocab,
                                          lang,
                                          support_dict=vocab_label_bn)

    print("Successfully saved {} out of {}".format(correctly_saved, len(prediction_results)))
    del prediction_results
    print("Of these, {} were MFS".format(mfs_counter))
    mfs_counter = 0
    return
    pass

def __write_result_multilang(filename: str,
                   frase,
                   outputh_path: str,
                   predictions,
                   lang_dict: Dict,
                   lang: str,
                   support_dict:Dict = None) -> int:
    """
    Write results in the file system
    :param filename: the name of the file to save
    :param frase: the object from which recover the sentence
    :param predictions: the predictions made by the model
    :param lang_dict: the language dictionary
    :param lang: the language that I'm working with
    :param support_dict: the dictionary of labels of Babelnet
    :return: 1 if succeeds
    """

    global mfs_counter
    bn2wn = get_bn2wn()
    to_write = []
    for index, parola in enumerate(frase):
        name = parola.xpath("name()")
        if name == 'instance':
            id = parola.get('id')
            if parola.text == 'documento':
                print("ciao")
            list_of_possible_senses_first_step = lang_dict.get(parola.text)
            if not list_of_possible_senses_first_step:
                # MFS
                the_actual_meaning = MFS(parola,
                                         bn2wn,
                                         lang_dict,
                                         lang)
                mfs_counter += 1
                to_write.append((id, the_actual_meaning))
                continue
            candidates, list_of_possible_senses_first_step= convert_from_bnlist_2_argmax_candidates(list_of_possible_senses_first_step, support_dict,predictions)
            the_actual_meaning = None
            if candidates:
                argmax = np.argmax(candidates)
                the_actual_meaning = list_of_possible_senses_first_step[argmax]
            else:
                #MFS
                mfs_counter += 1
                the_actual_meaning = MFS(parola,
                                         bn2wn,
                                         lang_dict,
                                         lang)
            to_write.append((id, the_actual_meaning))
    with open(outputh_path + "/"+filename, "a") as test_saving:
        for tupla in to_write:
            test_saving.write(tupla[0] + " " + tupla[1]+"\n")
    del to_write
    del bn2wn
    return 1


def convert_from_bnlist_2_argmax_candidates(list_of_bn: List, label_vocab: Dict, predictions) -> Tuple:
    """
    Cast the given list (which contains only BN ids) to numbers that are going to be candidates for the argmax function
    :param list_of_bn: the list to cast
    :param label_vocab: the vocabulary to use to perform the conversion
    :param predictions: the predictions made by the system
    :return: the converted list
    """
    list_of_candidates = []
    list_of_indices_to_delete = []
    for candidate_index in range(len(list_of_bn)):
        try:
            is_it_here = next(value for key, value in label_vocab.items() if '_bn:' in key and key.split('_')[1] == list_of_bn[candidate_index])
            conversion = predictions[int(is_it_here)]
            list_of_candidates.append(conversion)
        except:
            list_of_indices_to_delete.append(candidate_index)
    if list_of_indices_to_delete:
        list_of_bn = [list_of_bn[prov_index] for prov_index in range(len(list_of_bn)) if prov_index not in list_of_indices_to_delete]
    return list_of_candidates,list_of_bn


def MFS(parola, vocab: Dict,lang_vocab:Dict, lang: str) -> str:
    """
    Returns the sense by applying the Most Frequent Sense (MFS) strategy with a multilingual approach
    :param parola: the word to use for the MFS approach
    :param vocab: the babelnet 2 wordnet dictionary
    :param lang_vocab: the language dictionary
    :param lang: the language currently used
    :return: the chosen sense with the MFS technique
    """

    pos = parola.get('pos')
    pos_input = __decide_pos(pos)
    look_up = lang_vocab.get(parola.text)
    language_to_use = next(language for language in wordnet.langs() if language[0:2] == lang or language == lang )
    wordnet_object = wordnet.synsets(parola.text if not look_up else look_up[0], pos=pos_input, lang=language_to_use)
    try:
        wordnet_object = wordnet_object[0]
    except:
        print(wordnet_object)
        print(parola.text)
    wn_synset = "wn:" + str(wordnet_object.offset()).zfill(8) + wordnet_object.pos()
    the_actual_meaning = next(key for key, value in vocab.items() if wn_synset in value)
    if type(the_actual_meaning) == str:
        return the_actual_meaning
    if type(the_actual_meaning) == list:
        return the_actual_meaning[0]

if __name__ == "__main__":
    predict_multilingual("/Users/gimmi/Desktop/Università/MAGISTRALE/NLP/nlp-finalproject/dataset/eval_dataset/semeval2015.it.data.xml","../output","../resources","ita")
