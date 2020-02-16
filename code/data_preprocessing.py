from typing import List, Set, Dict
from collections import defaultdict
import string
from nltk.corpus import wordnet as wn
from lxml import etree
import tensorflow as tf
import os
from io import StringIO, BytesIO
import time
tf.compat.v1.enable_eager_execution()

def load_dataset(filepath_train: str, filepath_label_train: str) -> List[str]:
    """
    Reads the specified data from input and returns a list that will represent the dataset of the given language
    :param filepath_train: the path of the file that holds the training sentences
    :param filepath_label_train: the path of the file that holds the labels
    :return the loaded dataset
    """
    is_it_multilingual = any(lang in os.path.basename(filepath_label_train) for lang in ["it","es","de","fr"])
    path_for_eventual_existing_label_data = '../dataset/parsed/' + ('eval_dataset/parsed' if is_it_multilingual else '') +os.path.basename(filepath_label_train[0:-3]+'casted.txt')
    path_for_eventual_existing_train_data = '../dataset/parsed/' + ('eval_dataset/parsed' if is_it_multilingual else '') +os.path.basename(filepath_train[0:-3]+'casted.txt')
    train_data = None
    train_labels = None
    if os.path.exists(path_for_eventual_existing_train_data):
        train_data = __reload_data_from_disk(path_for_eventual_existing_train_data)
    if os.path.exists(path_for_eventual_existing_label_data):
        train_labels = __reload_data_from_disk(path_for_eventual_existing_label_data, data='label')
        return train_data, train_labels

    with open(filepath_train, "rb") as infile:
        tree_train = etree.fromstring(infile.read())

    sentences = __load_train_sentences(tree_train)
    train_data = [__clean_phrase(sentence) for sentence in sentences]

    with open(path_for_eventual_existing_train_data, 'a') as train:
        for frase in train_data:
            train.write("#"+frase+"#")

    with open(filepath_label_train, encoding="utf-8") as infilez:
        contents_label = infilez.read().splitlines()

    contents_label = [label for label in contents_label if label != "" or label is not None]
    pool_of_labels = {frase.split(" ")[0] : frase.split(" ")[1] for frase in contents_label}
    tree_train_sentences = [sentence for sentence in tree_train.xpath("/*/*/*")]
    train_labels = __create_labeled_sentences(pool_of_labels,tree_train_sentences, path_for_eventual_existing_label_data)
    return train_data, train_labels

def __reload_data_from_disk(filepath: str, data: str ='train'):
    """
    Reload the previously parsed files from disk
    :param filepath: the path of the file to be loaded
    :param data: whether the file is for train or other purposes
    :return: a List (if data == 'train') or a List of List (if data != 'train') containing the data of the loaded file
    """
    with open(filepath, 'r') as writer:
        content = writer.read()
    toReturn = []
    if data == 'train':
        toReturn = [word for word in content[1:-1].split('#')]
        return toReturn
    for word in content[1:-1].split('#'):
        if not word or word == '':
            continue
        tmp = [frase for frase in word.split('~')]
        toReturn.append(tmp)
    return toReturn


def __load_train_sentences(tree : etree._Element ) -> List:
    """
    Parse the lxml tree object to get the train sentences
    :param tree: the parsed .xml objet
    :return: the list containing all the train sentences
    """
    toReturn = []
    sentences_xml_elements = tree.xpath("/*/*/*")
    for sentence_xml in sentences_xml_elements:
        children = sentence_xml.getchildren()
        phraseToBuild = ' '.join([word.text for word in children])
        toReturn.append(phraseToBuild)
    return toReturn

def __create_labeled_sentences(pool_of_labels: dict, tree_train_sentences : List, filepath: str) -> List[List]:
    """
    Creates the labels for the model
    :param pool_of_labels: the gold key file parsed to a dict
    :param tree: the parsed .xml file
    :param filepath: where to save the output of this function
    :return: a List of List which contains all the labeled sentences
    """
    toReturn = []
    wn_domains_sentence = ""
    bn_domains_sentence = ""
    lex_domains_sentence = ""
    for word in tree_train_sentences:
        name = word.xpath("name()")
        if name is not None or name is not "":
            if name == 'wf':
                wn_domains_sentence += ' ' + word.text
                bn_domains_sentence += ' ' + word.text
                lex_domains_sentence += ' ' + word.text
                continue
            if name == 'instance':
                id = word.get('id')
                sense_key = pool_of_labels[id]
                wordnet_synset = wn.lemma_from_key(sense_key).synset()
                synset_id = "wn:" + str(wordnet_synset.offset()).zfill(8) + wordnet_synset.pos()
                wn_domains_sentence += ' ' + put_correct_id(word.text,synset_id,mode='wn_dom')
                bn_domains_sentence += ' ' + put_correct_id(word.text,synset_id)
                lex_domains_sentence += ' ' + put_correct_id(word.text,synset_id,mode='lex')
                continue
        else:
            continue
        toReturn.append([wn_domains_sentence.strip(),bn_domains_sentence.strip(),lex_domains_sentence.strip()])
        wn_domains_sentence = ""
        bn_domains_sentence = ""
        lex_domains_sentence = ""
    with open(filepath, 'a') as writer:
        for matrix in toReturn:
            writer.write('#' + matrix[0].strip() + '~' + matrix[1].strip() + '~' + matrix[2].strip() + '#')
    return toReturn


def put_correct_id(word:str, synset_id: str, mode: str='bn') -> str:
    """
    Finds the correct synset id for the given id
    :param word: the word that the system is evaluating right now
    :param synset_id: the id associated with the current :word
    :param mode: the id that I have to look for. Possible options: BabelNet (bn), WordNet Domains (wn_dom), LexNames (lex)
    :return: the found id if exists, else the word that the system is looking in this exact moment
    """
    if mode == 'wn_dom':
        try:
            synset_bn = [key for key, value in bn2wn.items() if synset_id in value][0]
            synset_wn_dom = bn2wndomains[synset_bn][0]
            return synset_wn_dom
        except:
            return "factotum"
    if mode == 'bn':
        for key, value in bn2wn.items():
            if synset_id in value:
                return key
    else:
        try:
            synset_bn = [key for key, value in bn2wn.items() if synset_id in value][0]
            synset_lex = bn2lex[synset_bn][0]
            return synset_lex
        except IndexError:
            import traceback
            traceback.print_exc()
            return word
    return word


def calculate_train_output_size(train_data: List, max_size_of_vocab: int = 30000, min_count: int = 3):
    """
    Calculates the output size of the train data
    :param train_data: the list that will represent the train data
    :param max_size_of_vocab: max size of the vocab. Usually goes for 30k
    :param min_count: min count of the vocab. Usually is 3
    :return: an integer with the size of the output_size unit
    """
    pass



def find_count_of_words(data, max_size_of_vocab: int, min_count: int, mode: str = "word") -> Dict:
    """
    Calculate the number of words
    :param data: the data where to count the words. Can either be a list or a list of list.
    :param max_size_of_vocab: max size of the vocab. Usually goes for 30k
    :param min_count: min count of the vocab. Usually is 3
    :param mode: whether I need to count the #words, or #bn ids or #lex ids or #wndmn ids
    :return: the count inside a dict ordered like word -> count
    """
    toReturn = None
    if mode == 'bn':
        pass


def __clean_phrase(sentence: str) -> str:
    """
    Given a sentence to clean, the function cleans the string
    :param sentence: the string to clean
    :return: the cleared string
    """
    punctuation_set = set(string.punctuation+"\n\t")
    # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    # Serve quindi una lookup table --> dizionario
    dictionary = dict((sign, " ") for sign in punctuation_set)
    sentence = sentence.translate(str.maketrans(dictionary))
    return sentence


def __load_synset_mapping(filepath: str) -> Dict:
    """
    Loads the synset mapping from the system
    :param filepath: the path of the file to load
    :return: a dict representation of the loaded file
    """
    dizionario = defaultdict(list)
    with open(filepath) as file:
        content = file.read()
    for arr in content.split("\n"):
        for key, value in zip(arr.split()[::-1], arr.split()[1::-1]):
            dizionario[key] += [value] if type(value) is str else value
    return dizionario


bn2wn = __load_synset_mapping('../resources/babelnet2wordnet.tsv')
bn2wndomains = __load_synset_mapping('../resources/babelnet2wndomains.tsv')
bn2lex = __load_synset_mapping('../resources/babelnet2lexnames.tsv')


if __name__ == "__main__":
    train,label = load_dataset("../dataset/SemCor/semcor.data.xml", "../dataset/SemCor/semcor.gold.key.txt")
    print(train[17:100])
