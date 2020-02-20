from typing import List, Set, Dict
from collections import defaultdict, Counter, OrderedDict
import string
from nltk.corpus import wordnet as wn
from lxml import etree
import tensorflow as tf
import os
import re
from nltk.corpus import stopwords


def load_dataset(filepath_train: str, filepath_label_train: str,dev_parsing: bool = False) -> List[str]:
    """
    Reads the specified data from input and returns a list that will represent the dataset of the given language
    :param filepath_train: the path of the file that holds the training sentences
    :param filepath_label_train: the path of the file that holds the labels
    :param dev_parsing: whether the data that the function parses is for dev purposes
    :return the loaded dataset
    """
    is_it_multilingual = any(lang in os.path.basename(filepath_label_train) for lang in ["it","es","de","fr"])
    path_for_eventual_existing_label_data = None
    path_for_eventual_existing_train_data = None

    if is_it_multilingual:
        path_for_eventual_existing_label_data = '../dataset/parsed/eval_dataset/parsed/' +os.path.basename(filepath_label_train[0:-3]+'casted.txt')
        path_for_eventual_existing_train_data = '../dataset/parsed/eval_dataset/parsed/' +os.path.basename(filepath_train[0:-3]+'casted.txt')
    elif dev_parsing:
        path_for_eventual_existing_label_data = '../dataset/parsed/dev/'+os.path.basename(filepath_label_train[0:-3]+'casted.txt')
        path_for_eventual_existing_train_data = '../dataset/parsed/dev/'+ os.path.basename(filepath_train[0:-3] + 'casted.txt')
    else:
        path_for_eventual_existing_label_data = '../dataset/parsed/'+os.path.basename(filepath_label_train[0:-3]+'casted.txt')
        path_for_eventual_existing_train_data = '../dataset/parsed/'+ os.path.basename(filepath_train[0:-3] + 'casted.txt')

    train_data = None
    train_labels = None
    if os.path.exists(path_for_eventual_existing_train_data):
        train_data = __reload_data_from_disk(path_for_eventual_existing_train_data)
    if os.path.exists(path_for_eventual_existing_label_data):
        train_labels = __reload_data_from_disk(path_for_eventual_existing_label_data, data='label')
        return train_data, train_labels
    print("Parse train starting...")
    with open(filepath_train, "rb") as infile:
        tree_train = etree.fromstring(infile.read())

    sentences = __load_train_sentences(tree_train)
    train_data = [sentence.strip() for sentence in sentences]
    print("Done with train. Starting with labels...")
    with open(path_for_eventual_existing_train_data, 'a') as train:
        for frase in train_data:
            train.write("#"+frase+"#"+"\n")

    with open(filepath_label_train, encoding="utf-8") as infilez:
        contents_label = infilez.read().splitlines()

    contents_label = [label for label in contents_label if label != "" or label is not None]
    pool_of_labels = {frase.split(" ")[0] : frase.split(" ")[1] for frase in contents_label}
    train_labels = __create_labeled_sentences(pool_of_labels,tree_train, path_for_eventual_existing_label_data)
    print("Done with labels...")
    return train_data, train_labels

def __reload_data_from_disk(filepath: str, data: str ='train'):
    """
    Reload the previously parsed_yeah files from disk
    :param filepath: the path of the file to be loaded
    :param data: whether the file is for train or other purposes
    :return: a List (if data == 'train') or a List of List (if data != 'train') containing the data of the loaded file
    """
    with open(filepath, 'r') as writer:
        content = writer.read()
    toReturn = []
    if data == 'train':
        toReturn = [word for word in content[1:-1].split('#') if word and word is not '\n']
        return toReturn
    for word in content[1:-1].split('#'):
        if not word or word == '' or word == '\n':
            continue
        tmp = [frase for frase in word.split('~')]
        toReturn.append(tmp)
    return toReturn


def __load_train_sentences(tree : etree._Element ) -> List:
    """
    Parse the lxml tree object to get the train sentences
    :param tree: the parsed_yeah .xml objet
    :return: the list containing all the train sentences
    """
    toReturn = []
    sentences_xml_elements = tree.xpath("/*/*/*")
    for sentence_xml in sentences_xml_elements:
        children = sentence_xml.getchildren()
        phraseToBuild = ' '.join([word.text for word in children])
        toReturn.append(phraseToBuild)
    return toReturn


def __create_labeled_sentences(pool_of_labels: dict, tree : etree._Element, filepath: str) -> List[List]:
    """
    Creates the labels for the model
    :param pool_of_labels: the gold key file parsed_yeah to a dict
    :param tree: the parsed_yeah .xml file
    :param filepath: where to save the output of this function
    :return: a List of List which contains all the labeled sentences
    """
    print("Saving labeled stuff")
    toReturn = []
    sentences_xml_elements = tree.xpath("/*/*/*")
    for sentence in sentences_xml_elements:
        wn_domains_sentence = ""
        bn_domains_sentence = ""
        lex_domains_sentence = ""
        for word in sentence:
            name = word.xpath("name()")
            if name is not None or name is not "":
                if name == 'wf':
                    wn_domains_sentence += ' ' + word.text
                    bn_domains_sentence += ' ' + word.text
                    lex_domains_sentence += ' ' + word.text
                    continue
                if name == 'instance':
                    print("Looking for instance...")
                    id = word.get('id')
                    sense_key = pool_of_labels[id]
                    wordnet_synset = wn.lemma_from_key(sense_key).synset()
                    synset_id = "wn:" + str(wordnet_synset.offset()).zfill(8) + wordnet_synset.pos()
                    wn_domains_sentence += ' ' + put_correct_id(word.text,synset_id,mode='wn_dom')
                    bn_domains_sentence += ' ' + word.text + '_' + put_correct_id(word.text,synset_id)
                    lex_domains_sentence += ' ' + put_correct_id(word.text,synset_id,mode='lex')
                    print("Found ID. done. continuing")
                    continue
            else:
                continue
        toReturn.append([wn_domains_sentence.strip(),bn_domains_sentence.strip(),lex_domains_sentence.strip()])
        print("Fine della sentence")
    print("Scrivo su file...")
    with open(filepath, 'a') as writer:
        for matrix in toReturn:
            writer.write('#' + matrix[0].strip() + '~' + matrix[1].strip() + '~' + matrix[2].strip() + '#' +"\n")
    print("end")
    return toReturn


def put_correct_id(word:str, synset_id: str, mode: str='bn') -> str:
    """
    Finds the correct synset id for the given id
    :param word: the word that the system is evaluating right now
    :param synset_id: the id associated with the current :word
    :param mode: the id that I have to look for. Possible options: BabelNet (bn), WordNet Domains (wn_dom), LexNames (lex)
    :return: the found id if exists, else the word that the system is looking in this exact moment
    """
    # next() seems to be way faster than a classic list comprehension
    #ref -> next(key for key, value in bn2wn.items() if synset_id in value)
    if mode == 'wn_dom':
        try:
            synset_bn = next(key for key, value in bn2wn.items() if synset_id in value)
            synset_wn_dom = bn2wndomains[synset_bn][0]
            return synset_wn_dom
        except:
            return "factotum"
    if mode == 'bn':
        return next(key for key, value in bn2wn.items() if synset_id in value)
    else:
        try:
            synset_bn = next(key for key, value in bn2wn.items() if synset_id in value)
            synset_lex = bn2lex[synset_bn][0]
            return synset_lex
        except IndexError:
            import traceback
            traceback.print_exc()
            return word
    return word


def calculate_train_output_size(data, max_size_of_vocab: int = 30000, min_count: int = 3, mode: str="word") -> Dict:
    """
    Calculates the output size of the train data
    :param data: the list that will represent the train data
    :param max_size_of_vocab: max size of the vocab. Usually goes for 30k
    :param min_count: min count of the vocab. Usually is 3
    :param mode: whether I need to handle the #words, or #bn ids or #lex ids or #wndmn ids
    :return: an integer with the size of the output_size unit

    """
    filename = None
    if mode == "word":
        filename = "vocab_train.txt"
    if mode == 'wndmn' or mode == 'lex':
        filename = "vocab_wn_domains_labels.txt" if mode == 'wndmn' else "vocab_lex_labels.txt"
    if mode == 'bn':
        filename = "vocab_bn_labels.txt"
    path = "../resources/vocabularies/"

    if os.path.exists(path+filename):
        toReturn = __load_synset_mapping(path+filename)
        return toReturn

    count_of_words = __count_words(data,mode=mode)
    count_of_words = OrderedDict(
        sorted(count_of_words.items(), key=lambda k: int(k[1]), reverse=True)
    )
    i = 0
    tmp_container = dict()
    for key, value in count_of_words.items():
        if i > max_size_of_vocab:
            break
        tmp_container[key] = value
        i += 1
    i = 2
    tmp_container = {word: counter for word, counter in tmp_container.items() if counter >= min_count}
    toReturn = {'<PAD>':0, '<UNK>':1}
    for chiave, valore in tmp_container.items():
        if chiave not in toReturn:
            toReturn[chiave] = i
            i += 1
        else:
            continue


    if not os.path.exists(path):
        os.mkdir(path)
    with open(path+filename, "w+") as fileToWrite:
        for chiavi, valori in toReturn.items():
            fileToWrite.write(chiavi + " " + str(valori) + "\n")
    return toReturn



def __count_words(data, mode: str = "word") -> Dict:
    """
    Calculate the number of words and saves it in a file with an offset value
    :param data: the data where to count the words. Can either be a list or a list of list.
    :param max_size_of_vocab: max size of the vocab. Usually goes for 30k
    :param min_count: min count of the vocab. Usually is 3
    :param mode: whether I need to count the #words, or #bn ids or #lex ids or #wndmn ids
    :return: the count inside a dict ordered like word -> count
    """
    count_of_words = None
    if mode == 'word':
        count_of_words = Counter(word for line in data for word in line.split())
    if mode == 'wndmn' or mode == 'lex':
        indexToChoose = 1 if mode == 'wnmdn' else -1
        data = [[line for line in test] for test in data]
        data= [dato[indexToChoose] for dato in data if dato]
        count_of_words = Counter(
            word
            for line in data
            for word in line.split(" ")
            if "UNK" not in word
        )
    if mode == 'bn':
        data = [[line for line in test if line and  "bn:" in line] for test in data]
        data= [dato[0] for dato in data if dato]
        count_of_words = Counter(word for line in data for word in line.split(" "))
    return count_of_words




def __clean_phrase(sentence: str, special: bool = False) -> str:
    """
    Given a sentence to clean, the function cleans the string
    :param sentence: the string to clean
    :param special: whether a different set of punctuation should be used or not
    :return: the cleared string
    """
    tmp = [word for word in sentence.lower().split() if word not in stopwords.words('english')]
    return tmp


def __load_synset_mapping(filepath: str) -> Dict:
    """
    Loads the synset mapping from the system
    :param filepath: the path of the file to load
    :return: a dict representation of the loaded file
    """
    dizionario = defaultdict(list)
    with open(filepath) as file:
        content = file.readlines()
    for arr in content:
        k, *v = arr.split()
        dizionario[k].extend(v)
    return dizionario


bn2wn = __load_synset_mapping('../resources/babelnet2wordnet.tsv')
bn2wndomains = __load_synset_mapping('../resources/babelnet2wndomains.tsv')
bn2lex = __load_synset_mapping('../resources/babelnet2lexnames.tsv')


if __name__ == "__main__":
    train,label = load_dataset("../dataset/SemCor/semcor.data.xml", "../dataset/SemCor/semcor.gold.key.txt")
    print(train[17:100])
