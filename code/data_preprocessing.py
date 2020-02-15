from typing import List, Set, Dict
from bs4 import BeautifulSoup
import string
from nltk.corpus import wordnet as wn
from lxml import etree
from io import StringIO, BytesIO
import time

def load_dataset(filepath_train: str, filepath_label_train: str) -> List[str]:
    """
    Reads the specified data from input and returns a list that will represent the dataset of the given language
    :param filepath_train: the path of the file that holds the training sentences
    :param filepath_label_train: the path of the file that holds the labels
    :return the loaded dataset
    """
    print("entered")
    print("Loading with lxml")
    t = time.process_time()
    with open(filepath_train, "rb") as infile:
        tree_train = etree.fromstring(infile.read())
    print("Loading done in ", (time.process_time() - t))
    print("Creating the sentences...")
    t = time.process_time()
    sentences = __load_train_sentences(tree_train)
    print("Sentences created in ", (time.process_time() - t))
    print("Cleaning...")
    t = time.process_time()
    train_data = [__clean_phrase(sentence) for sentence in sentences]
    print("Clean done in ", (time.process_time() - t) )
    print("Loading ", filepath_label_train)
    with open(filepath_label_train, encoding="utf-8") as infilez:
        contents_label = infilez.read().splitlines()
    #Assuming that the ids are unique
    #Also, cleaning empty strings
    print("Cleaning...")
    contents_label = [label for label in contents_label if label != "" or label is not None]
    print("Creating dict")
    pool_of_labels = {frase.split(" ")[0] : frase.split(" ")[1] for frase in contents_label}
    print("Creating labels...")
    train_labels = __create_labeled_sentences(pool_of_labels,tree_train)
    return train_data


def __load_train_sentences(tree : etree._Element ):
    toReturn = []
    sentences_xml_elements = tree.xpath("/*/*/*")
    for sentence_xml in sentences_xml_elements:
        children = sentence_xml.getchildren()
        phraseToBuild = ' '.join([word.text for word in children])
        toReturn.append(phraseToBuild)
    return toReturn

def __create_labeled_sentences(pool_of_labels: dict, tree : etree._Element ) -> None:
    toReturn = []
    sentences_xml_elements = tree.xpath("/*/*/*")
    for sentence in sentences_xml_elements:
        wn_domains_sentence = ""
        bn_domains_sentence = ""
        lex_domains_sentence = ""
        children = sentence.getchildren()
        for word in sentence:
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
                    print("getting wordnet_synset")
                    t = time.process_time()
                    wordnet_synset = wn.lemma_from_key(sense_key).synset()
                    print("Done in ", (time.process_time() - t))
                    synset_id = "wn:" + str(wordnet_synset.offset()).zfill(8) + wordnet_synset.pos()
                    wn_domains_sentence += ' ' + synset_id

                    continue
            else:
                continue

        #getting all things to predict...
        instances = sentence.find_all('instance')

        print("ciao")

def put_correct_id(synset_id: str, mode: str='bn') -> str:
    if mode == 'wn_dom':
        try:
            pass
            #Cerca nella mappa
        except:
            pass
            #Non l'ho trovato. Metti factotum
    else:
        pass
        #semplicemnete, cercalo normalmente



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
    with open(filepath) as file:
        content = file.read()
    dictionary = {mapping.split()[0]:mapping.split()[1] for mapping in content.split("\n") if mapping}
    return dictionary

wndomains2bn = __load_synset_mapping('../resources/babelnet2wordnet.tsv')
wn2bn = __load_synset_mapping('../resources/babelnet2wndomains.tsv')
lex2bn = __load_synset_mapping('../resources/babelnet2wlexnames.tsv')


if __name__ == "__main__":
    lista = load_dataset("../dataset/SemCor/semcor.data.xml", "../dataset/SemCor/semcor.gold.key.txt")
    print(lista[17:100])
