from typing import List, Set, Dict
from bs4 import BeautifulSoup
import string
import re

def load_dataset(filename: str) -> List[str]:
    """
    Reads the specified data from input and returns a list that will represent the dataset of the given language
    :param filename: the file that, inside the path, has to be read
    :return the loaded dataset
    """
    infile = open(filename, "r")
    contents = infile.read()
    soup = BeautifulSoup(contents, 'xml')
    sentences = soup.find_all('sentence')
    sentences = [__clean_phrase(sentence.get_text()) for sentence in sentences]
    return sentences


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


if __name__ == "__main__":
    lista = load_dataset("../dataset/eval_dataset/semeval2013.it.data.xml")
    print(lista[17:100])
