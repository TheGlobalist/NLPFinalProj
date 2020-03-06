from typing import List, Dict, Tuple
from collections import defaultdict, Counter, OrderedDict
from nltk.corpus import wordnet as wn
import nltk
from lxml import etree
import os

def __check_if_file_exists(filename: str) -> bool:
    """
    Check if the file already exists parsed
    :param filename: the name of the file to look for
    :return: True if the file is already in the resources directory, parsed. False otherwhise.
    """
    return any(files for root, dirs, files in os.walk("../") if filename in files)


def __start_dataset_parsing(filepath: str):
  """
  Parse the given filepath and returns an lxml etree object.
  :param filepath: the path of the file to parse
  :return its representation as an lxml etree object
  """
  with open(filepath, "rb") as infile:
    tree_train = etree.fromstring(infile.read())
  return tree_train


def load_dataset(filepath: str) -> Tuple:
    """
    Loads a given file from input and returns it as a List of List. This function supposes that the incoming input is a .xml file and that the path
    actually brings to the file
    :param filepath_train: the either relative or absolute path
    :return: a List of List that contains the parsed data and the etree object that will have to be passed to the load_gold_key_file, if a label file will be supplied
    """
    filename = filepath[filepath.rfind('/') + 1:-3] + "casted.txt"
    already_parsed = __check_if_file_exists(filename)
    tree_train = __start_dataset_parsing(filepath)
    if already_parsed:
        print("File was already parsed. Reloading it...")
        filepath = "../resources/parsed/"+filename if 'dev' not in filepath else "../resources/parsed/dev/"+filename
        data = __reload_data_from_disk(filepath, data='train')
        return data, tree_train
    print("Parse dataset starting...")
    sentences = __load_train_sentences(tree_train)
    train= [sentence.strip() for sentence in sentences]
    with open('../resources/parsed/'+filename, 'a') as to_save:
        for frase in train:
            to_save.write("#"+frase+"#"+"\n")
    print("Done. Returning...")
    return train, tree_train


def load_gold_key_file(filepath: str, tree_train: etree._Element) -> List[List]:
    """
    Loads a given file from input and returns it as a List. This function supposes that the incoming input is a .txt file and that the path
    actually brings to the file. The file is considered to be the gold.key of the previously loaded file
    :param filepath_train: the either relative or absolute path
    :return: a List of List that contains the parsed data
    """
    filename = filepath[filepath.rfind('/') + 1:-3] + "casted.txt"
    print("Checking ", filename)
    already_parsed = __check_if_file_exists(filename)
    if already_parsed:
        filepath = "../resources/parsed/"+filename if 'dev' not in filepath else "../resources/parsed/dev/"+filename
        data = __reload_data_from_disk(filepath, data='test')
        print("Done loading gold key")
        return data
    with open(filepath, encoding="utf-8") as infilez:
        contents_label = infilez.read().splitlines()
    contents_label = [label for label in contents_label if label != "" and label != "\n" or label is not None]
    pool_of_labels = {frase.split(" ")[0] : frase.split(" ")[1] for frase in contents_label}
    labels = __create_labeled_sentences(pool_of_labels,tree_train, '../resources/parsed/'+filename)
    print("Done with labels...")
    return labels

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
    mapping_to_persist = defaultdict(list)
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
                    mapping_to_persist[word.text].extend([synset_id])
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
    with open('../resources/mapping/lemma2wn.txt','a') as mapping_lemma:
        for chiave, valore in mapping_to_persist.items():
            stringa = ' '.join(wn_id for wn_id in valore)
            mapping_lemma.write(chiave + "~" + stringa+"#\n")
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


def create_mapping_dictionary(resources_path: str, data = None, max_size_of_vocab: int = 30000, min_count: int = 3, mode: str="word") -> Dict:
    """
    Creates (or reload from disk) the mapping label dictionary of the given mode and file
    :param resources_path: the path where the program should go to look for files
    :param data: The collection that will represent the data from where create the dictionary. This can be None (so, default type) if I just want to "force" the load of something from disk
    :param max_size_of_vocab: max size of the vocab. Usually goes for 30k
    :param min_count: min count of the vocab. Usually is 3
    :param mode: whether I need to handle the #words, or #bn ids or #lex ids or #wndmn ids
    :return: a Dictionary that goes by word -> serial

    """
    print("Creating Mapping dictionary....")
    filename = "/vocabularies/"
    if mode == "word":
        filename += "vocab_train.txt"
    if mode == 'wndmn' or mode == 'lex':
        filename += "vocab_wn_domains_labels.txt" if mode == 'wndmn' else "vocab_lex_labels.txt"
    if mode == 'bn':
        filename += "vocab_bn_labels.txt"
    print(resources_path+filename)
    if os.path.exists(resources_path+filename) and data is None:
        toReturn = __load_vocab_mapping(resources_path+filename)
        print("Done loading dictionary")
        return toReturn

    count_of_words = __count_words(data,mode=mode)
    count_of_words = OrderedDict(
        sorted(count_of_words.items(), key=lambda dict_record: int(dict_record[1]), reverse=True)
    )
    i = 0
    tmp_container = dict()
    for key, value in count_of_words.items():
        if i > max_size_of_vocab:
            #I've reached the max consented length of the dictionary. No need to continue
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


    if not os.path.exists(resources_path):
        os.mkdir(resources_path)
    with open(resources_path+filename, "w+") as fileToWrite:
        for chiavi, valori in toReturn.items():
            fileToWrite.write(chiavi + " " + str(valori) + "\n")
    return toReturn


def load_multilingual_mapping(filepath: str, lang: str='EN') -> Dict:
    """
    Given the base path, looks for the multilingual mapping provided, parse it, and return the 5 dictionaries of the languages (IT,FR,DE,ES)
    :param filepath: the first part of the path that needs to be used to load the file
    :param lang: The language to which pay special attention in the parsing process. Possible values:
        EN --> English will be the only language considered
        IT --> Italian will be the only language considered
        DE --> Deutsch will be the only language considered
        ES --> Spanish will be the only language considered
        FR --> French will be the only language considered
        N.B.: These are just an example basing on the slides given as a guideline. All languages are welcome thanks to the implementation.
    :return: a dictionary that actually represent the mapping of the language
    """
    language_dict = defaultdict(list)
    try:
        nltk.data.find('corpora/omw.zip')
        print("OMW already here")
    except:
        print("Installing OMW...")
        nltk.download('omw')
    with open(filepath) as fileToRead:
        content = fileToRead.readlines()
    for riga in content:
        contenuto_interno = riga.split()
        lemma = contenuto_interno[1]
        lemma = lemma[:lemma.rfind('#')]
        if contenuto_interno[0] == lang:
            language_dict[lemma].extend(contenuto_interno[2:])
    return language_dict

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
        count_of_words = Counter(word for line in data for word in line.split(" ") if "UNK" not in word)
    if mode == 'bn':
        data = [[line for line in test if line and  "bn:" in line] for test in data]
        data= [dato[0] for dato in data if dato]
        count_of_words = Counter(word for line in data for word in line.split(" "))
    return count_of_words


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


def __load_vocab_mapping(filepath: str) -> Dict:
    """
    Loads the vocab mapping from the system
    :param filepath: the path of the file to load
    :return: a dict representation of the loaded file
    """
    print("Reloading dict from disk")
    dizionario = dict()
    with open(filepath) as file:
        content = file.readlines()
    for arr in content:
        k, *v = arr.split()
        dizionario[k] = v[0]
    return dizionario


def reload_word_mapping(filepath:str) -> Dict:
    """
    Loads the lemma to wn id mapping from the FS
    :param filepath: the path of the file to load
    :return: a dict representation of the loaded file
    """
    dizionario = dict()
    with open(filepath) as file:
        content = file.read()
    for arr in content.split('#'):
        if arr == '\n':
            continue
        try:
            k, *v = arr.split('~')
            k = k[1:] if '\n' in k else k
            v = v[0].split(' ')
        except:
            print("ciao")
        dizionario[k] = v
    return dizionario


bn2wn = __load_synset_mapping('../resources/babelnet2wordnet.tsv')
bn2wndomains = __load_synset_mapping('../resources/babelnet2wndomains.tsv')
bn2lex = __load_synset_mapping('../resources/babelnet2lexnames.tsv')


def get_bn2wn() -> Dict:
    return bn2wn

def get_bn2wndomains() -> Dict:
    return bn2wndomains

def get_bn2lex() -> Dict:
    return bn2lex

if __name__ == "__main__":
    train,etree_file = load_dataset("../dataset/SemCor/semcor.data.xml")
    label = load_gold_key_file("../dataset/SemCor/semcor.gold.key.txt", etree_file)
