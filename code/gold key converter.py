from collections import defaultdict
from typing import Dict
from nltk.corpus import wordnet as wn
from data_preprocessing import __load_synset_mapping, load_dataset, load_gold_key_file


bn2wn = __load_synset_mapping("/Users/gimmi/Desktop/Università/MAGISTRALE/NLP/nlp-finalproject/resources/babelnet2wordnet.tsv") 
new_dict = dict()
bn2wndomains = __load_synset_mapping("/Users/gimmi/Desktop/Università/MAGISTRALE/NLP/nlp-finalproject/resources/babelnet2wndomains.tsv")
bn2lex = __load_synset_mapping("/Users/gimmi/Desktop/Università/MAGISTRALE/NLP/nlp-finalproject/resources/babelnet2lexnames.tsv")


with open("/Users/gimmi/Desktop/Università/MAGISTRALE/NLP/nlp-finalproject/dataset/test/senseval3.gold.key.txt", encoding="utf-8") as infilez:
        contents_label = infilez.read().splitlines()
contents_label = [label for label in contents_label if label != "" or label is not None]
pool_of_labels = {frase.split(" ")[0] : frase.split(" ")[1] for frase in contents_label}

for key, value in pool_of_labels.items():
    wordnet_synset = wn.lemma_from_key(value).synset() 
    synset_id = "wn:" + str(wordnet_synset.offset()).zfill(8) + wordnet_synset.pos() 
    bn = next(chiave for chiave, valore in bn2wn.items() if synset_id in valore) 
    lex = bn2lex.get(bn)
    new_dict[key] = lex[0]


with open("hope_file.lex.txt","a") as hoping:
    for nchiave, nvalore in new_dict.items(): 
        hoping.write(nchiave + " " + nvalore+"\n")

