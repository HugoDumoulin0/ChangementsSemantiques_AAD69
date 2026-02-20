#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:05:05 2025

@author: hugodumoulin
"""

import grewpy
from grewpy import Corpus, Request, Graph
import os
import matplotlib.pyplot as plt
import pandas as pd


grewpy.set_config("ud") # ud or basic 

# path="/Users/hugodumoulin/Desktop/ArchivU/Travail/motifs/grewpy-tutorial/SUD_English-PUD/"
# treebank_path="/Users/hugodumoulin/Desktop/ArchivU/Travail/motifs/grewpy-tutorial/SUD_English-PUD/en_pud-sud-test.conllu"

path="./"
treebank_path="/Users/hugodumoulin/Desktop/Cours_SDL/MCF_2025-2026/projet tuteuré/ChangementsSemantiques_AAD69/bibliographie/projet Tuteuré 2025/AAD69-synonymie-contextuelle/data/ParlaMint_100/ParlaMint-FR_2018-07-05-E1006.conll'

corpus = Corpus(treebank_path)
# print(type(corpus))


req1 = Request("pattern { X1 [upos=NOUN] ; X2 [upos=ADJ] ; X1 < X2}")
# req1 = Request('pattern { X1 [Number=Sing,upos=DET,lemma="le",Definite=Def,PronType=Art] ; X2 [Number=Sing] ; X1<X2}')
# req1 = Request('pattern { X1 [upos=DET]; X2 [upos=ADJ]; X3 [upos=NOUN]; X1 < X2; X2 < X3}')

liste_match = corpus.search(req1, deco=True)

def dump_graphique(match):
    sent_id = match["sent_id"]
    print(sent_id)
    deco1 = liste_match[1]["deco"]
    graph = corpus[sent_id]
    os.makedirs(f"{path}/images", exist_ok = True)
    with open(f"{path}/images/{sent_id}.svg", 'w') as f:
            f.write (graph.to_svg(deco=deco1)) 

def index(treebank_path, req, param):
    corpus = Corpus(treebank_path)
    liste_match=corpus.search(req)
    index=[]
    for match in liste_match:
        sent_id = match["sent_id"]
        liste_node=[]
        for node_number in match["matching"]["nodes"].values():
            liste_node.append(int(node_number))
        liste_node.sort()
        liste_forms = []
        for node_number in liste_node:
            liste_forms.append(corpus[sent_id].features[str(node_number)][param])
        index.append(liste_forms)
    return index

for i in index(treebank_path, req1, "form"):
    print(i)

print(liste_match[1])
dump_graphique(liste_match[1])




