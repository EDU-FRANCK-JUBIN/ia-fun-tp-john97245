# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 17:06:23 2020

@author: johna
"""

import spacy
from spacy import displacy
nlp = spacy.load('fr_core_news_sm')

doc = nlp("Demain je travaille à la maison avec Martin Noma. C'est un terrible choix")

def decoupageEnMot(texte):
    for token in texte:
        print(token.text)
        
#decoupageEnMot(doc)   


"""
text: Le texte/mot original:

lemma_: la forme de base du mot (pour un verbe conjugué par exemple on aura son infinitif)

pos_. Le tag part - of - speech 

tag: Les informations détaillées part- of - speech 

dep : Dépendance syntaxique 

shape: format/pattern

is_alpha: Alphanumérique ?

is_ stop: Le mot fait-il partie d’une Stop-List ?
"""
def DecoupageEnMotDetaille(texte):
    for token in texte:
        print("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}".format(
                token.text,
                token.idx,
                token.lemma,
                token.is_punct,
                token.is_space,
                token.shape_,
                token.pos_,
                token.tag_,
                token.ent_type_
                ))
        
#DecoupageEnMotDetaille(doc)

def DecoupageEnPhrase(texte):
    for sent in texte.sents:
        print(sent)
        
#DecoupageEnPhrase(doc)

def RecupererPhrasesNominales(texte):
    for chunk in doc.noun_chunks:
        print(chunk.text,"-->",chunk.label_)
        
#RecupererPhrasesNominales(doc)
        
def NameEntityRecognition(texte):
    for ent in texte.ents:
        print(ent.text, ent.label_)
        
#NameEntityRecognition(doc)

def Dependances(texte):
    for token in texte:
        print("{0}/{1} <--{2}-- {3}/{4}".format(
                token.text,
                token.tag_,
                token.dep_,
                token.head.text,
                token.head.tag_
        ))
    displacy.render(doc, style='dep',jupyter=True, options={'distance': 130})
Dependances(doc)