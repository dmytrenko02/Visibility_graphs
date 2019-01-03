# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 09:33:23 2018

@author: dmytr
"""

from pandas import Series, DataFrame, read_csv
from networkx import nx

import time
t0 = time.time()

def read_files(): #зчитування файлів csv з папки documents
    directory = 'C:/Users/dmytr/Documents/Python Scripts/VG/ngrams/'
    file_words = read_csv(directory+"words.csv", index_col = 'Id' )
    file_bigrams = read_csv(directory+"bigrams.csv", index_col = 'Id' )
    file_threegrams = read_csv(directory+"threegrams.csv", index_col = 'Id' )
    return file_words, file_bigrams, file_threegrams


file_words, file_bigrams, file_threegrams = read_files()

list_of_words = file_words['Label']
list_of_bigrams_label = file_bigrams['Label']
list_of_threegrams_label = file_threegrams['Label']

list_of_bigrams = list()
list_of_threegrams = list()

for bigram in list_of_bigrams_label:
    list_of_bigrams.append(bigram.split('_'))

for three in list_of_threegrams_label:
    list_of_threegrams.append(three.split('_'))

print 'List_of_words'
print list_of_words
print 'List_of_bigrams'
print list_of_bigrams 
print 'List_of_threegrams'  
print list_of_threegrams

dg = nx.DiGraph()


for word in list_of_words:
    for bigram in list_of_bigrams: #входження слова в біграму
        if word in bigram:
            dg.add_node(word)
            dg.add_node(bigram[0]+'_'+bigram[1])
            dg.add_edge(word, bigram[0]+'_'+bigram[1], weight = 1.0)
    for threegram in list_of_threegrams: #входження слова в триграму
        if word in threegram:
            dg.add_node(word)
            dg.add_node(threegram[0]+'_'+threegram[1]+'_'+threegram[2])
            dg.add_edge(word, threegram[0]+'_'+threegram[1]+'_'+threegram[2], weight = 1.0)

for bigram in list_of_bigrams:
    for threegram in list_of_threegrams:
        if (bigram[0] in threegram)and(bigram[1] in threegram):
            dg.add_node(bigram[0]+'_'+bigram[1])
            dg.add_node(threegram[0]+'_'+threegram[1]+'_'+threegram[2])
            dg.add_edge(bigram[0]+'_'+bigram[1], threegram[0]+'_'+threegram[1]+'_'+threegram[2], weight = 1.0) 

list_of_directedgraph_labels = list()
list_of_directedgraph_labels = dg.nodes()

#mapping = dict(zip(dg.nodes(),list_of_directedgraph_labels))
#dg = nx.relabel_nodes(dg,mapping)
filename = 'nnth.graphml'
nx.write_graphml(dg, filename)

print 'Edges'            
print dg.edges()
print 'Nodes'
print dg.nodes()