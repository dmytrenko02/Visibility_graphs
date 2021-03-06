# -*- coding: utf-8 -*-
"""
Created on Wed Jan 02 11:28:58 2019

@author: dmytr
"""

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
import math
import os
from textblob import TextBlob as tb
from sklearn.feature_extraction import text

import numpy as np
from networkx import nx
from itertools import combinations
import matplotlib.pyplot as plt
from visibility_graph import visibility_graph

import time
t0 = time.time()

def visibility_graph2(g, series, n):
    # convert list of magnitudes into list of tuples that hold the index
    tseries = []
    for magnitude in series:
        tseries.append( (n, magnitude) )
        n += 1
    # contiguous time points always have visibility
    for n in range(0,len(tseries)-1):
        (ta, ya) = tseries[n]
        (tb, yb) = tseries[n+1]
        g.add_node(ta, mag=ya)
        g.add_node(tb, mag=yb)
        g.add_edge(ta, tb)#, weight = int(np.sqrt((yb-ya)*(yb-ya)+(tb-ta)*(tb-ta))))

    for a,b in combinations(tseries, 2):
        # two points, maybe connect
        (ta, ya) = a
        (tb, yb) = b
        connect = True
        
        # let's see all other points in the series
        for tc, yc in tseries:
            # other points, not a or b
            if tc != ta and tc != tb:
                # does c obstruct?
                if yc > yb + (ya - yb) * ( (tb - tc) / (tb - ta) ):
                    connect = False
                   
        if connect:
            g.add_edge(ta, tb)    
    return g

def horizontal_visibility_graph2(g, series, i):

    # convert list of magnitudes into list of tuples that hold the index
    tseries = []
    i = 0
    for magnitude in series:
        tseries.append( (i, magnitude) )
        i += 1
        
    for n in range(0,len(tseries)):
        (ta, ya) = tseries[n]
        g.add_node(ta, mag=ya)
        
    for i in range(0,len(tseries)-1):
        for j in range(i+1,len(tseries)):
            (ti, yi) = tseries[i]
            (tj, yj) = tseries[j]
            if yj>=yi:
                g.add_edge(ti, tj)
                g.add_edge(tj, ti) #if I am seen then I see too
                break
            
    i = len(tseries)-1
    j = i-1
    while i != 0:
        while j > -1:
            (ti, yi) = tseries[i]
            (tj, yj) = tseries[j]
            if yj>=yi:
                g.add_edge(ti, tj)
                g.add_edge(tj, ti) #if I am seen then I see too
                break
            j -= 1
        i -= 1
        j = i-1

    return g

def diff(first, second): #різниця множин
    second = set(second)
    return [item for item in first if item not in second]

def tf_(word, blob):
    return 1.0*blob.count(word) / len(blob)

def n_containing(word, bloblist): #повертає кількість документів, в які входить word
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    return math.log(1.0*len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return 1.0*tf_(word, blob) #* idf(word, bloblist)

def IDF(element, quantity_all_documents):
    return math.log(1.0*quantity_all_documents / (1 + n_containing(element, bloblist)))

def TF(element, all_elements, quantity_all_elements, bloblist):
    return 1.0*all_elements.count(element)/quantity_all_elements#* IDF(element, quantity_all_documents)

def quantity_all_elements(bloblist):
    return sum(len(blob) for blob in bloblist)

def read_files(): #зчитування файлів з папки documents
    blobs = []
    bloblist = []
    all_words = []
    bigrams = []
    bloblist_bigrams = []
    all_bigrams = []
    threegrams = []
    all_threegrams = []
    bloblist_threegrams = []
    directory = 'C:/Users/dmytr/Documents/Python Scripts/VG/documents/'
    files = os.listdir(directory) 
    for file_ in files:
        threegrams = []
        f_input = open(directory+file_, 'r')
        blob_ = tb(f_input.read().lower())
        
        all_words = all_words + list(ps.stem(w) for w in diff(blob_.words, my_stop_words)) #стемінг слів
        blobs = blob_.split('***')
            
        for text_ in blobs:
            tx = tb(text_)           
            bloblist.append(list(ps.stem(w) for w in diff(tx.words, my_stop_words)))#list(set(blob_.words).difference(my_stop_words)))
            
            bigrams = []
            for bigram in tx.ngrams(2): #формування списку біграм
                if len(diff(bigram, my_stop_words)) == 2:
                    bigram = list(ps.stem(w) for w in bigram) #стемінг слів
                    bigrams.append(bigram)
                    all_bigrams.append(bigram)
            bloblist_bigrams.append(bigrams)
            
            threegrams = []
            for threegram in tx.ngrams(3): #формування списку триграм
                if (len(diff(threegram, my_stop_words)) == 3) or ((len(diff(threegram, my_stop_words)) == 2)and(threegram[1] in my_stop_words)and(threegram[1] not in ['010','021','020','007','040','050'])):
                    threegram = list(ps.stem(w) for w in threegram) #стемінг слів
                    threegrams.append(threegram)
                    all_threegrams.append(threegram)
            bloblist_threegrams.append(threegrams)
            
    return all_words, all_bigrams, all_threegrams, bloblist, bloblist_bigrams, bloblist_threegrams

       
my_words = (open('MyStopWords.txt', 'r').read()).split() #мій словник stop-слів
my_stop_words = text.ENGLISH_STOP_WORDS.union(my_words) #формування розширеного словника stop-слів  
#print my_stop_words
 
ps = PorterStemmer() #створюється об'єкт PorterStemmer  

all_words, all_bigrams, all_threegrams, bloblist, bloblist_bigrams, bloblist_threegrams = read_files()

time_for_read_files = time.time() - t0
print 'Time_after_read_files '+str(time_for_read_files)

quantity_all_documents = len(bloblist) #кількість всіх документів у колекції
print 'quantity_all_documents '+str(quantity_all_documents)

#TF-IDF_прості слова==================================================================
scores_within_doc = list()
list_of_scores = list()
all_scores = list()
list_of_words = list()
black_list = list() #список слів, що мають tff<=lambda_
quantity_all_words = len(all_words)
lambda_ = 0.01*all_words.count('information')/quantity_all_words
for i, blob in enumerate(bloblist):
    print("W Document {}".format(i + 1))
    ##print("Top words in document {}".format(i + 1))
    ##word_value = {word: tfidf(word, blob, bloblist) for word in blob}
    #print word_value
    ##sorted_words = sorted(word_value.items(), key=lambda x: x[1], reverse=True)
    ##for word, score in sorted_words[:10]:
     ##   print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
    for word in blob:
        if (word not in list_of_words) and (word not in black_list): #(якщо розглянуте слово відсутнє в списку слів, що мають tff>lambda_)або(і це слово не в чорному списку)
            tff = TF(word, all_words, quantity_all_words, bloblist)
            #tff = tfidf(word, blob, bloblist)
            if tff > lambda_:
                scores_within_doc.append(tff)
                all_scores.append(tff)
                list_of_words.append(word)
            else:
                black_list.append(word)
        elif (word in list_of_words): #tff для даного слова вже розраховане й знаходиться у списку all_scores[list_of_words.index(word)]
            tff = all_scores[list_of_words.index(word)]
            scores_within_doc.append(tff)
            all_scores.append(tff)
            list_of_words.append(word)
    list_of_scores.append(scores_within_doc)
    scores_within_doc = []

print 'Max_A '+str(max(all_scores))
print list_of_words

time_for_tf_words = (time.time()-t0) - time_for_read_files
print 'Time_for_tf_words '+str(time_for_tf_words)
print 

#побулова графу видимості 1 ======================================================
print 'quantity_all_words '+str(quantity_all_words)
print 'list_of_words '+str(len(list_of_words))
print 'all_scores '+str(len(all_scores))
print list_of_words
print 
g1 = nx.Graph()
n = 0
for scores in list_of_scores:
    #g1 = visibility_graph2(g1, scores, n)
    g1 = horizontal_visibility_graph2(g1, scores, n)
mapping = dict(zip(g1.nodes(),list_of_words))
g1 = nx.relabel_nodes(g1,mapping)
filename = 'vg_words.graphml'
nx.write_graphml(g1,filename)
  
#TF-IDF_біграми=======================================================================
bigrams_scores_within_doc = list()
list_of_bigrams_scores = list()
all_bigrams_scores = list()
list_of_bigrams = list()
list_of_bigrams_name =list()
black_list = [] 
quantity_all_bigrams = len(all_bigrams)
print 'quantity_all_bigrams '+str(quantity_all_bigrams)
lambda_ = 0.01*all_bigrams.count(['information', 'extraction'])/quantity_all_bigrams
for i,blob in enumerate(bloblist_bigrams):
    print("B Document {}".format(i + 1))
    for bigram in blob:
        if (bigram not in list_of_bigrams) and (bigram not in black_list): #(якщо розглянуте слово відсутнє в списку слів, що мають tff>lambda_)або(і це слово не в чорному списку)
            tff = TF(bigram, all_bigrams, quantity_all_bigrams, bloblist_bigrams) 
            if tff > lambda_:
                bigrams_scores_within_doc.append(tff)
                all_bigrams_scores.append(tff)
                list_of_bigrams.append(bigram)
                list_of_bigrams_name.append(bigram[0]+'_'+bigram[1])#запис біграми одним словом
            else:
                black_list.append(bigram)
        elif (bigram in list_of_bigrams): #tff для даної біграми вже розраховане й знаходиться у списку all_bigrams_scores[list_of_bigrams.index(bigram)]
            tff = all_bigrams_scores[list_of_bigrams.index(bigram)]
            bigrams_scores_within_doc.append(tff)
            all_bigrams_scores.append(tff)
            list_of_bigrams.append(bigram)
            list_of_bigrams_name.append(bigram[0]+'_'+bigram[1])#запис біграми одним словом
    list_of_bigrams_scores.append(bigrams_scores_within_doc)
    bigrams_scores_within_doc = []

print 'Max_B' +str(max(all_bigrams_scores))
print 
time_for_tf_bigrams = (time.time()-t0) - time_for_tf_words
print 'Time_for_tf_bigrams '+str(time_for_tf_bigrams)
print

#побудова графу видимості 2 ======================================================
print 'quantity_all_bigrams '+str(quantity_all_bigrams)
print 'list_of_bigrams '+str(len(list_of_bigrams))
print 'all_bigrams_scores '+str(len(all_bigrams_scores))
print 
g2 = nx.Graph()
n = 0
for scores in list_of_bigrams_scores:
    #g2 = visibility_graph2(g2, scores, n)
    g2 = horizontal_visibility_graph2(g2, scores, n)
mapping = dict(zip(g2.nodes(),list_of_bigrams_name))
g2 = nx.relabel_nodes(g2,mapping)
filename = 'vg_bigrams.graphml'
nx.write_graphml(g2,filename)

#TF-IDF_триграми=======================================================================
threegrams_scores_within_doc = list()
list_of_threegrams_scores = list()
all_threegrams_scores = list()
list_of_threegrams = list()
list_of_threegrams_name = list()
black_list = []
quantity_all_threegrams = len(all_threegrams)
print 'quantity_all_threegrams '+str(quantity_all_threegrams)
lamda_ = 0.01*all_threegrams.count(['quantum', 'information', 'theory'])/quantity_all_threegrams
for i,blob in  enumerate(bloblist_threegrams):
    print("T Document {}".format(i + 1))
    for threegram in blob:
        if (threegram not in list_of_threegrams) and (threegram not in black_list): #(якщо розглянуте слово відсутнє в списку слів, що мають tff>lambda_)або(і це слово не в чорному списку)
            tff = TF(threegram, all_threegrams, quantity_all_threegrams, bloblist_threegrams) 
            if tff > lamda_:
                threegrams_scores_within_doc.append(tff)
                all_threegrams_scores.append(tff)
                list_of_threegrams.append(threegram)
                list_of_threegrams_name.append(threegram[0]+'_'+threegram[1]+'_'+threegram[2])#запис триграми одним словом
            else:
                black_list.append(threegram)
        elif (threegram in list_of_threegrams): #tff для даної триграми вже розраховане й знаходиться у списку all_threegrams_scores[list_of_threegrams.index(threegram)]
            tff = all_threegrams_scores[list_of_threegrams.index(threegram)]
            threegrams_scores_within_doc.append(tff)
            all_threegrams_scores.append(tff)
            list_of_threegrams.append(threegram)
            list_of_threegrams_name.append(threegram[0]+'_'+threegram[1]+'_'+threegram[2])#запис триграми одним словом
    list_of_threegrams_scores.append(threegrams_scores_within_doc)
    threegrams_scores_within_doc = []

print 'Max_C '+str(max(all_threegrams_scores))       
print  
 
time_for_tf_threegrams = (time.time()-t0) - time_for_tf_bigrams
print 'Time_for_tf_threegrams '+str(time_for_tf_threegrams)
print

#побулова графу видимості 3======================================================
print 'quantity_all_threegrams '+str(quantity_all_threegrams)
print 'list_of_threegrams '+str(len(list_of_threegrams))
print 'all_threegrams_scores '+str(len(all_threegrams_scores))
print 
g3 = nx.Graph()
n = 0
for scores in list_of_threegrams_scores:
    #g3 = visibility_graph2(g3, scores, n)
    g3 = horizontal_visibility_graph2(g3, scores, n)
mapping = dict(zip(g3.nodes(),list_of_threegrams_name))
g3 = nx.relabel_nodes(g3,mapping)

filename = 'vg_threegrams.graphml'
nx.write_graphml(g3,filename)

print 1.0*all_words.count('information')/len(all_words)
print 1.0*all_bigrams.count(['information', 'extraction'])/len(all_bigrams)
print 1.0*all_threegrams.count(['quantum', 'information', 'theory'])/len(all_threegrams)        


print 'Max_A '+str(max(all_scores))
print 
time_for_built_vg = (time.time()-t0) - time_for_tf_threegrams

print 'Time_for_read_files '+str(time_for_read_files)
print 'Time_for_tf_words '+str(time_for_tf_words)
print 'Time_for_tf_bigrams '+str(time_for_tf_bigrams)
print 'Time_for_tf_bigrams '+str(time_for_tf_threegrams)
print 'Time_for_built_vg '+str(time_for_built_vg)
print 'Time: '+str(time.time()-t0)

