# plotting modules
from wordcloud import WordCloud as WordCloud_
import matplotlib.pyplot as plt
# built-in utility modules
import math 
import re
import string
import time 
from operator import itemgetter
# data manupulation modules
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import pandas as pd
import numpy as np
# natural language processing modules
import nltk
from nltk.lm import MLE ,  KneserNeyInterpolated as KNI
from nltk.util import everygrams , pad_sequence
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from gensim.models import Word2Vec
import os
'''
These files might not be downloaded which could cause a crash.
In order to solve this issue files are updated in each execution.
nltk.download('stopwords')
nltk.download('punkt')
'''
try:
    with open("extended_stopwords.txt",encoding="utf-8") as stopwords_file:
        stopwords_ = stopwords_file.read().splitlines()
        stopwords_.extend(nltk.corpus.stopwords.words('turkish'))
except Exception as inst:
    stopwords_=nltk.corpus.stopwords.words('turkish')
def WordCloud(Docs,size,output_path,mode='TF',stopwords=True):
    # Necessary model is created as prescribed.
    if(mode=='TF'):
        weighting = CountVectorizer(analyzer= 'word',stop_words=(None if stopwords else stopwords_))
    elif(mode=='TFIDF' or mode=='TF-IDF'):
        weighting = TfidfVectorizer(analyzer= 'word',stop_words=(None if stopwords else stopwords_))
    else:
        raise Exception("mode can only be TF or TF-IDF")
    vecs=weighting.fit_transform(Docs)
    '''
    TF-IDF returns the term frequency of a term that occurs in a document so it is possible to observe different frequencies 
    for the same existing word. In order to solve this issue average frequencies are calculated.
    No problem for TF since if it occurs once than it will be averaged as a single instance (divided by 1.).
    '''
    weights = np.asarray(vecs.sum(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': weighting.get_feature_names(), 'weight': weights})
    '''
    weights_df.to_csv(f'{mode}.csv')
    Frequency dictionary in order to use for creating a Word Cloud.
    '''
    dict_ = {}
    for row in weights_df.iterrows():
        dict_[row[1][0]] = row[1][1]
    '''
    Plot specifications regarding on the Word Cloud is entered. Plotting specifications are assigned manually after this step since 
    this module feeds on matplotlib module.
    '''
    wordcloud = WordCloud_(width = size*100, height = size*100,background_color ='white',min_font_size = 10,max_words=200).generate_from_frequencies(dict_) 
    # Plot the WordCloud image                        
    plt.figure(figsize = (size, size), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    # Plot is saved to the given output path successfully.

    plt.savefig(os.getcwd()+"/"+output_path)
    plt.clf()
    plt.close()
    
def ZiphsPlot(Docs,zips_outputfile):
    # General frequency of a word will be contained in this dictionary.
    frequency_table = {}
    for index,Doc in enumerate(Docs):
            #Down-Casing
            Doc = Doc.lower()
            #Markup Removal -> Punctiation etc.
            Doc = re.findall(r'\w+',Doc)
            #Counting loop
            for word in Doc:
                word = word.translate(str.maketrans('','',string.punctuation+'’‘”'))
                if(word in frequency_table):
                    frequency_table[word]+=1
                else:
                    frequency_table[word]=1
    # Sorting reversely in order to obtain high ranked words in the front rows.
    sorted_dict = sorted(frequency_table.items(),key=itemgetter(1),reverse = True)
    x_axis = []
    y_axis = []
    for index,term in enumerate(sorted_dict):
        # Necessary log based operations are done and the results are spread to the proper lists. 
        x_axis.append(math.log(index+1))
        y_axis.append(math.log(term[1]))
    plt.scatter(x_axis,y_axis)
    plt.title("Zipf's Law")
    plt.xlabel("log(rank)")
    plt.ylabel("log(frequecy)")
    plt.savefig(os.getcwd()+"/"+zips_outputfile)
    plt.clf()
 
def HeapsPlot(Docs,heaps_outputfile):
    uniq = set()
    index = 1
    x_axis = []
    y_axis = []
    for Doc in Docs:
        #Down-Casing
        Doc = Doc.lower()
        #Markup Removal -> Punctiation etc.
        Doc = re.findall(r'\w+',Doc)
        #Counting loop
        for word in Doc:
            uniq.add(word)
            x_axis.append(index)
            y_axis.append(len(uniq))
            index+=1
    plt.scatter(x_axis,y_axis)
    plt.title("Heap's Law")
    plt.xlabel('term occurence')
    plt.ylabel('vocabulary size')
    plt.savefig(os.getcwd()+"/"+heaps_outputfile)

def LanguageModel(Docs,model_type="KneserNeyInterpolated",ngram=3):
    # A list for holding the pre-processed documents.
    TEXT=[]
    for Doc in Docs:
        sentences = sent_tokenize(Doc)
        # Each sentence is tokenized.
        for sentence in sentences:
            # For each word in sentence, punctuations are removed.
            words = word_tokenize(sentence.translate(str.maketrans('','',string.punctuation+'’‘”')))
            TEXT.append(words)
    # training data with padded results are generated with built-in function of nltk.
    train_data,padded = padded_everygram_pipeline(ngram,TEXT)
    if(model_type=="MLE"):
        # Maximum likehood estimation
        MODEL = MLE(ngram)
    elif(model_type=="KneserNeyInterpolated"):
        # Kneser-Ney
        MODEL = KNI(ngram)
    MODEL.fit(train_data,padded)    
    return MODEL

def generate_sentence(model,text):
    generated_list = []
    # Iterating a loop for 5 times in order to detect 5 different instances of sentence generation.
    for index in range(5):
        # Initially , adding start of a sentence and the context text to the sentence list.
        sentence = [text if len(text.split())==1 else t for t in text.split()]
        sentence.insert(0,"<s>")
        chained = model.generate(text_seed=sentence)
        sentence.append(chained)
        # If end of the sentence is captured after the final iteration of the loop, it eliminates the loop.
        while(chained!="</s>"):
            # Different grams are tried to be expected but if possible 4 ngrams are selected in every iteration.
            chained = model.generate(1,text_seed=sentence[-2:])
            sentence.append(chained)
        '''
        Resulting string is merged in order to evaluate the perplexity score and string is pushed in to the list
        with the respective perplexity result
        '''
        try:
            generated_list.append([" ".join(sentence[1:-1])+".",model.perplexity(nltk.ngrams(sentence[1:-1],n=2))])
        except Exception as err:
            print(err,sentence)
    '''
    Generated sentences are compared with their perplexity score and sorted with respect to those values
    First element of the list contains the smallest element.
    (If all the perplexity scores are same or at least some are, the first string is approximated as the result.)
    '''
    return sorted(generated_list,key=itemgetter(1))[0]

def WordVectors(Docs,dimension_size,type_of,window_size):
    TEXT=[]
    for Doc in Docs:
        sentences = sent_tokenize(Doc)
        # For each sentence in each document.
        for sentence in sentences:
            # For each word in sentence, lower case and remove the punctuations.
            words = word_tokenize(sentence.translate(str.maketrans('','',string.punctuation+'’‘”')))
            words = [word.lower() for word in words]
            TEXT.append(words)
    # Word Embedding model is generated.
    model = Word2Vec(sentences=TEXT,size = dimension_size,sg = (0 if type_of=="cbow" else 1),window=window_size,workers=4)
    return model

def word_relationship(WE,example_tuple_list,example_tuple_test):
    # array for holding every distance calculated for the word pairs
    distances = []
    for inst in example_tuple_list:
        # non-existing pairs are not processed
        if(inst[0] not in WE.wv.vocab or inst[1] not in WE.wv.vocab):
            continue
        else:
            distances.append(WE[inst[1]]-WE[inst[0]])
    # an empty vector in order to append the sum of the vectors
    feature_vec = np.zeros((WE[inst[1]].shape[0], ), dtype='float32')
    for distance in distances:
        feature_vec = np.add(feature_vec, distance)
    # if at least one of the pairs are in the vocab. take the average of the vector
    if(len(distances)>0):
        feature_vec = np.divide(feature_vec, len(distances))
    # resulting vector is calculated after the average distance of pairs is calculated
    feature_vec = np.add(WE[example_tuple_test[0]],feature_vec)
    # n closest vectors are selected.
    result = []
    for vector in WE.similar_by_vector(feature_vec,topn=10):
        if(vector[0]!=example_tuple_test[0]):
            result.append(vector)
    print(sorted(result,key=itemgetter(1),reverse=True)[:5])