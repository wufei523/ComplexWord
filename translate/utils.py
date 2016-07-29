import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import sent_tokenize
from nltk.corpus import brown
from nltk.corpus import stopwords
import os
from os.path import join
import numpy as np
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from translate import functions as function
from translate import constructs
from gensim.models.word2vec import Word2Vec
import os
import json
import string
from os.path import join
import scipy.stats as ss
import math
from nltk import bigrams
from nltk import trigrams
from translate.alchemyapi import AlchemyAPI
alchemyapi = AlchemyAPI()


news_text = brown.words()
emma = nltk.corpus.gutenberg.words()
r = nltk.corpus.reuters.words()
#form corpus
corpus = emma + news_text
corpus += r
#lower case frequency dictionary based on brown + emma + reuters
fdist = nltk.FreqDist(w.lower() for w in corpus)

#language model
cfreq_corpus_2gram = nltk.ConditionalFreqDist(nltk.bigrams(w.lower() for w in corpus))
cprob_corpus_2gram = nltk.ConditionalProbDist(cfreq_corpus_2gram, nltk.MLEProbDist)
len_corpus = len(corpus)

# stopwords
stop_words = set(stopwords.words('english'))




def simplify(passage, min_frequent=100, min_frequent_diff = 1.2, min_similarity = 0.5, top_n_elements=20):


    sentences_tokenized_list = sent_tokenize(passage)
    simplified_passage = ''
    w2v_result_passage = []
    bluemix_result_passage = []
    wordnet_result_passage = []

    complex_word_object_list = []


    # process each sentence in passage
    for s in sentences_tokenized_list:


        print("")
        print("THIS sentence: " + str(s))



        tokenized_sentence = nltk.word_tokenize(s)
        tags = nltk.pos_tag(tokenized_sentence)#part of speech tag


        #go through each tag/go through token
        # tag[0] is word, tag[1] is tag
        for tag in tags:
            word_index = tags.index(tag)

            word_processing_original = tag[0]
            word_processing_lower = word_processing_original.lower()



            #addition function: find complex word
            complex_word_flag = False
            if isComplexWord(word_processing_lower) is not None:
                complex_word = isComplexWord(word_processing_lower)
                complex_word_object_list.append(complex_word)
                complex_word_flag = True


    simple = "No data here"
    return simple, complex_word_object_list




def needToProcess(word, pos, bluemix_list, min_freq):

    debugging = False
    word = word.lower()
    # only change tag = ['NN', 'NNS', 'JJ', 'RB', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    # correct pos
    # from translate import functions as f
    condition1 = function.checkPos(pos)
    # not in bluemix list
    condition2 = str(word) not in bluemix_list
    # in w2v model
    condition3 = str(word) in w2v_model
    # low frequency
    condition4 = fdist[str(word)] < min_freq
    # not 've
    condition5 = "'" not in word
    # not in stopwords
    condition6 = str(word) not in stop_words

    if debugging:
        print(" ")
        print("This word: " + str(word))
        print("Pos?        "+str(condition1))
        print("in Bluemix? "+str(condition2))
        print("in w2v?     "+str(condition3))
        print("hard word?  "+str(condition4))

    if condition1 and condition2 and condition3 and condition4 and condition5 and condition6:
        return True
    else:
        return False



def isValidCandidate():
    return 0


def getContextSimilarity(c, tokenized_sentence):

    #print(" ")
    #print("For this candidate: " + str(c))
    sum = 0
    count = 0
    for w in tokenized_sentence:
        w = w.lower()
        if w not in string.punctuation and w in w2v_model:
            #print("between: " + str(c) + " and " + str(w))
            sum += w2v_model.similarity(c, w)
            count += 1

    return sum/count



def unigram_prob(word):
    return fdist[word]/len_corpus





def getNgramProbability(bi_grams, tri_grams, t, candidate):

    #print("NOW doing this word: " + str(t))
    bi_grams_for_this = [bi_token for bi_token in bi_grams if t in bi_token]
    #print("bi: ")
    #print(bi_grams_for_this)
    num_bi_grams = len(bi_grams_for_this)
    bi_sum = 0

    for bi in bi_grams_for_this:
        bi = list(bi)
        if bi[0] == t:
            bi[0] = candidate
        if bi[1] == t:
            bi[1] = candidate

        # P(how do) = P(how) * P(do|how)
        # print(bi)
        bi_sum += unigram_prob(bi[0]) * cprob_corpus_2gram[bi[0]].prob(bi[1])
    avg_bi_gram_prob = bi_sum / num_bi_grams
    # print("bi gram prob: " + str(avg_bi_gram_prob))


    tri_grams_for_this = [tri_token for tri_token in tri_grams if t in tri_token]
    #print("tri: ")
    #print(tri_grams_for_this)
    num_tri_grams = len(tri_grams_for_this)
    tri_sum = 0


    for tri in tri_grams_for_this:
        tri = list(tri)
        if tri[0] == t:
            tri[0] = candidate
        if tri[1] == t:
            tri[1] = candidate
        if tri[2] == t:
            tri[2] = candidate
        # P(how do you) = P(how) * P(do|how) * P(you|do)
        # print(tri)
        tri_sum += unigram_prob(tri[0]) * cprob_corpus_2gram[tri[0]].prob(tri[1]) * cprob_corpus_2gram[tri[1]].prob(tri[2])
    avg_tri_gram_prob = tri_sum / num_tri_grams
    #print("tri gram prob: " + str(avg_tri_gram_prob))

    avg_ngram_prob = (bi_sum + tri_sum) / (num_bi_grams + num_tri_grams)
    #print("Ngram prob: " + str(avg_ngram_prob))

    return avg_bi_gram_prob, avg_tri_gram_prob, avg_ngram_prob




def isComplexWord(word, min_freq=50):

    if word not in string.punctuation:

        word = word.lower()
        freq = fdist[str(word)]
        NumOfSyllables = function.countSyllables(word)
        length = len(word)

        condition1 =  freq < min_freq
        condition2 = NumOfSyllables >= 3
        condition3 = length >= 11

        #testing conditions
        condition4 = freq<min_freq*2 and freq>=min_freq and NumOfSyllables >= 4
        condition5 = freq<min_freq and NumOfSyllables >=3
        condition6 = freq<20

        condition7 = "'" not in word

        complex_word = constructs.ComplexWord(word, freq, NumOfSyllables, length)

        condition8 = str(word) not in stop_words

        condition9 = word.isalpha()


        if (condition4 or condition5 or condition3 or condition6) and condition7 and condition8 and condition9:
            return complex_word
        else:
            return None
    else:
        return None




def filterCandidateList(candidate_list_w2v, freq_list_w2v, similarity_list_w2v, syllables_list_w2v, min_similarity, word_processing_frequency, word_processing_lower, word_processing_slb):

    debugging = False
    if debugging:
        print("for this word " + str(word_processing_lower))
        print("limits")
        print(word_processing_frequency)
        print(min_similarity)
        print(word_processing_slb)
        print(" ")

    #candidate_list_w2v, freq_list_w2v, similarity_list_w2v, syllables_list_w2v = zip(*c_f_list_w2v)
    length = len(candidate_list_w2v)
    c_list = []
    f_list = []
    s_list = []
    slb_list = []

    for x in range(0,length):
        c = candidate_list_w2v[x]
        f = freq_list_w2v[x]
        s = similarity_list_w2v[x]
        slb = syllables_list_w2v[x]

        #keep candidate if meets all conditions
        condition1 = f > word_processing_frequency
        condition2 = s > min_similarity
        condition3 = slb <= word_processing_slb
        condition4 = function.samePos(word_processing_lower,c)
        if debugging:
            print("for this candidate " + str(c))
            print(str(condition1) + " " + str(condition2) + " " + str(condition3) + " " + str(condition4))
            print("")


        if condition1 and condition2 and condition3 and condition4:
            c_list.append(c)
            f_list.append(f)
            s_list.append(s)
            slb_list.append(slb)

    return c_list, f_list, s_list, slb_list




