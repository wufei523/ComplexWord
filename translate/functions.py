import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import sent_tokenize
from nltk.corpus import brown
from nltk.corpus import wordnet
import os
from os.path import join
import numpy as np
import gensim, logging
from nltk.wsd import lesk
import scipy.stats as ss

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def checkPos(word):
	pos = ['NN', 'NNS', 'JJ', 'RB', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
	if word in pos:
		return True
	else:
		return False

def readSetting():
	path = os.getcwd()
	with open(join(path, 'setting'), 'r') as f:
		setting = f.readlines()
	return setting

def readDoc():
	path = os.getcwd()
	p = open(join(path, 'doc'), 'r')
	text = [line.decode('utf-8').strip() for line in p.readlines()]
	p.close()
	return '\n'.join(text)

def getTopwords():
	s = readDoc()
	setting = readSetting()
	model_path = setting[0]
	model = gensim.models.Word2Vec.load_word2vec_format(model_path, binary=True)
	text = nltk.word_tokenize(s)
	tags = nltk.pos_tag(text)
	res = ''
	for tag in tags:
		word = tag[0]
		if checkPos(tag[1]):
			if word in model:
				res = res + word + ': ' 
				top_words = model.most_similar(positive=[word], topn = 10)
				for w in top_words:
					res = res + '(' + w[0] + ',' + str(w[1]) + ')'
				res = res + '\n'
			
	path = os.getcwd()
	p = open(join(path, 'topwords'), 'w')
	p.write(res.encode('utf-8'))
	p.close()

def samePos(word1, word2):
	return nltk.pos_tag([word1])[0][1] == nltk.pos_tag([word2])[0][1]

def freq_diff(freq1, freq2):
	return freq1*1.2 < freq2

def getSynonmys(word):
	res = []
	syns = wordnet.synsets(word)
	for s in syns:
		for l in s.lemmas:
			res.append(l.name)
	return list(set(res))


def get_Candidate_Frequency_from_wordnet(word, tag, context):
    wordnet_tag = get_wordnet_pos(tag)
    sent = list(context)
    #print("context is")
    #print(sent)
	#syns = lesk(sent, word, wordnet_tag)
    syns = lesk(sent, word)

    res = []
    if syns:
        for l in syns.lemmas():
            if l:
                lemma_name = str(l.name())
                st = LancasterStemmer()
                if ((st.stem(word) != st.stem(lemma_name))):
                    res.append(lemma_name)


    candidate_list = list(set(res))
    #print("for word: " + word)
    #print(candidate_list)

    '''
    if candidate_list:
        for c in candidate_list:
            allsyns1 = set(ss for ss in wordnet.synsets(c))
            print(allsyns1)
            allsyns2 = set(ss for ss in wordnet.synsets(word))
            print(allsyns2)
            best = max((wordnet.wup_similarity(s1, s2) or 0, s1, s2) for s1, s2 in product(allsyns1, allsyns2))
            print(best)
        '''


    '''
    for c in candidate_list:
        wordsFromList1 = wordnet.synsets(word)
        wordsFromList2 = wordnet.synsets(c)
        if wordsFromList1 and wordsFromList2:  # Thanks to @alexis' note
            s = wordsFromList1[0].wup_similarity(wordsFromList2[0])
            similarity_list.append(s)
    '''
    return candidate_list


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''



from nltk.corpus import cmudict
d = cmudict.dict()

def countSyllables(word):

    try:
        num_of_syllables = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]]
        return  min(num_of_syllables)
    except:
        # print(word + " not found in cmudict, so num_of_syllables=0")
        return 999


from translate.alchemyapi import AlchemyAPI
alchemyapi = AlchemyAPI()
from translate import constructs

def getBluemixConcept(s):

	alchemyapi_concept_response = alchemyapi.concepts('text', s)
	bluemix_concept_str_list = []
	bluemix_concept_obj_list = []
	# print("BLUE MIX Concept")
	if alchemyapi_concept_response['status'] == 'OK':
		for concept in alchemyapi_concept_response['concepts']:
			bluemix_concept_str_list.append(str(concept['text']))
			bluemix_concept_obj_list.append(constructs.BluemixConceptKeywordObject(str(concept['text']), str(concept['relevance'])))
		#print("Bluemix Concept: " + str(bluemix_concept_str_list))

	else:
		print('Error in concept tagging call: ', alchemyapi_concept_response['statusInfo'])

	return bluemix_concept_str_list, bluemix_concept_obj_list



def getBluemixKeyword(s):

	alchemyapi_keyword_response = alchemyapi.keywords('text', s, {'sentiment': 1})
	bluemix_keyword_str_list = []
	bluemix_keyword_obj_list = []
	# print("BLUE MIX keyword")
	if alchemyapi_keyword_response['status'] == 'OK':
		for keyword in alchemyapi_keyword_response['keywords']:
			bluemix_keyword_str_list.append(str(keyword['text']))
			bluemix_keyword_obj_list.append(
				constructs.BluemixConceptKeywordObject(str(keyword['text']), str(keyword['relevance'])))
			#print("Bluemix keyword: " + str(bluemix_keyword_str_list))
	else:
		print('Error in keyword extaction call: ', alchemyapi_keyword_response['statusInfo'])


	return bluemix_keyword_str_list, bluemix_keyword_obj_list


def rankList(list):

	return ss.rankdata(list)

def rankListReverse(list):
	return (len(list) - ss.rankdata(list) + 1)


def sentenceCapitalizer (string1: str):
    sentences = string1.split(". ")
    sentences2 = [sentence[0].capitalize() + sentence[1:] for sentence in sentences]
    string2 = '. '.join(sentences2)
    return string2