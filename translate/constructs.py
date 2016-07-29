

class CandidateObject:
    def __init__(self,n, f, c, s, slb, cs, f_rank, c_rank, s_rank, slb_rank, cs_rank, bigram_p, bigram_rank, trigram_p, trigram_rank, ngram_p, ngram_rank, avg_rank):
        self.name = n
        self.frequency = f
        self.complexity = c
        self.similarity = s
        self.syllables = slb
        self.context_similarity = cs
        self.f_rank = f_rank
        self.c_rank = c_rank
        self.s_rank = s_rank
        self.slb_rank = slb_rank
        self.cs_rank = cs_rank
        self.bigram_p = bigram_p
        self.bigram_rank = bigram_rank
        self.trigram_p = trigram_p
        self.trigram_rank = trigram_rank
        self.ngram_p = ngram_p
        self.ngram_rank = ngram_rank
        self.avg_rank = avg_rank

    def __str__(self):
        return self.name + ", " + self.frequency + ", " + self.similarity + ", " + self.syllables



class ResultObject:

    def __init__(self, w,f, r,c):
        self.original_word = w
        self.frequency = f
        self.replace_with = r
        self.candidate_list = c # a list of candidate objects




class BluemixConceptKeywordObject:

    def __init__(self, n, r):
        self.name = n
        self.relevance = r



class ComplexWord:

    def __init__(self, word, freq, numOfSyllables, length):
        self.name = word
        self.freq = freq
        self.numOfSyllables = numOfSyllables
        self.length = length
        self.link = "http://www.thesaurus.com/browse/" + word + "?s=t"