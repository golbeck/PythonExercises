import nltk
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

#The collocations package provides collocation finders which by default consider all ngrams in a text as candidate collocations:
text = "I do not like green eggs and ham, I do not like them Sam I am!"
tokens = nltk.wordpunct_tokenize(text)
finder = BigramCollocationFinder.from_words(tokens)
#raw frequency scoring
scored = finder.score_ngrams(bigram_measures.raw_freq)
sorted(bigram for bigram, score in scored)

#We could otherwise construct the collocation finder from manually-derived FreqDists:
word_fd = nltk.FreqDist(tokens)
bigram_fd = nltk.FreqDist(nltk.bigrams(tokens))
finder = BigramCollocationFinder(word_fd, bigram_fd)
scored == finder.score_ngrams(bigram_measures.raw_freq)
sorted(bigram for bigram, score in scored)

#A similar interface is provided for trigrams:
finder = TrigramCollocationFinder.from_words(tokens)
scored = finder.score_ngrams(trigram_measures.raw_freq)
set(trigram for trigram, score in scored) == set(nltk.trigrams(tokens))

#We may want to select only the top n results:
sorted(finder.nbest(trigram_measures.raw_freq, 2))

#Alternatively, we can select those above a minimum score value:
sorted(finder.above_score(trigram_measures.raw_freq,1.0 / len(tuple(nltk.trigrams(tokens)))))

#Now spanning intervening words:
finder = TrigramCollocationFinder.from_words(tokens)
finder = TrigramCollocationFinder.from_words(tokens, window_size=4)
sorted(finder.nbest(trigram_measures.raw_freq, 4))

#A closer look at the finder's ngram frequencies:
sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[:10]

##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

