#use for processing the tweets
import nltk
from nltk import word_tokenize
pattern = r'''(?x)    # set flag to allow verbose regexps
    ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
    | \w+(-\w+)*        # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
    | \$\w+  # currency and percentages, e.g. $12.40, 82%
    | @\S+             # tokenize twitter mentions
    | \.\.\.            # ellipsis
    | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
    '''
##############################################################################################
##############################################################################################
##############################################################################################
#build a dictionary of contractions
contractions={'arent':'are not','cant':'cannot','couldnt':'could not','didnt':'did not','doesnt':'does not','dont':'do not','hadnt':'had not','hasnt':'has not','havent':'have not','hed':'he had', 'hell':'he will','hes':'he is','Id':'I had','Ill':'I will','Im':'I am',
'Ive':'I have','isnt':'is not','lets':'let us','mightnt':'might not','mustnt':'must not','shant':'shall not','shed':'she had',
'shell':'she will','shes':'she is','shouldnt':'should not','thats':'that is','theres':'there is','theyd':'they had','theyll':'they will',
'theyre':'they are','theyve':'they have','wed':'we had','were':'we are','weve':'we have','werent':'were not',
'whatll':'what will','whatre':'what are','whats':'what is','whatve':'what have','wheres':'where is','whos':'who had','wholl':'who will',
'whore':'who are','whos':'who is','whove':'who have','wont':'will not','wouldnt':'would not','youd':'you had',
'youll':'you will','youre':'you are','youve':'you have'}

keys_old=contractions.keys()
keys_new=[w.lower() for w in contractions.keys()]
for i in range(len(contractions)):
    contractions[keys_new[i]]=contractions[keys_old[i]]
    if not keys_new[i]==keys_old[i]:
        del contractions[keys_old[i]]
##############################################################################################
##############################################################################################
##############################################################################################
#example text
temp=['i','wouldnt','buy','this','if','i','didnt','break','it']
##############################################################################################
##############################################################################################
##############################################################################################
#swap out contractions
for i in range(len(temp)):
    if temp[i] in contractions.keys():
        temp_text=nltk.regexp_tokenize(contractions[temp[i]], pattern)
        temp[i]=temp_text[0]
        temp.insert(i+1,temp_text[1])
##############################################################################################
##############################################################################################
##############################################################################################
#remove stop words
#first construct the stopwords lexicon
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
stop_words_exclude=['against','above','below','up','down','over','under','no','nor','not']
for i in range(len(stop_words_exclude)):
    del stopwords[stopwords.index(stop_words_exclude[i])]
#process tweets and remove stop words
temp = [w for w in temp if w not in stopwords]
##############################################################################################
##############################################################################################
##############################################################################################
#negations
negations=['no','nor','not','none','never','neither','nothing']
#build up list of indices of negations
neg_index=[]
for i in range(len(temp)):
    if temp[i] in negations:
        neg_index.append(i)
#append the final index to the negation index list if it is not a negation
if len(temp) not in neg_index:
    neg_index.append(len(temp))
#process all trailing non-negation words with an 'n_' prefix
if len(neg_index)>1:
    for i in range(0,len(neg_index)-1):
        for j in range(neg_index[i]+1,neg_index[i+1]):
            temp[j]='n_'+temp[j]

#remove the negation words
temp = [w for w in temp if w not in negations]

