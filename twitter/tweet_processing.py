
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
temp=['this','is','test','werent','mustnt','again','help']
for i in range(len(temp)):
    if 