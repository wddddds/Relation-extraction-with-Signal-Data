import nltk

s = 'Will part of speech tags helpful for relation extraction task?'
s_token = nltk.word_tokenize(s)
s_pos = nltk.pos_tag(s_token)

print s_pos
print s_pos[0]
print s_pos[0][1]
print [x[1] for x in s_pos]
print ' '.join([x[1] for x in s_pos])