import re
from collections import Counter
from math import sqrt

corpus = """Ahmad ibn Hanbal was an Arab jurist and founder of the Hanbali school who is widely recognized as the
scholar who memorized the most Hadiths in Islamic history. One of the most venerated Islamic intellectual figures, Ibn Hanbal is notable for his
unmatched memorization of over one million prophetic narrations, an unprecedented
number that has never been claimed by any other muhaddith. Ibn Hanbal also compiled the largest hadith collection, al-Musnad, which has
continued to exercise considerable influence on the field of hadith studies up to the
present time, shaping the methodological framework later employed in both Sahih Bukhari and Sahih Muslim. Imam al-Dhahabi described him as the true Imam, the proof of the religion, the master
of hadith, and the leader of the Sunnah."""

def tokenize(text):
  return re.findall(r'\w+|[^\w\s]', text)

words = tokenize(corpus.lower())


counts = {}

def bi_gram(words):
  for i in range(len(words)-2):
    bigram =words[i] + ' ' + words[i + 1]
    next_token = words[i+2]
    if bigram not in counts:
      counts[bigram] = {}

    # If we've never seen this next token, .get() returns 0; either way add 1
    counts[bigram][next_token] = counts[bigram].get(next_token, 0) + 1
  return counts

counts = bi_gram(words)

def predict_next(context):
  context = tokenize(context.lower())
  bigram = context[-2] + ' ' + context[-1]

  if bigram not in counts:
    return None
  
  best_count = 0
  best_token = None

  for token , count in counts[bigram].items():
    if count > best_count:
      best_count = count
      best_token = token
  return best_token

#print(predict_next('Ahmad Ibn'))
#print(predict_next('any other'))

def autocomplete(context):
  num_tokens = 10
  res = context.lower()
  for i in range(num_tokens):
    nxt = predict_next(res)
    if nxt is not None:

      seq = res + ' ' + nxt
      print(seq,end=' ')
      seq = tokenize(seq.lower())
      res = seq[-2] + ' ' + seq[-1]
    else:
      print('sorry this was out of corpus')
      break
#autocomplete('ahmad ibn')
#autocomplete('malik ibn')

words = tokenize(corpus.lower())
vocab = [w for w in words if w.isalnum()]
#print(vocab)

def one_hot_encode(word ,vocab):
  embd = [0] * len(vocab)
  idx = vocab.index(word)
  embd[idx] = 1
  return embd

embeddings = []
for w in vocab:
  embeddings.append(one_hot_encode(w,vocab))

def ham_dist(v1,v2):
  count = 0
  for i,j in zip(v1,v2):
    if i != j:
      count += 1
  return count

#print(ham_dist(one_hot_encode('ahmad',vocab),one_hot_encode('hanbal',vocab)))

documents = [
  'Ahmad Ibn Hanbal was a Muslim Imam and a Scholar',
  'Malik Ibn Anas was a Muslim Imam and a Scholar',
  'Al Shafi\'i was a Muslim Imam and a Scholar',
]

def build_vocab_bow(documents):
  bow_vocab = []
  for doc in documents:
    for word in doc.lower().split():
      if word not in bow_vocab:
        bow_vocab.append(word)
  return bow_vocab

def bag_of_words(doc,bow_vocab):
  vector = [0] * len(vocab)
  for word in doc.lower().split():
    if word in bow_vocab:
      idx = bow_vocab.index(word)
      vector[idx] += 1 
  return vector 

bow_vocab = build_vocab_bow(documents)
bow_vectors = []
for d in documents:
  bow_vectors.append(bag_of_words(d,bow_vocab))


#print(bow_vocab)
#print(bow_vectors[0])
#print(bow_vectors[1])
#print(bow_vectors[2])


doca = 'Ahmad Ibn Hanbal defended the Sunnah and Quran'
docb = 'Malik Ibn Anas defended Hadiths and Quranic commands'
docc = 'Jahm Safwan was a kafir'


def dot_product(v1, v2):
  
  total = 0
  for a, b in zip(v1, v2):
    total += a * b
  return total

def magnitude(v):
  
  total = 0
  for x in v:
    total += x * x
  return sqrt(total)

def cosine_similarity(v1, v2):
  
  prod = dot_product(v1,v2)
  mg1 = magnitude(v1)
  mg2 = magnitude(v2)
  return prod / (mg1 * mg2)




v1 = bag_of_words(doca,vocab)
v2 = bag_of_words(docb,vocab)
v3 = bag_of_words(docc,vocab)

#print(cosine_similarity(v1,v2))
#print(cosine_similarity(v1,v3))
#print(cosine_similarity(v3,v2))

def search(query, documents):
  # Build a shared vocabulary across the query and all documents
  vocab = build_vocab_bow([query] + documents)
  query_vector = bag_of_words(query, vocab)

  results = []
  # TODO: for each document, turn it into a bag-of-words vector,
  lst =[]
  for doc in documents:
    vc = bag_of_words(doc,vocab)
    prod = cosine_similarity(vc,query_vector)
    results.append((prod,vc))

  # compute its cosine similarity with query_vector, and append
  # the (score, doc) tuple to results

  results.sort(reverse=True)
  return results

documents2 = [
  'the cat sat on the warm mat',
  'the dog chased a stick in the yard',
  'I love pizza and pasta for dinner',
  'the kitten purred while she slept',
  'my puppy loves to play with toys',
  'homemade lasagna is my favorite meal',
]

res = search('cat mat',documents2)
res2 = search('dog yard',documents2)
res3 = search('pizza dinner',documents2)

print(res[0][0])
print(res2[0][0])
print(res3[0][0])


