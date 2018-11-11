import os 
import json 
from collections import defaultdict
from gensim import corpora, models

def load_json_corpus(corpus_dir):
    corpus = list()
    for subdir, dirs, files in os.walk(corpus_dir):
        for f in files:
            wiki_file = os.path.join(subdir, f)
            #print('reading', wiki_file)
            with open(wiki_file, encoding='utf-8') as wiki_reader:
                lines = wiki_reader.readlines()
                for line in lines:
                    json_doc = json.loads(line)
                    doc_id = json_doc['id']
                    title = json_doc['title']
                    text = json_doc['text']
                    corpus.append((doc_id, title, text))
    return corpus
    
    
corpus_path = '/home/ubuntu/arwiki/arwikiExtracts/20181020/'    
wiki_corpus = load_json_corpus(corpus_path)
print('corpus loaded')
texts = [[word for word in d.split()] for i, t, d  in wiki_corpus]
print('texts collected')
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]
print('texts filtered')
dictionary = corpora.Dictionary(texts)
dictionary.save('ar_wiki_20181020.dict')
print('dictionary saved')
corpus = [dictionary.doc2bow(text) for text in texts]
print('gensim corpus transformed')
corpus_tfidf = models.TfidfModel(corpus) 
print('tfidf corpus transformed')
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=500) 
print('lsi corpus built')
corpus_lsi = lsi[corpus_tfidf] 
print('lsi corpus transformed')
lsi.save('ar_wiki_20181020.lsi') 
print('lsi corpus saved')
print('done!')
