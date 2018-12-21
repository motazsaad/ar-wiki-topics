import glob
import json
import os
from collections import defaultdict

from alphabet_detector import AlphabetDetector
from bs4 import BeautifulSoup
from gensim import corpora, models


def clean_text(text):
    text = BeautifulSoup(text, 'html.parser').text
    text = text.replace('&quot;', '\"')
    text = text.replace('&#39;', '\'')
    return text


def load_json_newsletters(corpus_dir):
    alphabet_detector = AlphabetDetector()

    arb_corpus = list()
    eng_corpus = list()
    ids = list()
    json_files = glob.glob(corpus_dir + '/*.json')
    print(json_files)
    for json_file in json_files:
        line = open(json_file).read()
        json_doc = json.loads(line)
        try:
            j_articles = json_doc['articles']
            for doc in j_articles:
                doc_id = doc['id']
                title = clean_text(doc['title'])
                text = clean_text(doc['body'])
                link = doc['link']
                if doc_id not in ids:
                    if 'ARABIC' in alphabet_detector.detect_alphabet(text):
                        arb_corpus.append(text)
                    else:
                        eng_corpus.append(text)
        except KeyError:
            continue
    return arb_corpus, eng_corpus


def load_json_wiki_corpus(corpus_dir):
    json_corpus = list()
    for subdir, dirs, files in os.walk(corpus_dir):
        for f in files:
            wiki_file = os.path.join(subdir, f)
            # print('reading', wiki_file)
            with open(wiki_file, encoding='utf-8') as wiki_reader:
                lines = wiki_reader.readlines()
                for line in lines:
                    json_doc = json.loads(line)
                    doc_id = json_doc['id']
                    title = json_doc['title']
                    text = json_doc['text']
                    json_corpus.append(text)
    return json_corpus
    

def load_plain_corpus(corpus_dir):
    plain_corpus = list()
    for subdir, dirs, files in os.walk(corpus_dir):
        for f in files:
            doc_file = os.path.join(subdir, f)
            # print('reading', wiki_file)
            with open(doc_file, encoding='utf-8') as file_reader:
                text = file_reader.read()
                plain_corpus.append(text)
    return plain_corpus


def build_model(corpus, corpus_name, min_freq, topics):
    print('min freq:', min_freq, 'topics:', topics)
    texts = [[word for word in d.split()] for d in corpus]
    print('texts collected')
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    
    texts = [[token for token in text if frequency[token] > min_freq] for text in texts]
    print('texts filtered')
    dictionary = corpora.Dictionary(texts)
    dictionary.save('{}_{}freq_{}topics.dict'.format(corpus_name, min_freq, topics))
    print('dictionary saved')
    corpus = [dictionary.doc2bow(text) for text in texts]
    print('gensim corpus transformed')
    tf_idf = models.TfidfModel(corpus)
    print('tf_idf modeled')
    corpus_tf_idf = tf_idf[corpus]
    print('tf_idf corpus transformed')
    # initialize an LSI transformation
    lsi = models.LsiModel(corpus_tf_idf, id2word=dictionary, num_topics=topics)
    print('lsi corpus built')
    # create a double wrapper over the original corpus: 
    # bow->tfidf->fold-in-lsi
    corpus_lsi = lsi[corpus_tf_idf]
    print('lsi corpus transformed')
    lsi.save('{}_{}freq_{}topics.lsi'.format(corpus_name, min_freq, topics)) 
    print('lsi corpus saved')
    print('done!')


# my_corpus = load_json_wiki_corpus('../arwikiExtracts/20181020/')
# build_model(my_corpus, 'ar_wiki_20181020', min_freq=7, topics=300)
#
# my_corpus = load_plain_corpus('../jsc/jsc_plain_pages_100k/')
# build_model(my_corpus, 'ar_jsc_100k', min_freq=7, topics=300)

# json newsletters
arabic_docs, english_docs = load_json_wiki_corpus('../newsletter_json')
build_model(arabic_docs, 'ar_newsletters', min_freq=3, topics=500)
