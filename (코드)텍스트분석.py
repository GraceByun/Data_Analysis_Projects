#!/usr/bin/env python
# coding: utf-8

# # 1. 텍스트 전처리 및 형태소 분석 후 JSON 형태로 파일 저장

# In[1]:


import csv
import ujson
from konlpy.tag import Komoran


def split_sentences(text):
    
    #특수문자를 없앰
    text = text.strip().replace(". ", ".\n").replace("? ", "?\n").replace("! ", "!\n")
    text = text.replace("#", "").replace("(", "").replace(")", "").replace("%", "")
    sentences = text.splitlines()
    
    return sentences

def get_pos(analyzer, text):
    
    morph_anals = []
    sentences = split_sentences(text)                        
    
    for sentence in sentences:
        morph_anal = analyzer.pos(sentence)                   
        morph_anals.append(morph_anal)
        
    return morph_anals

def read_text(input_file_name):        
    
    key_names = ['doc', 'review']
    data = []                        

    with open(input_file_name, "r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.reader(input_file)
        for row_num, row in enumerate(reader):
            if row_num == 0:
                continue

            reviews = {}

            for key_name, val in zip(key_names, row):
                reviews[key_name] = val

            data.append(reviews)
            print(reviews)

    return data             

def pos_review(data):  
    
    data_pos = []
    komoran  = Komoran()
    
    for reviews in data:
        
        body = reviews["review"]                        
        review_pos = get_pos(komoran, body)             
        
        reviews["review_pos"] = review_pos              
        data_pos.append(reviews)

    return data_pos                                       


def write_pos_review(output_file_name, data_pos):       
    
    with open(output_file_name, "w", encoding="utf-8") as output_file:
        for review_pos in data_pos:
            review_str = ujson.dumps(review_pos, ensure_ascii=False)
            print(review_str, file=output_file)

            
def main(): 
    
    input_file_name = r"FinalData.csv"
    output_file_name = r"FinalData_pos.json"
    
    data = read_text(input_file_name)                                          
    data_pos = pos_review(data)                                           
    write_pos_review(output_file_name, data_pos)                          
            
main()


# # 2. 확률기반 특징표현_Word2vec_TSNE
# 
# 

# In[2]:


import ujson
import gensim
from gensim.models import Word2Vec
import pandas as pd
import re

POS_KEY = "review_pos"

#komoran 에서 일반명사, 고유명사, 형용사, 일반부사, 접속부사만 포함하였지만 동사는 알아보기 힘들어서 제외하였다. 

FEATURE_POS = ["NNG",'NNP', "VA", "MAG",'MAJ']

def read_documents(input_file_name):
    """주어진 이름의 파일에서 문서집합을 읽어서 돌려준다."""
    
    print("Reading documents.")
    
    documents = []

    with open(input_file_name, "r", encoding="utf-8") as input_file:
        for line in input_file:
            doc = ujson.loads(line)
            
            words = []            
            for sent_anal in doc[POS_KEY]:            
                for word, pos in sent_anal:
                    if pos not in FEATURE_POS:
                        continue
                        
                    words.append(word)
        
            documents.append(words)   
    
    return documents

def get_word_vectors(documents):

    print("Building word vectors.")    
    model = Word2Vec(documents, window=3, min_count=5, size=100, workers=2, sample=1e-3, sg=1)  # sg=1 을 지정하면 skip gram 
    word_vectors = model.wv
    
    return word_vectors


def main():  
    
    input_file_name = r"FinalData_pos.json"
    documents = read_documents(input_file_name)    
    word_vectors = get_word_vectors(documents)
    
    word_vectors.save_word2vec_format("skipgram_100feaures_3window_5count") 
    print("All work is done.")

# 실행

main()


# In[4]:


import sys
import ujson
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt

POS_KEY = "review_pos"
FEATURE_POS = ["NNG",'NNP', "VA", "MAG",'MAJ']


def read_documents(input_file_name):
    
    print("Reading documents.")
    
    documents = []

    with open(input_file_name, "r", encoding="utf-8") as input_file:
        for line in input_file:
            doc = ujson.loads(line)
            
            words = []            
            for sent_anal in doc[POS_KEY]:            
                for word, pos in sent_anal:
                    if pos not in FEATURE_POS:
                        continue
                        
                    words.append(word)
                    
            documents.append(words)   
    
    return documents


def get_word_vectors(documents):

    print("Building word vectors.")    
    
    model = Word2Vec(documents, window=5, min_count=100, size=100, workers=2, sample=1e-3)  # CBOW 
    word_vectors = model.wv
    
    return word_vectors


    
def set_font():
    
    if sys.platform.startswith("win"):
        matplotlib.rc("font", family="Malgun Gothic")
    elif sys.platform.startswith("darwin"):
        matplotlib.rc("font", family="AppleGothic")   

    
def tsne_plot(model):
    
    print("Building and displaying t-SNE.")
    
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    set_font()
    
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
    
    
def main():
    
    input_file_name = r"FinalData_pos.json"
    documents = read_documents(input_file_name)
    word_vectors = get_word_vectors(documents)
    tsne_plot(word_vectors)    

# 실행

main()


# # 3. 빈도 기반 TF-IDF term-document matrix를 생성

# In[3]:


from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import ujson

#komoran 에서 일반명사, 고유명사, 형용사, 일반부사, 접속부사만 포함하였지만 동사는 알아보기 힘들어서 제외하였다. 

FEATURE_POS = ["NNG",'NNP', "VA", "MAG",'MAJ']

POS_KEY = "review_pos"

def read_documents(input_file_name):
    
    documents = []

    with open(input_file_name, "r", encoding="utf-8") as input_file:
        
        for line in input_file:
            json_obj = ujson.loads(line)
            text_pos = json_obj[POS_KEY]
            
            words = []
            
            for sent_pos in text_pos:
                for word, pos in sent_pos:
                    if pos not in FEATURE_POS:
                        continue

                    words.append(word)

            document = " ".join(words)

            documents.append(document)
    documents = np.asarray(documents)
    #print(documents)
    return documents

    

# main

input_file_name = r"FinalData_pos.json"
output_file_name = "tfidf_pos_FinalData_100.txt"

documents = read_documents(input_file_name)

vectorizer = TfidfVectorizer(tokenizer=str.split, max_features =100)

doc_term_mat = vectorizer.fit_transform(documents)   


count = vectorizer.fit_transform(documents).toarray().sum(axis=0)

idx = np.argsort(-count)

count = count[idx]


feature_name = np.array(vectorizer.get_feature_names())[idx]

with open(output_file_name, "w", encoding="utf-8") as output_file:
    doc_num = len(doc_term_mat.toarray())
    for i in range(doc_num):                            
        for j in doc_term_mat[i]:
            for i1, j in zip(j.indices, j.data):
                print("{}\t{}\t{}".format(i, vectorizer.get_feature_names()[i1], j), file = output_file)


# # 4. 트리맵

# In[4]:


import squarify
import sys
import matplotlib
import matplotlib.pyplot as plt


def set_font():
     
    if sys.platform in ["win32", "win64"]:
        font_name = "malgun gothic"
      
    elif sys.platform == "darwin":
        font_name = "AppleGothic"
        
    matplotlib.rc("font", family=font_name)


def draw_treemap(feature_name, count):
    
    set_font()
    squarify.plot(sizes=count, label=feature_name, alpha=0.4)
    
    plt.axis("off")
    
    # 현재 폴더에 tdf_treemap.png 파일명으로 저장, dpi는 그림 해상도 지정 
    # bbox_inches='tight'는 그림을 둘러싼 여백 제거 
    plt.savefig('tdf_treemap.png', dpi= 400, bbox_inches='tight');
    plt.show() 
    

draw_treemap(feature_name, count)


# # 5. 워드 클라우드

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from PIL import Image

def draw_wordcloud(keywords):
    
    wordcloud = WordCloud()
    font_path = '/Users/grace/Desktop/기말프로젝트/MalgunGothic.ttf'
    wordcloud = WordCloud(font_path = font_path, width = 1200, height = 800, background_color="black")

    wordcloud = wordcloud.generate_from_frequencies(keywords)

    fig = plt.figure(figsize=(12,9))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

    fig.savefig('wordcloud_black.png')    


# main   
keywords = {}

for word, freq in zip(feature_name, count):
    keywords[word] = freq            
#print(keywords)

draw_wordcloud(keywords)    


# # 6. LDA를 이용한 토픽 모델

# In[12]:


#단계1: 전처리 후 텍스트 파일 불러오기

import ujson

FEATURE_POS = ["NNG",'NNP', "VA", "MAG",'MAJ']

POS_KEY = "review_pos"

def read_documents(input_file_name):
    
    documents = []

    with open(input_file_name, "r", encoding="utf-8") as input_file:
        for line in input_file:
            morphs = []
            json_obj = ujson.loads(line)

            for sent_anal in json_obj[POS_KEY]:
                for word, pos in sent_anal:
                    if pos not in FEATURE_POS:
                        continue
                        
                    morphs.append(word)

            documents.append(morphs)
            

    return documents

# main

input_file_name = r"FinalData_pos.json"

documents = read_documents(input_file_name)


print(len(documents))


# In[22]:


#단계2-1: Term-document matrix 생성 -BOW 기준(TF 기준)

from gensim import corpora
from gensim import models


dictionary = corpora.Dictionary(documents)

n_items = len(dictionary)

dictionary.values()

#for key, value in dictionary.items():
   # print(key, value)


# In[20]:


#문서와 빈도
corpus_tf = []

for document in documents:
    bow = dictionary.doc2bow(document)
    corpus_tf.append(bow)


# In[13]:


#단계2-2: Term-document matrix 생성 - TF-IDF 기준

from gensim import corpora
from gensim import models

dictionary = corpora.Dictionary(documents)

n_items = len(dictionary)
print(n_items)


# In[14]:


corpus= []

for text in documents:
    bow = dictionary.doc2bow(text)
    corpus.append(bow)

tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus] 

print(corpus_tfidf)


# In[23]:


#단계3: LDA model 생성

NUM_TOPICS = 10  
# 토픽 개수 생성
lda_model_tf = models.ldamodel.LdaModel(corpus_tf, num_topics=NUM_TOPICS, id2word=dictionary)

lda_model_tfidf = models.ldamodel.LdaModel(corpus_tfidf, num_topics=NUM_TOPICS, id2word=dictionary)   # 토픽모델 = LDA


# In[24]:


from pprint import pprint

pprint(lda_model_tf.print_topics())


# In[25]:


#단계4: LDA model 결과 출력

def print_document_topics(lda_model_tf, corpus_tf):
    """주어진 토픽 모델링 결과와 문서 어휘 행렬에서 문서별 토픽 분포를 출력한다."""
    
    for doc_num, doc in enumerate(corpus_tf):
        topic_probs = lda_model_tf[doc]
        print("Doc num: {}".format(doc_num))

        for topic_id, prob in topic_probs:
            print("\t{}\t{}".format(topic_id, prob))
            

        print("\n")   
        
print_document_topics(lda_model_tf, corpus_tf)


# In[26]:


def print_document_topics(lda_model_tfidf, corpus_tfidf):
    """주어진 토픽 모델링 결과와 문서 어휘 행렬에서 문서별 토픽 분포를 출력한다."""
    
    for doc_num, doc in enumerate(corpus_tfidf):
        topic_probs = lda_model_tfidf[doc]
        print("Doc num: {}".format(doc_num))

        for topic_id, prob in topic_probs:
            print("\t{}\t{}".format(topic_id, prob))
            
        # break

        print("\n")   
        
print_document_topics(lda_model_tfidf, corpus_tfidf)


# In[27]:


import pyLDAvis.gensim

vis_data = pyLDAvis.gensim.prepare(lda_model_tfidf, corpus_tf, dictionary)

pyLDAvis.save_html(vis_data, "FinalData_tfidf_topic_model.html")            


# # 7. 텍스트 네트워크 분석

# In[98]:


from operator import itemgetter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import ujson


FEATURE_POS = ["NNG",'NNP', "VA", "MAG",'MAJ']

POS_KEY = "review_pos"

def read_documents(input_file_name):
     
    documents = []

    with open(input_file_name, "r", encoding="utf-8") as input_file:
        for line in input_file:
            json_obj = ujson.loads(line)
            text_pos = json_obj[POS_KEY]
            
            words = []
            
            for sent_pos in text_pos:
                for word, pos in sent_pos:
                    if pos not in FEATURE_POS:
                        continue

                    words.append(word)
            
            document = " ".join(words)
            documents.append(document)
    
    documents = np.asarray(documents)
    
    return documents

# main

input_file_name = r"FinalData_pos.json"
documents = read_documents(input_file_name)


# In[99]:


vectorizer = CountVectorizer(tokenizer=str.split, binary=True, max_features=500)
doc_term_mat = vectorizer.fit_transform(documents)   
words= vectorizer.get_feature_names()
print(doc_term_mat )


# In[100]:


from itertools import combinations
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

"""주어진 문서-어휘 행렬부터 어휘 공기 행렬을 생성하여 돌려준다."""
#단어가 몇번 나왔는지 알기 위해서 word_cooc_mat 를 구해준다.
word_cooc_mat = doc_term_mat.T * doc_term_mat
word_cooc_mat.setdiag(0)
print(word_cooc_mat)


# In[101]:


"""주어진 co-occurence matrix 에 대하여 word similirity matrix을 구하여 돌려준다."""
SIM_MEASURE = "correlation"

#단어끼리 얼마나 가까운지  단어끼러 얼마나 동시에 등장했는 지 유사도를 pdist를 통해 구한다 
word_sim_mat = pdist(word_cooc_mat.toarray(), metric=SIM_MEASURE)
word_sim_mat = squareform(word_sim_mat)


# In[102]:


"""주어진 어휘 유사도 행렬을 정렬하여 출력한다."""
    
def print_word_sim_mat(output_file_name, word_sim_mat, words):
    
    word_sims = []
    
    for i, j in combinations(range(len(words)), 2):
        sim = word_sim_mat[i, j]
        
        if sim == 0:
            coontinue
            
        word_sims.append((words[i], words[j], sim))
        
    with open(output_file_name, "w", encoding="utf-8") as output_file:
        for word_i, word_j, sim in sorted(word_sims, key=itemgetter(2), reverse=True):
            print("{}\t{}\t{}".format(word_i, word_j, sim), file=output_file)
            
# main

output_file_name = "movie.wcr.txt"

print_word_sim_mat(output_file_name, word_sim_mat, words)


# In[103]:


#어휘 공기(단어 동시 등장) 네트워크 생성

import sys
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community


NUM_WORD_COOCS = 50

def get_sorted_word_sims(word_sim_mat, words):   
    """주어진 어휘 유사도 행렬을 정렬하여 출력한다."""

    word_sims = []
    
    for i, j in combinations(range(len(words)), 2):
        sim = word_sim_mat[i, j]
        
        if sim == 0:
            coontinue
            
        word_sims.append((words[i], words[j], sim))
        
    sorted_word_sims = sorted(word_sims, key=itemgetter(2), reverse=True)
    
    return sorted_word_sims

def build_word_sim_network(sorted_word_sims):
    """어휘 유사도 네트워크를 생성하여 돌려준다."""
    
    G = nx.Graph()

    for word1, word2, sim in sorted_word_sims[:NUM_WORD_COOCS]:
        G.add_edge(word1, word2, weight=sim)
        

    return G

  
def get_community_colors(G):
    """주어진 그래프에 대하여 커뮤니티 탐지를 수행하여 색상 배정을 하여 돌려준다."""

    nodes = nx.nodes(G)
    colors = ["salmon", "gold", "lightblue", "orange", "aquamarine"]
    clusters = list(community.asyn_fluidc(G, 5))
    node_colors = []

    for node in nodes:
        for i, cluster in enumerate(clusters):
            if node in cluster:
                node_colors.append(colors[i])
                break
                
    return node_colors


def draw_network(G):
    """어휘 공기 네트워크를 화면에 표시한다."""
    
    node_colors = get_community_colors(G)
    font_name = get_font_name()

    layouts = {'spring': nx.spring_layout(G), 
                'kamada_kawai':nx.kamada_kawai_layout(G), 
               'spectral':nx.spectral_layout(G), 
               'shell':nx.shell_layout(G), 
               'circular':nx.circular_layout(G),
               'random':nx.random_layout(G)
          }

    f, axes = plt.subplots(3, 2)
    f.set_size_inches((12, 18)) 
    

    ## 각 axis마다 그림을 따로 그려줌
    for i, kv in enumerate(layouts.items()):
        title, pos, ax = kv[0], kv[1], axes[i//2][i%2]
        print(ax)
        nx.draw_networkx(G, kv[1], ax=ax,
            node_size=500,
            node_color=node_colors,
            font_family=font_name)
        
        ax.set_title("{} layout".format(title), fontsize=20)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("fully_network.png")
    plt.show()

    
def get_font_name():
    """플랫폼별로 사용할 글꼴 이름을 돌려준다."""
    
    if sys.platform in ["win32", "win64"]:
        font_name = "malgun gothic"
    elif sys.platform == "darwin":
        font_name = "AppleGothic"
        
    return font_name


def main():
    """어휘 유사도 행렬을 구성한 뒤 이를 네트워크로 시각화한다."""
    
    sorted_word_sims = get_sorted_word_sims(word_sim_mat, words)
    G = build_word_sim_network(sorted_word_sims)
    draw_network(G)
    
    
# 실행
main()


# In[104]:


#최소 신장 트리 기반의 어휘 공기(단어 동시 등장) 네트워크 생성

import sys
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community


NUM_WORD_COOCS = 50

def get_sorted_word_sims(word_sim_mat, words):   
    """주어진 어휘 유사도 행렬을 정렬하여 출력한다."""

    word_sims = []
    
    for i, j in combinations(range(len(words)), 2):
        sim = word_sim_mat[i, j]
        
        if sim == 0:
            coontinue
            
        word_sims.append((words[i], words[j], sim))
        
    sorted_word_sims = sorted(word_sims, key=itemgetter(2), reverse=True)
    
    return sorted_word_sims

def build_word_sim_network(sorted_word_sims):
    """어휘 유사도 네트워크를 생성하여 돌려준다."""
    
    G = nx.Graph()

    for word1, word2, sim in sorted_word_sims[:NUM_WORD_COOCS]:
        G.add_edge(word1, word2, weight=sim)
    
    #이 함수를 쓰면 최소신장트리를 적용시킬 수 있다
    T = nx.minimum_spanning_tree(G)

    return T

  
def get_community_colors(G):
    """주어진 그래프에 대하여 커뮤니티 탐지를 수행하여 색상 배정을 하여 돌려준다."""

    nodes = nx.nodes(G)
    colors = ["salmon", "gold", "lightblue", "orange", "aquamarine"]
    clusters = list(community.asyn_fluidc(G, 5))
    node_colors = []

    for node in nodes:
        for i, cluster in enumerate(clusters):
            if node in cluster:
                node_colors.append(colors[i])
                break
                
    return node_colors


def draw_network(G):
    """어휘 공기 네트워크를 화면에 표시한다."""
    
    node_colors = get_community_colors(G)
    font_name = get_font_name()

    layouts = {'spring': nx.spring_layout(G), 
                'kamada_kawai':nx.kamada_kawai_layout(G), 
               'spectral':nx.spectral_layout(G), 
               'shell':nx.shell_layout(G), 
               'circular':nx.circular_layout(G),
               'random':nx.random_layout(G)
          }

    f, axes = plt.subplots(3, 2)
    f.set_size_inches((12, 18)) 
    

    ## 각 axis마다 그림을 따로 그려줌
    for i, kv in enumerate(layouts.items()):
        title, pos, ax = kv[0], kv[1], axes[i//2][i%2]
        print(ax)
        nx.draw_networkx(G, kv[1], ax=ax,
            node_size=500,
            node_color=node_colors,
            font_family=font_name)
        
        ax.set_title("{} layout".format(title), fontsize=20)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("spanning_network.png")
    plt.show()

    
def get_font_name():
    """플랫폼별로 사용할 글꼴 이름을 돌려준다."""
    
    if sys.platform in ["win32", "win64"]:
        font_name = "malgun gothic"
    elif sys.platform == "darwin":
        font_name = "AppleGothic"
        
    return font_name


def main():
    """어휘 유사도 행렬을 구성한 뒤 이를 네트워크로 시각화한다."""
    
    sorted_word_sims = get_sorted_word_sims(word_sim_mat, words)
    T = build_word_sim_network(sorted_word_sims)
    draw_network(T)
    
    
# 실행
main()


# In[ ]:




