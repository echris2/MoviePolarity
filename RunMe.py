# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 12:00:03 2020

@author: eachr
"""

import os
import json
import csv
import sys
import pandas as pd
import numpy as np
from ast import literal_eval
from nltk.probability import FreqDist
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

##Set path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
sys.path.append(dname + '/Scripts')
print(os.getcwd())

from Scripts import db_ec
from Scripts import MovieRead
from Scripts import DataPull

#set locations to build database and to print various CSVs
db_loc = "Data/Movies.db"
meta_csv_loc = "Data/movies_metadata.csv"
rating_csv_loc = "Data/ratings.csv"
keyword_csv_loc = "Data/keywords.csv"
links_loc = "Data/links.csv"

loadask = input('Do you need to build and load the database? [Y/N]')
loadask = db_ec.str2bool(loadask)

if loadask == True:

    #create database
    db_ec.create_db_connection(db_loc)
    print("DB Created")
    newRatingloc, newMetaloc,newLinkloc = MovieRead.alignIds(meta_csv_loc, rating_csv_loc, links_loc)
    print("Databases Synced")
    MovieRead.movieRatings_load(db_loc, newRatingloc)
    print("Ratings loaded")
    MovieRead.movieMeta_load(db_loc, newMetaloc)
    print("Metadata Loaded")
    MovieRead.define_Controversy(db_loc)
    print("Controversiality Metrics defined")
    MovieRead.movieKeywords_load(db_loc, keyword_csv_loc)
    print("Keywords loaded")
    
    stat= '''SELECT *
            FROM metadata m
            INNER JOIN contRatings r ON m.ratingId = r.movieId
            WHERE m.release_date > '1994-12-31' AND r.TotalVotes > 30;'''
            
    analysis_file = 'Data/TheMovieDatabase.csv'
    
    DataPull.movieCSVcur(db_loc,stat,analysis_file)
elif loadask == False:
    print("Assumed Movie Database Csv already created")


dat = pd.read_csv('Data/TheMovieDatabase.csv')
print("Movie Database Accessed")
##add tagline sentiment

from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob
from nltk.tokenize import sent_tokenize

def get_textBlob_score(corp, sent = False, nans = 77):
    
    ##IF ERROR will return 77
    if sent:
        scores = []
        try:
            sentences = sent_tokenize(corp)
            for s in sentences:
                score = TextBlob(s).sentiment.polarity
                scores.append(score)
            
            avg_score = np.mean(scores)
            return(avg_score)
        except:
            return(nans)
        
    else:
        try:
            polarity = TextBlob(corp).sentiment.polarity
            return(polarity)
        except:
            return(nans)
    
    print("Missing Values notated with ", str(nans))
    
np.random.seed(1144)

##fill missing values
def fillNaN_Age(df):
    a = df.values
    m = np.isnan(a) 
    mu, sigma = df.mean(), df.std()
    a[m] = np.random.normal(mu, sigma, size=m.sum())
    return df
 

## Assess polarity of film's overview 
dat['overview_polarity'] = dat['overview'].apply(get_textBlob_score,sent = True, nans = None)

dat['overview_polarity'] = fillNaN_Age(dat['overview_polarity'])
    
# Tag line polarity
dat['tagline_polarity'] = dat['tagline'].apply(get_textBlob_score,sent = False, nans = None)

dat['tagline_polarity'] = fillNaN_Age(dat['tagline_polarity'])


print("Tagline Polarity Assessed")
##add controversiality

## Define Controversiality
ratio= 4/10
PolarRating = 3/10
dat['Controversial'] = np.where((dat['ratio'] > ratio) & (dat['ratio'] < 1/ratio) & (dat['PolarFreq'] > PolarRating),1,0)

print("Contrversial Ratio set between " + str(ratio) + " and " + str(1/ratio))
print("Polar Voting must make up  " + str(PolarRating) + " percentile of ratings")
##add rewrites
dat['Rewrite'] = np.where(dat['title'] != dat['original_title'],1,0)



##pull keywords from movies after 1994 and with more than 30 votes

stat= '''SELECT k.id, k.keywords
            FROM metadata m, keywords k, contratings r
            WHERE k.id = m.id AND r.movieId = m.ratingId AND m.release_date > '1994-12-31' AND r.TotalVotes > 30;'''
 
conn = db_ec.connect_db(db_loc)           
keywords_Df = pd.read_sql_query(stat, conn)

print("Keywords pulled from database")

##format from json to list
def get_key_name_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        return names
    return []

keywords_Df['keywords']=keywords_Df['keywords'].apply(literal_eval)
keywords_Df['keywords']=keywords_Df['keywords'].apply(get_key_name_list)



print("Keywords parsed")
lst_col = 'keywords'

##spread dataframe out to multiple ids, one keyword tag per row
dupli_keys = pd.DataFrame({
      col:np.repeat(keywords_Df[col].values, keywords_Df[lst_col].str.len())
      for col in keywords_Df.columns.drop(lst_col)}
    ).assign(**{lst_col:np.concatenate(keywords_Df[lst_col].values)})[keywords_Df.columns]



###use word2Vec to cluster and vectorize keywords


from gensim.models import Word2Vec
import gensim.models
from gensim.models import KeyedVectors

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet 
import string

#initialize tokenizer
tokenizer = RegexpTokenizer(r'\w+')
#add create list of stop words
stop = stopwords.words('english') + list(string.punctuation)

##Ask about devleopment of WordVec model
vec_ask = input('Is the Word2Vec Model Trained? [Y/N]')
vec_ask = db_ec.str2bool(vec_ask)
if vec_ask == True:
    modelGoog = KeyedVectors.load("preTrainGoog.model")
elif vec_ask == False:
    modelGoog = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    modelGoog.save('preTrainGoog.model')
 
 #Create corpus of keywords   
keyword_corp = dupli_keys['keywords'].apply(tokenizer.tokenize)
keyword_corp = keyword_corp.apply(lambda x: [token for token in x if token not in stop])

#initialize to hold words and vectors    
vecs = {}
#initalize words that cannot be matched by word2Vec
missinwords = set()
##loop through corpus to get vector
for i in range(0,len(keyword_corp)):
    
    #individual words within keyword
    word_vecs = []
    ##get vector of each word
    for w in keyword_corp[i]:
        try:
            #store vector of each word
            word_vecs.append(modelGoog.wv[w])
        
        #except where word is not contained in google dictionary
        except KeyError:
            missinwords.add(w)
            ##pull synoym for missing words
            syns = []
            for synset in wordnet.synsets(w):
                for lemma in synset.lemmas():
                    syns.append(lemma.name().lower())
            ##add vectors of synonyms
            syn_vecs = []        
            for syn in syns:
                try:
                    syn_vecs.append(modelGoog.wv[syn])
                    print(str(w), ' synonym ', str(syn), ' applied')
                #skip where synoyms are not matched
                except KeyError:
                    pass
            ##add average vector fo all contained synontms for words
            if syn_vecs:
                word_vecs.append(sum(syn_vecs)*1/len(syn_vecs))   
            
            pass    
    
    ##if there are vectors to add, create dictionary with keyword as key, and vector as value
    if word_vecs:
        keyVec = sum(word_vecs)        
        vecs[dupli_keys['keywords'][i]] = keyVec

    else:
        print('Unmatched:     ', w)
    
for i in missinwords:
    print("Unmatched:   ", i)

    
vectors = np.stack(vecs.values())
print("Keywords vectorized")

##Dont run PCA on vectors
pca_ask = False
#
#run PCA

if pca_ask == True:
    from sklearn.decomposition import PCA
    pca = PCA(n_components = 150)
    
    pca.fit(vectors)
    print(pca.components_)
    print(pca.explained_variance_)
    
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    
    vectors = pca.transform(vectors)
    
    centers, clusters = clustering_on_wordvecs(vectors, 150)
    centroid_map = dict(zip(vecs.keys(), clusters)) 
    
    
 ##clustering the vectors   
from sklearn.cluster import KMeans;
def clustering_on_wordvecs(word_vectors, num_clusters):
    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters = num_clusters, init='k-means++').fit(word_vectors)
    idx = kmeans_clustering.fit_predict(word_vectors);
    #return cluster centers and index
    return kmeans_clustering.cluster_centers_, idx;

def plot_keyClus(vectors, krange):
    Sum_of_squared_distances = []
    silohuette_avg = {}
    K = krange
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(vectors)
        labs = km.fit_predict(vectors)
        
        Sum_of_squared_distances.append(km.inertia_)
        silohuette_avg[k] = silhouette_score(vectors, labs)
        
    
    
    plt.plot(K, list(silohuette_avg.values()), 'bx-')
    plt.show()  
        
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

##get centers and index using 300 clusters
centers, clusters = clustering_on_wordvecs(vectors, 300)
#match keywords and cluster numbers
centroid_map = dict(zip(vecs.keys(), clusters)) 

print("Keywords clustered and identified")  
#apply cluster numbers to movie dataset    
dupli_keys['keyClus'] = dupli_keys['keywords'].map(centroid_map)
## one hot encouding
r =pd.get_dummies(dupli_keys[['id','keyClus']], columns = ['keyClus'])
#reduce to unqiue id rows
keyword_cols = r.groupby(['id'], as_index = False).sum()
#transition to binary
keyword_cols.iloc[:,1:301] = np.where(keyword_cols.iloc[:,1:301] >= 1,1,0)

##Genre mapping

dat['gen']=dat['genres'].apply(literal_eval)
dat['gen']=dat['gen'].apply(get_key_name_list)


gen_Dat = dat[['id','gen']]
lst_col = 'gen'
##spread dataframe out to multiple ids, one keyword tag per row
dupli_gens = pd.DataFrame({
      col:np.repeat(gen_Dat[col].values, gen_Dat[lst_col].str.len())
      for col in gen_Dat.columns.drop(lst_col)}
    ).assign(**{lst_col:np.concatenate(gen_Dat[lst_col].values)})[gen_Dat.columns]


rr = pd.get_dummies(dupli_gens, columns = ['gen'], prefix = "genre_")
genre_cols = rr.groupby(['id'], as_index = False).sum()
genre_cols.iloc[:,1:21] = np.where(genre_cols.iloc[:,1:21] >= 1,1,0)

##Transform original language into dummmy cols
langs = pd.get_dummies(dat[['id','original_language']], columns = ['original_language'], prefix = "language_")

#drop languages with less than ten in
langs = langs.drop([col for col, val in langs.sum().iteritems() if val < 10], axis=1, inplace=False)
col_list= list(langs)
col_list.remove('id')

##assign to small
langs['language_other'] = np.where(langs[col_list].sum(1) > 0,0,1)


###Pull production companies
dat['production_companies'] = dat['production_companies'].apply(literal_eval)
dat['production_companies'] = dat['production_companies'].apply(get_key_name_list)

pc_Dat = dat[['id','production_companies']]
lst_col = 'production_companies'

dupli_pcs = pd.DataFrame({
      col:np.repeat(pc_Dat[col].values, pc_Dat[lst_col].str.len())
      for col in pc_Dat.columns.drop(lst_col)}
    ).assign(**{lst_col:np.concatenate(pc_Dat[lst_col].values)})[pc_Dat.columns]

r4 = pd.get_dummies(dupli_pcs, columns = ['production_companies'], prefix ='', prefix_sep = '')


#drop studios with less than ten movies

##bin by number of studio productions

company = r4.sum(0)
company = company[1:]

bins = [0,1,10,60,1000]
label = ["Single","Small", "Medium", "Large"]
compBins = pd.cut(company,bins,labels = label)

compSizeDict = {}
compCountDict = {}
for i in range(0,len(compBins)):
    compSizeDict[compBins.index[i]] = compBins[i]
    compCountDict[company.index[i]] = company[i]
 
ts = []
for i in dat['production_companies']:
    sizes = []
    for studio in i:
        sizes.append(compSizeDict[studio])
    
    if "Large" in sizes:
        ts.append("Large")
    elif "Medium" in sizes:
        ts.append("Medium")
    elif "Small" in sizes:
        ts.append("Small")
    elif "Single" in sizes:
        ts.append("Single")
    else:
        ts.append(None)

dat['production_company_MaxSize'] = ts
dat = pd.get_dummies(dat, columns = ['production_company_MaxSize'], prefix ='StudioSize')
dat['production_company_MaxSize'] = ts

counts = list(company)



#b = pc_cols.drop([col for col, val in pc_cols.sum().iteritems() if val < 10], axis=1, inplace=False)
#col_list= list(b)
#col_list.remove('id')
##assign to small
#b['company_small'] = np.where(pc_cols[col_list].sum(1) > 0,0,1)

#merge 
p = pd.merge(dat, genre_cols, left_on = 'id', right_on = 'id', how = 'left')
p = pd.merge(p,keyword_cols, left_on = 'id', right_on = 'id', how = 'left')
p = pd.merge(p,langs, left_on = 'id', right_on = 'id', how = 'left')
#p = pd.merge(p,b, left_on = 'id', right_on = 'id', how = 'left')

drop_cols = ['adult','homepage',"popularity", "poster_path", "production_countries", "tagline", "gen", "genres",
             "belongs_to_collection", "original_title", "spoken_languages"]

p.drop(drop_cols,1,inplace = True)

p.to_csv("Data/TheMovieDatabase_keywords.csv",index = False)

##Cluster/Keyword Map
ClusGuide = dupli_keys.drop_duplicates(subset = ['keywords'])

def cloudCluster(clus_no):
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    ##create text
    targets = dupli_keys.loc[dupli_keys['keyClus'] == clus_no]
    
    corp = list(targets.keywords)
    
    text = ' '.join(corp)
    
    wordcloud = WordCloud(collocations=False, background_color="white").generate(text)
    
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def cloudCluster_byId(mID):
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    ##create text
    targets = dupli_keys.loc[dupli_keys['id'] == mID]
    targets = targets['keywords'].str.replace(' ','_')
    #corp = list(targets.keywords)
    
    text = ' '.join(targets)
    
    try:
        wordcloud = WordCloud(collocations=False, background_color="white").generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    except:
        print("Movie ID not matched")

     
             
      
##lemmatize and remove stop words from key words
# =============================================================================
# def stemmKeys(x,output = ["stem","lem"]):
#     from nltk.corpus import stopwords
#     from nltk.stem import WordNetLemmatizer 
#     from nltk.stem import SnowballStemmer
#     lemmatizer = WordNetLemmatizer()
#     stemmer = SnowballStemmer('english')
#     
#     from nltk.tokenize import RegexpTokenizer
#     tokenizer = RegexpTokenizer(r'\w+')
#     
#     words = tokenizer.tokenize(x)
#     word_list = [token for token in words if token not in stopwords.words('english')]
#     
#     if output == "stem":
#         stem_output = ' '.join([stemmer.stem(w) for w in word_list])
#         return(stem_output)
#     elif output == "lem":
#         lem_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
#         return(lem_output)
# 
# dupli_keys['keywordsStem'] = dupli_keys['keywords'].apply(stemmKeys, output = 'stem')
# dupli_keys['keywordsLem'] = dupli_keys['keywords'].apply(stemmKeys, output = 'lem')
# ##map ids and keywords to dummy cols
# r =pd.get_dummies(dupli_keys, columns = ['keywordsStem'])
# ##reduce across id row
# keyword_cols = r.groupby(['id'], as_index = False).sum()
# 
# keywordColNames = keyword_cols.columns
# keyword_cols.columns = keyword_cols.columns.str.replace(" ", "_")
# 
# ##only keywrods that have been used more than once
# reduced_keys = keyword_cols[keyword_cols.columns[keyword_cols.sum()>15]]
# =============================================================================


# =============================================================================
# import nltk
# from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords 
# from nltk.tokenize import word_tokenize 
#  
# keywords_Df.head()
# 
# =============================================================================
