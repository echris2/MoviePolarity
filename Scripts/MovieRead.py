# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:21:07 2020

@author: eachr

Functions necessary to construct SQLite database storing Movie metadata, ratings data, and keyword data

"""

import os
import json
import csv
import sys
import pandas as pd



#keyword_loc = '../Data/keywords.csv'
def Keyword_Ratings_Clean(keyword_loc,ratings_loc):
##check ratings and key words for duplicate ids
    keyword_df = pd.read_csv(keyword_loc)
    ratings_df = pd.read_csv(ratings_loc)
    
    duplicateRate_df = ratings_df.groupby(['userId',"movieId","timestamp"]).filter(lambda x: len(x) > 1).sort_values(by = ['movieId'])
    duplicateKey_df = keyword_df.groupby("id").filter(lambda x: len(x) > 1).sort_values(by = ['id',len('keywords')])
    
    ##add column of keyword length, will be basis for dropping duplicates
    keyword_df['keyCount'] = keyword_df['keywords'].apply(len,1)
    
    keyword_df_Clean = keyword_df.sort_values(by = ['id','keyCount'], ascending = False)
    keyword_df_Clean.drop_duplicates(subset = 'id', keep = 'first', inplace = True)



def alignIds(meta_csv_loc, rating_csv_loc, linking_csv_loc):
    ##read in datat
    links_df = pd.read_csv(linking_csv_loc)
    ratings_df = pd.read_csv(rating_csv_loc)
    meta_df = pd.read_csv(meta_csv_loc, low_memory = False)
    
    ##LINKING Clean
    
    #identify duplicate in links
    
    links_df.dropna(axis = 'rows', subset = ['tmdbId'], inplace = True)
    link_dupes = links_df[links_df.duplicated(subset=['tmdbId'],keep=False)]
    
    #loop through duplicate links and combine ratings to one metadata movie rating
    for i in link_dupes['tmdbId'].unique():
        ratingIds = link_dupes.movieId.loc[link_dupes['tmdbId'] == i].sort_values(ascending = True)
        
        for ii in ratingIds[1:]:
            ratings_df['movieId'].replace(ii,ratingIds.iloc[0], inplace = True)
            links_df['movieId'].replace(ii,ratingIds.iloc[0], inplace = True) 
            #ratings_df['movieId'].loc[ratings_df['movieId'] == ii] = ratingIds.iloc[0]
            #links_df['movieId'].loc[links_df['tmdbId'] == ii] = ratingIds.iloc[0]
            
    #drop duplicates
    links_df.drop_duplicates(subset = ['movieId','tmdbId'], inplace = True)
    
    
     
    #META DATA
    meta_df_Clean = meta_df.sort_values(by = ['id', 'revenue'], ascending = False)
    meta_df_Clean.drop_duplicates(subset = 'id', keep = 'first', inplace = True)
    
    ##assessing data types
    meta_df_Clean['budget'] = meta_df_Clean.budget.apply(pd.to_numeric, errors = 'coerce')
    meta_df_Clean['budget']= meta_df_Clean.budget.fillna(0)
    meta_df_Clean['id'] = meta_df_Clean['id'].str.replace('-','')
    
    
    meta_df_Clean['runtime'] = meta_df_Clean.runtime.apply(pd.to_numeric, errors = 'coerce')
    meta_df_Clean['id'] = meta_df_Clean.id.apply(pd.to_numeric, errors = 'coerce')
    meta_df_Clean['popularity'] = meta_df_Clean.popularity.apply(pd.to_numeric, errors = 'coerce')
    
    meta_df_Clean = meta_df_Clean.astype({"budget": int, "id": int,"popularity":float,"revenue":float,"runtime":float})
    

    
    meta_df_Clean['ratingId'] = meta_df_Clean['id'].map(links_df.set_index('tmdbId')['movieId'])
    
    
    ##New csv files
    rating_CSVclean_loc = "Data/ratings_clean.csv" 
    meta_CSVclean_loc = 'Data/movies_metadata_clean.csv'
    links_CSVclean_loc = 'Data/linking_clean.csv'
    
    meta_df_Clean.to_csv(meta_CSVclean_loc,index=False)
    ratings_df.to_csv(rating_CSVclean_loc, index = False) 
    links_df.to_csv(links_CSVclean_loc, index = False)
 
    return(rating_CSVclean_loc,meta_CSVclean_loc,links_CSVclean_loc)

def movieRatings_load(db_loc, rating_csv_loc):
    
    from Scripts import db_ec
    ratings_stat = '''CREATE TABLE IF NOT EXISTS ratings(
                            userID integer,
                            movieId integer,
                            rating real,
                            timestamp integer,
                            
                            primary key(userID,movieID, timestamp))
                            
                            ;                    
                            '''
    try:
        conn = db_ec.connect_db(db_loc)
        db_ec.create_table(conn, ratings_stat)
    except:
        print("Error creating table 2")
        
    finally: 
        conn.close()
    

    ##load data from clean ratings data    
    try:
        conn = db_ec.connect_db(db_loc)
        cur = conn.cursor()
        
        with open(rating_csv_loc,'r') as myfile:
            reader = csv.DictReader(myfile)
            
            to_db = [(i['userId'], i['movieId'],i['rating'],i['timestamp']) for i in reader]
        
            cur.executemany('''INSERT OR REPLACE into ratings(userID,movieID,rating,timestamp) 
                            VALUES(?,?,?,?);''',to_db)
                            
            conn.commit()
            
    except:
        print('Error loading ratings')
    finally:
        conn.close()
        

 

def movieMeta_load(db_loc, meta_csv_loc):
    from Scripts import db_ec
    
    meta_stat = '''CREATE TABLE IF NOT EXISTS metadata(
                            adult text, 
                            belongs_to_collection text,
                            budget integer,
                            genres text,
                            homepage text,
                            id integer,
                            imdb_id text,
                            original_language text,
                            original_title text,
                            overview text,
                            popularity real,
                            poster_path text,
                            production_companies text,
                            production_countries text,
                            release_date text,
                            revenue real,
                            runtime real,
                            spoken_languages text,
                            status text,
                            tagline text,
                            title text,
                            video text, 
                            vote_average real,
                            vote_count real,
                            ratingId integer,
                            
                            primary key(id),
                            foreign key (ratingId) references ratings(movieId))                       
                            ;
                    '''
                    
    try:
        conn = db_ec.connect_db(db_loc)
        db_ec.create_table(conn,meta_stat)
    except:
        print("Error creating metadata table")
    finally:
        conn.close()
        
        
    ##load
    try:
        conn = db_ec.connect_db(db_loc)
        cur = conn.cursor()
         
        records = cur.execute('SELECT * FROM metadata;')
    #clear current records
        if records:
            cur.execute("DELETE FROM metadata;")
            conn.commit()
         
        with open(meta_csv_loc,'r', encoding="utf8") as myfile:
        
            reader = csv.DictReader(myfile) 
            
            to_db = [(i['adult'],i['belongs_to_collection'],i['budget'],i['genres'],i['homepage'],i["id"],
                i["imdb_id"],i["original_language"],i["original_title"],i["overview"],i["popularity"],
                i["poster_path"],i["production_companies"],i["production_countries"],i["release_date"],
                i["revenue"],i["runtime"],i["spoken_languages"],i["status"],i["tagline"],i["title"],
                i["video"],i["vote_average"],i["vote_count"], i["ratingId"]) for i in reader]
            
            cur.executemany('''INSERT OR REPLACE INTO metadata (adult, belongs_to_collection,budget,genres, homepage, id,
                            imdb_id, original_language,original_title,overview,popularity,poster_path,
                            production_companies,production_countries,release_date,revenue,runtime,spoken_languages,
                            status,tagline,title,video,vote_average,vote_count, ratingId) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
                             ''', 
                             to_db)
 
        conn.commit()
    except:
        print("Error loading DB")
    finally:
        conn.close()
       
         
       
def movieKeywords_load(db_loc,keyword_csv_loc):
    
    from Scripts import db_ec
    keyword_stat = '''CREATE TABLE IF NOT EXISTS keywords(
                            id integer, 
                            keywords text,
                            
                            primary key(id),
                            foreign key(id) references metadata(id)
                            );
                            '''
    try:
        conn = db_ec.connect_db(db_loc)
        db_ec.create_table(conn, keyword_stat)
    except:
        print("Error creating table")
        
    finally: 
        conn.close()
        
    try:
        conn = db_ec.connect_db(db_loc)
        cur = conn.cursor()
        
        with open(keyword_csv_loc,'r', encoding = 'utf8') as myfile:
            reader = csv.DictReader(myfile)
            
            to_db = [(i['id'],i['keywords']) for i in reader]
        
            cur.executemany('''INSERT OR REPLACE into keywords(id,keywords) 
                            VALUES(?,?);''',to_db)
                            
            conn.commit()
            
    except:
        print('Error loading keywords')
    finally:
        conn.close()
 
    
def define_Controversy(db_loc):
    from Scripts import db_ec
    contro_table_stat = '''CREATE TABLE IF NOT EXISTS contRatings(
                            movieId integer,
                            ratio real,
                            PolarFreq real,
                            TotalVotes integer,
                            AvgRating real,
                            
                            
                            primary key (movieId),
                            foreign key (movieId) references ratings(movieId),
                            foreign key (movieId) references metadata(ratingId));'''
    
    calc_stat_data = '''SELECT t.movieId, t.HighCount*1.0/t.LowCount as Ratio, 
                            ((t.HighCount + t.LowCount)*1.0/t.TotalCount) as PolarFreq, 
                            t.TotalCount, t.AvgRating
                     FROM (SELECT movieId, 
                          COUNT(CASE WHEN rating <= 1 THEN rating END) as LowCount,
                          COUNT(CASE WHEN rating > 4 THEN rating END) as HighCount,
                          COUNT(rating) as TotalCount,
                          AVG(rating) as AvgRating
                          
                          FROM ratings
                          GROUP BY movieId)  t
                    ;
                  '''
            
        
    try:
        conn = db_ec.connect_db(db_loc)
        db_ec.create_table(conn, contro_table_stat)
        conn.commit()
    except:
        print("Error creating table")
        
    finally: 
        conn.close()
        
    ##load controversy table
        
    try:
        conn = db_ec.connect_db(db_loc)
        cur = conn.cursor()
        
        cont_dat = cur.execute(calc_stat_data)
        
        cur.executemany('INSERT OR REPLACE INTO contRatings(movieId,ratio,PolarFreq,TotalVotes, AvgRating) values (?,?,?,?,?)', cont_dat.fetchall())
        conn.commit()
    finally:
        conn.close()
        
# =============================================================================
# def id_avgs(db_loc):
#     stat = '''SELECT AVG(r.rating), (SELECT rating
#                                             FROM ratings as r2
#                                             WHERE movieID = m2.movieId
#                                             GROUP BY rating
#                                             ORDER BY COUNT(*) DESC
#                                             LIMIT 1)
#                 FROM (SELECT DISTINCT movieId
#                       FROM ratings) as m2, ratings as r
#                 GROUP BY r.movieId
#                 
#                 INNER JOIN ratings
#                     ON r.movieId = ratings.movieId;
#                 ;
#                 '''
# =============================================================================
                

        
        
        
# =============================================================================
# with open('../Data/movies_metadata.csv','r') as file:
#     reader = csv.DictReader(file)
#     
#     conn = db_ec.connect_db(db_loc)
#     cur = conn.cursor()
#     
#     records = cur.execute('SELECT * FROM metadata;')
#         #clear current records
#     if records:
#         cur.execute("DELETE FROM metadata;")
#         conn.commit()
#     
#     to_db = [(i['adult'],i['belongs_to_collection'],i['budget'],i['genres'],i['homepage'],i["id"],
#                 i["imdb_id"],i["original_language"],i["original_title"],i["overview"],i["popularity"],
#                 i["poster_path"],i["production_companies"],i["production_countries"],i["release_date"],
#                 i["revenue"],i["runtime"],i["spoken_languages"],i["status"],i["tagline"],i["title"],
#                 i["video"],i["vote_average"],i["vote_count"]) for i in reader]
#             
#     cur.executemany('''INSERT INTO metadata (adult, belongs_to_collection,budget,genres, homepage, id,
#                         imdb_id, original_language,original_title,overview,popularity,poster_path,
#                         production_companies,production_countries,release_date,revenue,runtime,spoken_languages,
#                         status,tagline,title,video,vote_average,vote_count) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);
#                         ''', 
#                         to_db)
#     conn.commit()
#     conn.close()
#     
# =============================================================================



# =============================================================================
# test_stat = '''SELECT movieId, 
#                       COUNT(CASE WHEN rating <= 1 THEN rating END) as LowCount,
#                       COUNT(CASE WHEN rating > 4 THEN rating END) as HighCount,
#                       COUNT(rating) as TotalCount
#                       
#                       FROM ratings
#                       GROUP BY movieId;
#               '''
#   
#     
# conn = db_ec.connect_db(db_loc)
# cur = conn.cursor()    
# 
# p = cur.execute(test_stat)
# pp = p.fetchall()
# =============================================================================
    
# =============================================================================
# credits_df = pd.read_csv('Data/credits.csv')
# meta_df = pd.read_csv('Data/movies_metadata.csv')
# keyword_df = pd.read_csv('Data/keywords.csv')
# ratings_df = pd.read_csv('Data/ratings.csv')
# 
# meta_df_pos_rev = meta_df[meta_df.revenue > 0]
# 
# plt.scatter(meta_df_pos_rev['revenue'], meta_df_pos_rev['vote_average'])
# 
# plt.scatter(meta_df_pos_rev['revenue'], meta_df_pos_rev['budget'].astype(int))
# 
# 
# 
# pd.read_json(p.cast[[0]])
# 
# meta_df['genres'] = meta_df['genres'].replace("\'", "\"")
# meta_df['genres'] = meta_df['genres'].apply.replace("\'", "\"")
# meta_df['genres'].apply(json.loads)
# 
# 
# from ast import literal_eval
# # Returns the list top l elements or entire list; whichever is more.
# def get_list(x, l=5):
#     if isinstance(x, list):
#         names = [i['name'] for i in x]
#         #Check if more than l elements exist. If yes, return only first three. If no, return entire list.
#         if len(names) > l:
#             names = names[:l]
#         return names
# 
#     #Return empty list in case of missing/malformed data
#     return []
# 
# 
# 
# =============================================================================
