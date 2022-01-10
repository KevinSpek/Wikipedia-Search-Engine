



from collections import defaultdict
import pickle
import gzip
import pandas as pd
import numpy as np
import math

from pandas._config.config import reset_option
from evaluation import Evaluation
import json
from nltk.corpus import stopwords
from time import time as t
import re
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="wiki-retrieval-669377a04b0f.json"


# Global indexes bucket locations
BUCKET_POSTINGS_BODY = 'postings_body/postings_gcp'
BUCKET_POSTINGS_STEM_BODY = 'postings_body_stem'
BUCKET_POSTINGS_TITLE = 'postings_title'
BUCKET_POSTINGS_ANCHOR = 'postings_anchor/postings_gcp'
BUCKET_POSTINGS_ANCHOR_DOUBLE = 'postings_anchor_double/postings_gcp'



class Backend:
    
    def __init__(self):

        """
        Put super indexes and relevent files in main memory to save time during queries
        """

        self.body_index = pickle.load(open("data/body_index.pkl", "rb"))
        self.body_stem_index = pickle.load(open("data/body_stem_index.pkl", "rb"))
        self.title_index = pickle.load(open("data/title_index.pkl", "rb"))
        self.anchor_index = pickle.load(open("data/anchor_index.pkl", "rb"))
        self.page_rank =  pd.read_csv(gzip.open('data/page_rank.csv.gz', 'rb'))
        self.page_view = pickle.load(open('data/page_view.pkl', 'rb')) # October Page view has missing documents
        self.page_view_12 = pickle.load(open('data/pageviews-202112.pkl', 'rb')) # Use December 2021 page views to include missing docs
        self.anchor_double_index = pickle.load(open('data/anchor_double_index.pkl', 'rb'))

        
        
       
        view_max = sorted(self.page_view_12.values(), reverse=True)[10] 
    
        self.norm_page_view = {} # Create a normalized version of page view
        for key, value_list in self.page_view_12.items():
            self.norm_page_view[key] = value_list/view_max
        

        rank_max = sorted(self.page_view.values(), reverse=True)[10]
        self.norm_page_rank = {}
        
        for key, value_list in self.page_rank.items(): # Create a normalized version of page rank
            self.norm_page_rank[key] = value_list/rank_max

        self.N = len(self.page_rank) # Total number of documents


    def tf_idf(self, posting_list, DL):

        """
        Params: 
        --------
        postings_list - A list of <doc_id, frequency> of a word
        DL - Dictionary of <doc_id, Doc_length> of all documents

        ========

        Calculate TFIDF for a word.
        
        Output:
        ------
        Dictionary: <doc_id, tfidf_score>
        """

        tf = defaultdict(int)

        idf = 1+math.log(self.N/len(posting_list), 10)
        for doc_id, freq in posting_list:
            tf[doc_id] += (freq*idf) / (1+math.log(DL[doc_id], 10))
        return tf


    def cosine_similarity(self, df, query_df):

        """"
        Params:
        -------
        df - A dataframe containing words as columns and tfidf in each row for documents
        Example:
                    word1    word2
        doc_id1     0.12      0
        doc_id2     0.43     0.2
        doc_id3     1.2      2.3
          ...       ...      ...
        doc_idn     0.7      2.4
          
                 doc_id1  doc_id2  doc_id3 ... doc_idn
        word1     0.12     0.43     1.2          0.7
        word2      0       0.2      2.3          2.4



        query_df - TFIDF of words in query. Similar to df but contains only 1 row
        Example:
                   word1     word2
            0      0.34       0.76
        
        ========


        Calculates cosine similarity between each doc with query using pandas dot product.
        
        Output:
        -------
        Dataframe: docs as columns and row as score
        Example:
                doc_id1    doc_id2   ...   doc_idn
           0      0.2       0.001            0.32
        
        
        """
        top = df.dot(query_df.T.to_numpy()).T # Dot product between documents and query
        
        bottom = float(np.sqrt(query_df.dot(query_df.T.to_numpy()).iat[0,0]) * np.sqrt(df.T.dot(df.to_numpy()).iat[0,0])) # sqrt of each vector in the power of 2 multiplied, 
        res = top.div(bottom)
        res = res.sort_values(by=res.index[0], ascending=False, axis=1)
        epsilon = 0.01
        res = res.loc[:, res.iloc[0] > epsilon] # Remove docs with too small score
        
        return res
        
    

    def get_body(self, query):

        """
        Params:
        ------
        query - list of tokens. i.e: ["hello", "world"]
        ======

        Iterates over each words and downloads its relevent postings list.
        Calculates the tfidf of each word in each document
        and calculates the tfidf of each word in query

        Also, we remove words that their tfidf in the query_tfidf is smaller than some epsilon from the highest tfidf.
        
        Output:
        Dataframe: docs as columns and row as score
        List of remaining words
        """

        query_doc_tfidf = {}
        query_idf = {}
        query_tf = defaultdict(int)
        query_tfidf = {}
        
        for w, posting_list in self.body_index.posting_lists_iter(BUCKET_POSTINGS_BODY, query): # Iterate over each posting list
            query_doc_tfidf[w] = self.tf_idf(posting_list, self.body_index.DL) # save tfidf for each doc
            query_idf[w] = 1+math.log(self.N/len(posting_list), 10) # save udf of each word in query

        if len(query_idf) == 0:
            return pd.DataFrame({}), []
    
        for w in query_idf:
            query_tf[w] += 1 # Calculate term frequency of words in query

        for w, freq in query_tf.items():
            score = freq*query_idf[w]
            query_tfidf[w] = [score] # save tfidf of each word in query


        max_tfidf = max(set().union(*query_tfidf.values())) # get the highest tfidf in query
        epsilon = 0.9 # filters words with too small tfidf
        
        df_query_tfidf = pd.DataFrame(query_tfidf)
        words = df_query_tfidf.columns
        
        df_query_tfidf = df_query_tfidf.loc[:, df_query_tfidf.iloc[0] > (max_tfidf-epsilon)] # filter irrelevent words
        words = list(filter(lambda i: i not in df_query_tfidf.columns, words)) # Extract the words that were removed



        df = pd.DataFrame(query_doc_tfidf) 
        df.drop(words, axis=1,inplace=True) # filter irrelevent words

    
  
        # df = pd.concat([df_query_tfidf, df]).T
        df = df.fillna(value=0)
        df_query_tfidf = df_query_tfidf.fillna(value=0)

       
        cosine_sim_body = self.cosine_similarity(df, df_query_tfidf) # calculate cosine similarity
        return cosine_sim_body, df.columns




    def get_kind(self, query, index, folder_name):

        """
        Params:
        ------
        query - list of tokens. i.e.: ["hello", "world"]
        index - An InvertedIndex object
        folder_name - the location of postings list in the bucket
        ======

        Calculates for each doc how many tokens from query the doc has and the total frequency of token of the doc
        
        Output:
        ------
        dictionary: <doc_id, (len(tokens), freq(tokens))>
        
        """


        res = defaultdict(list) 
        for w, posting_list in index.posting_lists_iter(folder_name, query): # Iterate over each posting list
            for doc, freq in posting_list:
                res[doc].append((w, freq))
        new_res = {}
        for doc, words in res.items():
            new_res[doc] = (len(words), sum([freq for word, freq in words])) # calculate for each doc how many words from query inside and the total frequency
        
        return new_res

    def get_page_kind(self, doc_ids,page):

        """
        Params:
        ------
        doc_ids: list of doc_id. i.e: [125, 653, 12]
        page: Dictionary of <doc_id, doc_score>
        ======
        
        Filters docs from pages that are inside the doc_ids

        Output:
        ------
        Dictionary: <doc_id, doc_score>
        """
        

        res = []
        for doc_id in doc_ids:
            try: res.append(page[doc_id])
            except: print("No such document")
        return res


    def get_title(self, query):

        """
        Params:
        ------
        query: list of tokens. i.e.: ["hello", "world"]
        ======


        Sorting all relevent docs of query with the title index

        Output:
        ------
        List of doc_ids sorted by anchor relevence
        """

        res = self.get_kind(query, self.title_index, BUCKET_POSTINGS_TITLE)
        res = sorted(res.items(), key = lambda x: x[1], reverse=True)
        return [x for x, y in res]

    def get_anchor(self, query):
        
        """
        Params:
        ------
        query: list of tokens. i.e.: ["hello", "world"]
        ======


        Sorting all relevent docs of query with the anchor index

        Output:
        ------
        List of doc_ids sorted by title relevence
        """
        res = self.get_kind(query, self.anchor_index, BUCKET_POSTINGS_ANCHOR)
        res = sorted(res.items(), key = lambda x: x[1], reverse=True)
        return [x for x, y in res]
       



    def get_page_rank(self, doc_ids):

        """
        Params:
        ------
        docs_ids: list of docs.: i.e.: [142, 655, 12]
        ======

        Output:
        Dictionary: <doc_id, doc_score_page_rank>
        
        """

        return self.get_page_kind(doc_ids, self.page_rank)

    def get_page_views(self, doc_ids):
        """
        Params:
        ------
        docs_ids: list of docs.: i.e.: [142, 655, 12]
        ======

        Output:
        Dictionary: <doc_id, doc_score_page_view>
        
        """
        return self.get_page_kind(doc_ids, self.page_view)




    def search(self, query):

        """
        Main Search Engine

        Params:
        ------
        List of tokens. i.e.: ["hello", "world"]
        ======


        Engine Strategy:

        The endine has an anchor index where each term is a multiwords joined with '-'. This is its main retrieval metric.
        First the engine searches if any pair of tokens from query is a term in the double-anchor-index. if so, retrieve the relvent docs from it.

        If there are more than 15 relevent docs from the double-anchor-index then the result is not definite. It needs more metrics to score the docs.

        Other metrics engine uses:
        Body tfidf scores, title scores, anchor scores combined, page view scores, page rank scores combined


        Output:
        ------
        List of relvent documents sorted by relevency.
        
        
        """
        
        query = self.preprocess(query)
        print(query)

        new_query = []
        
        for q in [(query[i], query[j]) for i in range(len(query)) for j in range(i + 1, len(query))]: # Search for combinations in double-anchor-index
            
            new_query.append('-'.join(q))
            new_query.append('-'.join(q[::-1]))
        
        query_with_k = '-'.join(query)
        if query_with_k not in new_query:
            new_query.append(query_with_k)

        anchor = self.get_kind(new_query, self.anchor_double_index, BUCKET_POSTINGS_ANCHOR_DOUBLE)
        anchor = sorted(anchor.items(), key=lambda x: x[1], reverse=True)[:100]

        anchor_docs = [x[0] for x in anchor]



        if len(anchor_docs) > 15: # Search in body docs to find better results
            body, query = self.get_body(query)
            body_docs = list(body.keys())[:50]
            res = []
            for id in anchor_docs:
                if id in body_docs:
                    res.append(id)
            if len(res) == 0:
                return anchor_docs
            
            return res

        if len(anchor_docs) == 0: # If there are no docs in the double-anchor-index search in other metrics.
            body, query = self.get_body(query)
            
            titles = self.get_kind(query, self.title_index, BUCKET_POSTINGS_TITLE)
            anchors = self.get_kind(query, self.anchor_index, BUCKET_POSTINGS_TITLE)
            
            
            res = {}
            for doc, score in body.items():
                # Calculates score for each doc
                total_score = score.iloc[0]
                
                if doc in self.norm_page_view:
                    if self.norm_page_view[doc] > 0.03:
                        total_score *= 5+self.norm_page_view[doc]
                
                if doc in self.norm_page_rank:
                    if self.norm_page_rank[doc] > 0.03:
                        total_score *= 3+self.norm_page_rank[doc]
                    
                
                if doc in titles:
                    total_score *= 5
                
                if doc in anchors:
                    total_score *= 3
                    
                res[doc] = total_score

            res = sorted(res.items(), key= lambda x: x[1], reverse=True)[:15]
            res = [doc for doc, score in res]
            return res

        return anchor_docs


    def preprocess(self, query):
        
        english_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links", 
                            "may", "first", "see", "history", "people", "one", "two", 
                            "part", "thumb", "including", "second", "following", 
                            "many", "however", "would", "became", "make", "good", "best", "worst"]

        all_stopwords = english_stopwords.union(corpus_stopwords)
        RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        tokens = [token.group() for token in RE_WORD.finditer(query.lower()) if token.group() not in all_stopwords]


        return tokens

    def evaluate(self):
        evaluator = Evaluation() # Create and evaluation object
        test_queries = json.loads(open("queries_train.json").read()) # Save the json queries
        predictions = []
        ground_trues = []
        time = []
        for query, true_label in list(test_queries.items()):
            
            
            start = t()
            pred = self.search(query)
            end = t()
            print(pred)
            time.append(end-start)
            print(f"Time: {end-start}")
            print()

            
            predictions.append(pred)
            ground_trues.append(true_label)
            
            
        print(f"AVG@TIME {np.mean(time)}")
        print(evaluator.evaluate(ground_trues, predictions, 40))

