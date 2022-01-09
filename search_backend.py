



from collections import defaultdict
import pickle
import gzip
import pandas as pd
import numpy as np
import math
from evaluation import Evaluation
import json
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from stop_words import get_stop_words
from time import time as t
import re

BUCKET_POSTINGS_BODY = 'postings_body/postings_gcp'
# BUCKET_POSTINGS_STEM_BODY = 'postings_body_stem'
BUCKET_POSTINGS_TITLE = 'postings_title/postings_gcp'
BUCKET_POSTINGS_ANCHOR = 'postings_anchor/postings_gcp'
BUCKET_POSTINGS_ANCHOR_DOUBLE = 'postings_anchor_double/postings_gcp'



class Backend:
    
    def __init__(self):
        self.body_index = pickle.load(open("body_index.pkl", "rb"))
        # self.body_stem_index = pickle.load(open("body_stem_index.pkl", "rb"))
        self.title_index = pickle.load(open("title_index.pkl", "rb"))
        self.anchor_index = pickle.load(open("anchor_index.pkl", "rb"))
        self.anchor_double_index = pickle.load(open('anchor_double_index.pkl', 'rb'))
        self.page_rank =  pd.read_csv(gzip.open('data/page_rank.csv.gz', 'rb'))
        self.page_view = pickle.load(open('page_view.pkl', 'rb'))
        self.page_view_12 = pickle.load(open('pageviews-202112.pkl', 'rb'))
        self.anchor_double_index = pickle.load(open('anchor_double_index.pkl', 'rb'))
        
       
        view_max = sorted(self.page_view_12.values(), reverse=True)[0]
    
        self.norm_page_view = {}
        for key, value_list in self.page_view_12.items():
            self.norm_page_view[key] = value_list/view_max
        

        rank_max = sorted(self.page_view.values(), reverse=True)[0]
        self.norm_page_rank = {}
        
        for key, value_list in self.page_rank.items():
            self.norm_page_rank[key] = value_list/rank_max

        self.N = len(self.page_rank)


    def tf_idf(self, posting_list, DL):
        tf = defaultdict(int)

        idf = 1+math.log(self.N/len(posting_list), 10)
        for doc_id, freq in posting_list:
            tf[doc_id] += (freq*idf) / (1+math.log(DL[doc_id], 10))
        return tf


    def cosine_similarity(self, df, query_df):
    
        top = df.dot(query_df.T.to_numpy()).T
        
        bottom = float(np.sqrt(query_df.dot(query_df.T.to_numpy()).iat[0,0]) * np.sqrt(df.T.dot(df.to_numpy()).iat[0,0]))
        res = top.div(bottom)
        res = res.sort_values(by=res.index[0], ascending=False, axis=1)
        epsilon = 0.01
        res = res.loc[:, res.iloc[0] > epsilon]
        print(res)
        return res
        

        # query_doc = np.array(df.iloc[:, 0])

        # i = -1

        
        # for column in df:
            
        #     i += 1
        #     if i == 0:
        #         continue
        #     doc = np.array(df.iloc[:,i])
        #     if len(query_doc) > 1:
        
                
        #         top = np.dot(query_doc, doc)
        #         bottom = np.sqrt(query_doc.dot(query_doc)) * np.sqrt(doc.dot(doc))
        #         res[column] = top / bottom
        #     else:
        #         res[column] = doc[0]
            
        # return res

    



    def get_body(self, query):
        query_doc_tfidf = {}
        query_idf = {}
        query_tf = defaultdict(int)
        query_tfidf = {}
        
        for w, posting_list in self.body_index.posting_lists_iter(BUCKET_POSTINGS_BODY, query):
            query_doc_tfidf[w] = self.tf_idf(posting_list, self.body_index.DL)
            query_idf[w] = 1+math.log(self.N/len(posting_list), 10)
        
        """
        {
        "life": {"doc1": 1, "doc5": 192, ...}
        "learn": {"doc2":3, "doc5": 100, ...}
        
                Doc1, Doc2, Doc5
        life     1    null  192
        learn   null   3    100

        }
        """

        

        for w in query:
            query_tf[w] += 1

        for w, freq in query_tf.items():
            query_tfidf[w] = [freq*query_idf[w]]


        max_tfidf = max(set().union(*query_tfidf.values()))
        epsilon = 0.9
        
        df_query_tfidf = pd.DataFrame(query_tfidf)
        words = df_query_tfidf.columns
        
        df_query_tfidf = df_query_tfidf.loc[:, df_query_tfidf.iloc[0] > (max_tfidf-epsilon)]
        words = list(filter(lambda i: i not in df_query_tfidf.columns, words))


        df = pd.DataFrame(query_doc_tfidf)
        df.drop(words, axis=1,inplace=True)

    
  
        # df = pd.concat([df_query_tfidf, df]).T
        df = df.fillna(value=0)
        df_query_tfidf = df_query_tfidf.fillna(value=0)

       
        cosine_sim_body = self.cosine_similarity(df, df_query_tfidf)
        return cosine_sim_body, df.columns



    def get_kind(self, query, index, folder_name):
        res = defaultdict(list)
        for w, posting_list in index.posting_lists_iter(folder_name, query):
            for doc, freq in posting_list:
                res[doc].append((w, freq))
        new_res = {}
        for doc, words in res.items():
            new_res[doc] = (len(words), sum([freq for word, freq in words]))
        
        return new_res

    def get_page_kind(self, doc_ids,page):
        res = []
        for doc_id in doc_ids:
            try: res.append(page[doc_id])
            except: print("No such document")
        return res


    def get_title(self, query):

        res = self.get_kind(query, self.title_index, BUCKET_POSTINGS_TITLE)
        res = sorted(res.items(), key = lambda x: x[1], reverse=True)
        return [x for x, y in res]

    def get_anchor(self, query):
        res = self.get_kind(query, self.anchor_index, BUCKET_POSTINGS_ANCHOR)
        res = sorted(res.items(), key = lambda x: x[1], reverse=True)
        return [x for x, y in res]
       



    def get_page_rank(self, doc_ids):
        return self.get_page_kind(doc_ids, self.page_rank)

    def get_page_views(self, doc_ids):
        return self.get_page_kind(doc_ids, self.page_view)




    def search(self, query):
        # body = dict(self.get_body(query))

    
        query = list(query)
        
        
        
        new_query = []
        
        for q in [(query[i], query[j]) for i in range(len(query)) for j in range(i + 1, len(query))]:
            
            

            new_query.append('-'.join(q))
            # new_query.append('-'.join(q[::-1]))
        
        
    
        query_with_k = '-'.join(query)
        if query_with_k not in new_query:
            new_query.append(query_with_k)

        anchor = self.get_kind(new_query, self.anchor_double_index, BUCKET_POSTINGS_ANCHOR_DOUBLE)
        anchor = sorted(anchor.items(), key=lambda x: x[1], reverse=True)[:100]

        anchor_docs = [x[0] for x in anchor]



        if len(anchor_docs) > 15:
            body, query = self.get_body(query)
            body_docs = list(body.keys())[:50]
            res = []
            for id in anchor_docs:
                if id in body_docs:
                    res.append(id)
            return res

        if len(anchor_docs) == 0:
            body, query = self.get_body(query)
           
            titles = self.get_kind(query, self.title_index, BUCKET_POSTINGS_TITLE)
            title_docs = []

            for title, metrics in titles.items():
                if metrics[0] == metrics[1] and metrics[0] == len(query):
                    
                    title_docs.append(title)

            res = {}
            for doc, score in body.items():
                total_score = score.iloc[0]
                
                if doc in title_docs:
                    total_score *= 10


                res[doc] = total_score

            res = sorted(res.items(), key= lambda x: x[1], reverse=True)[:100]
            res = [doc for doc, score in res]
            return res


        
        # Anchor is bigger than 1 

        return anchor_docs




    def preprocess(self, query):

        ps = PorterStemmer()
        english_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links", 
                            "may", "first", "see", "history", "people", "one", "two", 
                            "part", "thumb", "including", "second", "following", 
                            "many", "however", "would", "became", "make"]

        all_stopwords = english_stopwords.union(corpus_stopwords).union(get_stop_words('en'))
        RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        tokens = [token.group() for token in RE_WORD.finditer(query.lower()) if token.group() not in all_stopwords]
        # tokens_stem = [ps.stem(token.group()) for token in RE_WORD.finditer(query.lower()) if token.group() not in all_stopwords]


        return tokens

    def evaluate(self):
        evaluator = Evaluation()
        test_queries = json.loads(open("queries_train.json").read())
        predictions = []
        ground_trues = []
        time = []
        for query, true_label in list(test_queries.items()):
            
            preprocess_query = self.preprocess(query)
            start = t()
            print(preprocess_query)
            pred = self.search(preprocess_query)
            print(pred)
            end = t()
            time.append(end-start)
            print(f"Time: {end-start}")
            print()

            
            predictions.append(pred)
            ground_trues.append(true_label)
            
            

        print(evaluator.evaluate(ground_trues, predictions, 40))
        print(f"avgTIME {np.mean(time)}")







        

        
        
        # query_tfidf = {}
        # for w, posting_list in self.body_index.posting_lists_iter("postings_body", tokens):
        #     query_tfidf[w] = round(self.tf_idf(posting_list), 5)
        
        
        # cosine_sim_body, body_docs = self.cosine_similarity(tokens, self.body_index, query_tfidf, "postings_body")
        # cosine_sim_title, title_docs = self.cosine_similarity(tokens, self.title_index, query_tfidf, "postings_title")


        # doc_set = set(body_docs + title_docs)
        # total_sim = 0
        # count = 0
        # res = []
        # for doc in doc_set:
        #     if doc in self.norm_page_view:
        #         total_sim += self.norm_page_view[doc]
        
        #     if doc not in cosine_sim_title:
        #         total_sim += cosine_sim_body[doc]*0.6
        #     elif doc not in cosine_sim_body:
        #         count += 1
        #         total_sim += cosine_sim_title[doc]*0.6
        #     else:
        #         print(cosine_sim_title[doc], cosine_sim_body[doc], self.norm_page_view[doc])
        #         total_sim += cosine_sim_title[doc]*0.25  * cosine_sim_body[doc]*0.5
        #     res.append((doc, total_sim))
        # print(count)
  
        
        # res = sorted(res, key=lambda x: x[1], reverse=True)
        # res = [x[0] for x in res][:100]
        # print(res)

        

# from nltk.corpus import wordnet as wn
backend = Backend()
backend.evaluate()



# #Creating a list 
# synonyms = []
# for syn in wn.synsets("travel"):
#     for lm in syn.lemmas():
#              synonyms.append(lm.name())#adding into synonyms
# print (set(synonyms))