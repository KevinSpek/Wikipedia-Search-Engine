



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





class Backend:
    
    def __init__(self):
        self.body_index = pickle.load(open("body_index.pkl", "rb"))
        self.title_index = pickle.load(open("title_index.pkl", "rb"))
        self.anchor_index = pickle.load(open("anchor_index.pkl", "rb"))
        self.page_rank =  pd.read_csv(gzip.open('data/page_rank.csv.gz', 'rb'))
        self.page_view = pickle.load(open('page_view.pkl', 'rb'))


        view_max = sorted(self.page_view.values(), reverse=True)[0]
    
        self.norm_page_view = {}
        for key, value_list in self.page_view.items():
            self.norm_page_view[key] = value_list/view_max
        

        rank_max = sorted(self.page_view.values(), reverse=True)[0]
        self.norm_page_rank = {}
        
        for key, value_list in self.page_rank.items():
            self.norm_page_rank[key] = value_list/rank_max

        self.N = len(self.page_rank)


    

    def relevent_docs(self, query_tokens, index, file_loc):
        
        relev_docs = defaultdict(list)
        
        for w, posting_list in index.posting_lists_iter(file_loc, query_tokens):

            posting_list = sorted(posting_list, key=lambda x: x[0], reverse=True)
            for doc, freq in posting_list:
                relev_docs[doc].append((w, freq))
                
        return relev_docs

    def tf_idf(self, posting_list, DL):
        tf = defaultdict(int)

        idf = 1+math.log(self.N/len(posting_list), 10)
        for doc_id, freq in posting_list:
            tf[doc_id] += (freq*idf) / (1+math.log(DL[doc_id], 2))
        return tf


    def cosine_similarity(self, query_tokens,index, file_loc, df):
        # relev_docs = self.relevent_docs(query_tokens, index, file_loc) #find the relavent docs to the query
        res = {}
        print(df)
        query_doc = np.array(df.iloc[:, 0])

        i = -1

        
        for column in df:
            
            i += 1
            if i == 0:
                continue
            doc = np.array(df.iloc[:,i])
            if len(query_doc) > 1:
        
                
                top = np.dot(query_doc, doc)
                bottom = np.sqrt(query_doc.dot(query_doc)) * np.sqrt(doc.dot(doc))
                res[column] = top / bottom
            else:
                res[column] = doc[0]
            
        return res


        # for doc, freq in relev_docs.items():
































        # docs = []
        # res = {}#[]
        # for doc, freqs in relev_docs.items():
        #     D_L = DL[doc] 
        #     # self.body_index.DL
            
        #     total = 0
        #     # total_freq = 0
        #     for w, freq in freqs:
        #         total =  / (D_L * len(query_tokens))

        #         total += (freq / ((1+ len(query_tokens)) - len(freqs)))*(tfidf[w])
        #         # total_freq += freq
            
        #     # total *= (total_freq / index.DL[doc])


            

        #     # res[doc].append((doc,total))
        #     res[doc] = total
        #     docs.append(doc)
        # # res = sorted(res, key=lambda x: x[1], reverse=True)

        # ## change to more efficent and fast way if needed
        # new_res = {}
        # if len(res) > 0:
        #     max_val = max(res, key=res.get)
        #     for key, value_list in res.items():
        #         new_res[key] = value_list/max_val
        # return new_res


    def get_body(self, query):
        query_doc_tfidf = {}
        query_idf = {}
        query_tf = defaultdict(int)
        query_tfidf = {}

        for w, posting_list in self.body_index.posting_lists_iter("postings_body", query):
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
        
        
        df_query_tfidf = pd.DataFrame(query_tfidf)
        df = pd.DataFrame(query_doc_tfidf)

  
        df = pd.concat([df_query_tfidf, df]).T
        df = df.fillna(value=0)

       
        cosine_sim_body = self.cosine_similarity(query, self.body_index, "postings_body", df)
        return cosine_sim_body



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

        res = self.get_kind(query, self.title_index, "postings_title")
        res = sorted(res.items(), key = lambda x: x[1], reverse=True)
        return [x for x, y in res]

    def get_anchor(self, query):
        res = self.get_kind(query, self.anchor_index, "postings_anchor")
        res = sorted(res.items(), key = lambda x: x[1], reverse=True)
        return [x for x, y in res]
       



    def get_page_rank(self, doc_ids):
        return self.get_page_kind(doc_ids, self.page_rank)

    def get_page_views(self, doc_ids):
        return self.get_page_kind(doc_ids, self.page_view)



    


            

    def search(self, query):

        
        body = self.get_body(query)
        # print(body[23862])
        # print(body[23329])
        # print(body[48407627])
        # exit()
        # body = self.get_body(['google', 'work'])
        # print(body[44674524])
        # print(body[41585185])
        # exit()

        size = 400


        titles = self.get_kind(query, self.title_index, "postings_title")
        
        title_docs = []#defaultdict(list)
        for title, metrics in titles.items():
            if metrics[0] == metrics[1] and metrics[0] == len(query) and len(query) > 1:
                # title_docs[metrics].append(title)
                title_docs.append(title)

        

        # titles = sorted(title_docs.items(), key = lambda x: x[0], reverse=True)
        
        # print(titles)

        # title_docs = {}
        # for i in range(len(titles)):
        #     score = 2 - (i/len(titles))
        #     min_point = 1
        #     max_point = 2
            
        #     for doc_ids in titles[i][1]:
        #         score = ((score - min_point) / (max_point-min_point)) * 1.3
        #         title_docs[doc_ids] = score

        
        anchor = self.get_kind(query, self.anchor_index, "postings_anchor")
        anchor = sorted(anchor.items(), key = lambda x: x[1], reverse=True)[:size]

        
        anchor_docs = {}

        for i in range(size):   
            if i < len(anchor):
                min_point = 1
                max_point = 2
                score = 2 - (i/size)
                score = ((score - min_point) / (max_point-min_point)) * 1.2 
                anchor_docs[anchor[i][0]] = score       

        res = {}
        for doc, score in body.items():
            total_score = score
            # if doc in title_docs:
            #     total_score += 0.5

            # if doc in anchor_docs:
            #     total_score *= anchor_docs[doc]

            # if doc in self.norm_page_view:
            #     total_score *= (self.norm_page_view[doc]+1)
            
            # if doc in self.norm_page_rank:
            #     total_score *= (self.norm_page_rank[doc]+1)


            res[doc] = total_score

        res = sorted(res.items(), key= lambda x: x[1], reverse=True)[:100]
        res = [doc for doc, score in res]
        return res





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


        return tokens

    def evaluate(self):
        evaluator = Evaluation()
        test_queries = json.loads(open("queries_train.json").read())
        predictions = []
        ground_trues = []
        time = []
        for query, true_label in list(test_queries.items())[:10]:
            
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
            
            

        print(evaluator.evaluate(ground_trues, predictions, 50))
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