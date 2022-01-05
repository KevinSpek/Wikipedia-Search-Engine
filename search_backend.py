



from collections import defaultdict
import pickle
import gzip
import pandas as pd
import numpy as np
import math

from inverted_index_gcp import TF_MASK




class Backend:
    
    def __init__(self):
        self.body_index = pickle.load(open("body_index.pkl", "rb"))
        self.title_index = pickle.load(open("title_index.pkl", "rb"))
        self.anchor_index = pickle.load(open("anchor_index.pkl", "rb"))
        self.page_rank =  pd.read_csv(gzip.open('data/page_rank.csv.gz', 'rb'))
        self.page_view = pickle.load(open('page_view.pkl', 'rb'))


        view_max = sorted(self.page_view.values(), reverse=True)[3]
    
        self.norm_page_view = {}
        for key, value_list in self.page_view.items():
            self.norm_page_view[key] = value_list/view_max
        
        self.N = len(self.page_rank)


    

    def relevent_docs(self, query_tokens, index, file_loc):
        
        relev_docs = defaultdict(list)
        doc_size = defaultdict(int)
        for w, posting_list in index.posting_lists_iter(file_loc, query_tokens):
            posting_list = sorted(posting_list, key=lambda x: x[0], reverse=True)
            for doc, freq in posting_list:
                relev_docs[doc].append((w, freq))
                doc_size[doc] += freq
        return relev_docs



    def cosine_similarity(self, query_tokens,index, tfidf, file_loc):
        relev_docs = self.relevent_docs(query_tokens, index, file_loc)
        docs = []
        res = {}#[]
        for doc, freqs in relev_docs.items():
            
            total = 0
            total_freq = 0
            for w, freq in freqs:
                total += (freq / ((1+ len(query_tokens)) - len(freqs)))*tfidf[w]
                total_freq += freq
            
            total *= (total_freq / index.DL[doc])


            

            # res[doc].append((doc,total))
            res[doc] = total
            docs.append(doc)
        # res = sorted(res, key=lambda x: x[1], reverse=True)

        ## change to more efficent and fast way if needed
        max_val = max(res, key=res.get)
        new_res = {}
        for key, value_list in res.items():
            new_res[key] = value_list/max_val
        return new_res


    def get_body(self, query):
        query_tfidf = {}
        for w, posting_list in self.body_index.posting_lists_iter("postings_body", query):
            query_tfidf[w] = round(self.tf_idf(posting_list), 5)
        
        
        cosine_sim_body = self.cosine_similarity(query, self.body_index, query_tfidf, "postings_body")

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
            res.append(page[doc_id])
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




    def tf_idf(self, posting_list):
        tf = 0

        for doc_id, freq in posting_list:
            tf += freq

        idf = math.log(self.N/len(posting_list), 10)
        return tf*idf


            

    def search(self, query):

        
        body = self.get_body(query)
        body = sorted(body.items(), key=lambda x: x[1], reverse=True)[:1000]
        print(body)
        body_docs = [doc for doc, val in body]


        title = self.get_kind(query, self.title_index, "postings_title")
        title = sorted(title.items(), key = lambda x: x[1], reverse=True)[:1000]
        print(title)
        title_docs = [doc for doc, val in title]


        anchor = self.get_kind(query, self.anchor_index, "postings_anchor")
        anchor = sorted(anchor.items(), key = lambda x: x[1], reverse=True)[:1000]
        print(anchor)
        anchor_docs = [doc for doc, val in anchor]
        


        docs = set(body_docs + title_docs + anchor_docs)
        # page_view = self.get_page_views(docs)
        # page_rank = self.get_page_rank(docs)
        print(len(docs))
        







        

        
        
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
tokens = ["google", "trends"]
backend.search(tokens)


# #Creating a list 
# synonyms = []
# for syn in wn.synsets("travel"):
#     for lm in syn.lemmas():
#              synonyms.append(lm.name())#adding into synonyms
# print (set(synonyms))