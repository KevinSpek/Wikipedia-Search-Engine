



from collections import defaultdict
import pickle
import gzip
import pandas as pd
import numpy as np
import math

from inverted_index_gcp import TF_MASK

singleton = lambda c: c()

@singleton
class Backend:
    
    def __init__(self):
        self.body_index = pickle.load(open("body_index.pkl", "rb"))
        self.title_index = pickle.load(open("title_index.pkl", "rb"))
        self.anchor_index = pickle.load(open("anchor_index.pkl", "rb"))



        self.page_rank =  pd.read_csv(gzip.open('data/page_rank.csv.gz', 'rb'))


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

        return res, docs#[x[0] for x in res]


            
           

        
        
        # cosine = {}
        # doc_id = 0

        # for index, doc in D.iterrows():
        
        #     top = np.dot(doc, Q)
        #     bottom1 = np.sqrt(np.dot(doc, doc))
        #     bottom2 = np.sqrt(np.dot(Q, Q))
            
        #     cos_sim =top/(bottom1*bottom2)
        #     cosine[doc_id] = cos_sim

        #     doc_id += 1
        # return cosine

        
            




    



    # def get_candidate_documents_and_scores(self, query_to_search,index,pls):
    #     candidates = {}       
    #     for term in np.unique(query_to_search):        
        
    #         list_of_doc = pls[term]                        
    #         normlized_tfidf = [(doc_id,(freq/index.DL[doc_id])*math.log(self.N/index.df[term],10)) for doc_id, freq in list_of_doc]           
                        
    #         for doc_id, tfidf in normlized_tfidf:
    #             candidates[(doc_id,term)] = candidates.get((doc_id,term), 0) + tfidf               
    #     return candidates
     

    
    # def generate_document_tfidf_matrix(self, query_to_search,index,words,pls):
    #     total_vocab_size = len(index.term_total)
    #     candidates_scores = self.get_candidate_documents_and_scores(query_to_search,index,words,pls)
    #     unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    #     D = np.zeros((len(unique_candidates), total_vocab_size))
    #     D = pd.DataFrame(D)
        
    #     D.index = unique_candidates
    #     D.columns = index.term_total.keys()

    #     for key in candidates_scores:
    #         tfidf = candidates_scores[key]
    #         doc_id, term = key    
    #         D.loc[doc_id][term] = tfidf
    #     return D 

    def tf_idf(self, posting_list):
        tf = 0

        for doc_id, freq in posting_list:
            tf += freq

        idf = math.log(self.N/len(posting_list), 10)
        return tf*idf


    # def cosine_similarity(D,Q):
    #     # YOUR CODE HERE
    #     cosine = {}
    #     doc_id = 0

    #     for row, doc in D.iterrows():
        
    #         top = np.dot(doc, Q)
    #         bottom1 = np.sqrt(np.dot(doc, doc))
    #         bottom2 = np.sqrt(np.dot(Q, Q))
            
    #         cos_sim =top/(bottom1*bottom2)
    #         cosine[doc_id] = cos_sim

    #         doc_id += 1
    #     return cosine
            

    def body_cosine_tfidf(self, tokens):
        
        
        query_tfidf = {}
        for w, posting_list in self.body_index.posting_lists_iter("postings_body", tokens):
            query_tfidf[w] = round(self.tf_idf(posting_list), 5)
        
        
        cosine_sim_body, body_docs = self.cosine_similarity(tokens, self.body_index, query_tfidf, "postings_body")
        cosine_sim_title, title_docs = self.cosine_similarity(tokens, self.title_index, query_tfidf, "postings_title")


        doc_set = set(body_docs + title_docs)

        res = []
        for doc in doc_set:
            if doc not in cosine_sim_title:
                total_sim = cosine_sim_body[doc]*0.9
            elif doc not in cosine_sim_body:
                total_sim = cosine_sim_title[doc]*0.85
            else:
                total_sim = cosine_sim_title[doc]*0.7  * cosine_sim_body[doc]*0.3
            res.append((doc, total_sim))
        
        res = sorted(res, key=lambda x: x[1], reverse=True)
        res = [x[0] for x in res][:100]
        print(res)

            
# tokens = ["python"]
# Backend.body_cosine_tfidf(tokens)
