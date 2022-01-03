



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
        self.body_index = pickle.load(open("data/postings_body_index.pkl", "rb"))
        self.title_index = pickle.load(open("data/postings_title_index.pkl", "rb"))
        self.page_rank =  pd.read_csv(gzip.open('data/page_rank.csv.gz', 'rb'))
        self.N = len(self.page_rank)

    def get_candidate_documents_and_scores(self, query_to_search,index,words,pls):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.
        
        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.'). 
                        Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: generator for working with posting.
        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                key: pair (doc_id,term)
                                                                value: tfidf score. 
        """
        candidates = {}
                
        for term in np.unique(query_to_search):        
            if term in words:            
                list_of_doc = pls[words.index(term)]                        
                normlized_tfidf = [(doc_id,(freq/DL[str(doc_id)])*math.log(self.N/index.df[term],10)) for doc_id, freq in list_of_doc]           
                            
                for doc_id, tfidf in normlized_tfidf:
                    candidates[(doc_id,term)] = candidates.get((doc_id,term), 0) + tfidf               
            
        return candidates
     


    def generate_document_tfidf_matrix(self, query_to_search,index,words,pls):
        """
        Generate a DataFrame `D` of tfidf scores for a given query. 
        Rows will be the documents candidates for a given query
        Columns will be the unique terms in the index.
        The value for a given document and term will be its tfidf score.
        
        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.'). 
                        Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: generator for working with posting.
        Returns:
        -----------
        DataFrame of tfidf scores.
        """
        
        total_vocab_size = len(index.term_total)
        candidates_scores = self.get_candidate_documents_and_scores(query_to_search,index,words,pls) #We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
        D = np.zeros((len(unique_candidates), total_vocab_size))
        D = pd.DataFrame(D)
        
        D.index = unique_candidates
        D.columns = index.term_total.keys()

        for key in candidates_scores:
            tfidf = candidates_scores[key]
            doc_id, term = key    
            D.loc[doc_id][term] = tfidf

        return D 

    def tf_idf(self, posting_list, doc_tfidf):
        tf = 0
        
        for doc_id, freq in posting_list:
            tf += freq
            doc_tfidf[doc_id] += freq

        idf = math.log(self.N/len(posting_list), 10)
        return tf*idf


    def cosine_similarity(D,Q):
        """
        Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
        Generate a dictionary of cosine similarity scores 
        key: doc_id
        value: cosine similarity score
        
        Parameters:
        -----------
        D: DataFrame of tfidf scores.

        Q: vectorized query with tfidf scores
        
        Returns:
        -----------
        dictionary of cosine similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: cosine similarty score.
        """
        # YOUR CODE HERE
        cosine = {}
        doc_id = 0

        for row, doc in D.iterrows():
        
            top = np.dot(doc, Q)
            bottom1 = np.sqrt(np.dot(doc, doc))
            bottom2 = np.sqrt(np.dot(Q, Q))
            
            cos_sim =top/(bottom1*bottom2)
            cosine[doc_id] = cos_sim

            doc_id += 1
        return cosine
            

    def body_cosine_tfidf(self, tokens):
        
       
       
        # self.body_index.posting_locs.items()
        # print(self.body_index.posting_locs.items())
        # print(self.body_index.posting_locs.items())
       
        doc_tfidf = defaultdict(int)
        query_tfidf = {"tfidf": [], "word": []}
        for w, posting_list in self.body_index.posting_lists_iter("postings_body", tokens):
            query_tfidf["tfidf"].append(round(self.tf_idf(posting_list), 5))
            query_tfidf["word"].append(w)
        
        doc_tfidf 
        df_tfidf = pd.DataFrame(query_tfidf)
        print(df_tfidf)
        

    
            

        


            
        
            
# tokens = ["information", "retrieval"]
# Backend.body_cosine_tfidf(tokens)
