

import math
from tqdm import tqdm
from time import time as t

class Evaluation:
    def __init__(self):
        pass

  
    def intersection(self, l1,l2):      
        return list(set(l1)&set(l2))

  
    def recall_at_k(self, true_list,predicted_list,k=40):
        pred = predicted_list[:k]
        inter = self.intersection(pred, true_list)
        return round(len(inter) / len(true_list), 3)

    def precision_at_k(self, true_list,predicted_list,k=40):    

        pred = predicted_list[:k]
        inter = self.intersection(pred, true_list)

        return round(len(inter) / k, 3)

    def r_precision(self, true_list,predicted_list):
        pred = predicted_list[:len(true_list)]
        inter = self.intersection(pred, true_list)
        return round(len(inter) / len(true_list), 3)


    def reciprocal_rank_at_k(self, true_list,predicted_list,k=40):
        flag = False
        for i in range(min(k, len(predicted_list))):
            if predicted_list[i] in true_list:
                k = i + 1
                flag = True
                break
        if not flag:
            return 0
        return 1/k



    def f_score(self, true_list,predicted_list,k=40):


        if self.recall_at_k(true_list,predicted_list,k) + self.precision_at_k(true_list,predicted_list,k) == 0:
            return 0

        return (2 * self.recall_at_k(true_list,predicted_list,k) * self.precision_at_k(true_list,predicted_list,k)) / (self.recall_at_k(true_list,predicted_list,k) + self.precision_at_k(true_list,predicted_list,k))


    def average_precision(self, true_list,predicted_list,k=40):

        pred = predicted_list[:k]
        relevent = 0
        total = 0
        for i in range(min(k, len(pred))):
            if pred[i] in true_list:
                relevent += 1
                total += relevent / (i+1)
        if relevent == 0:
            return 0
        return round(total / relevent, 3)


    def ndcg_at_k(true_tuple_list,predicted_list,k=40):
        pred = predicted_list[:k]
        scores = {}
        for doc_id, score in true_tuple_list:
            scores[doc_id] = score
        
        total = 0
        i = 1
        relev = 0
        for doc in pred:
            if doc in scores:
                total += (2**scores[doc] - 1) / math.log(1 + i, 2)
                relev += 1
        i += 1

        ideal = [score for doc_id, score in true_tuple_list if doc_id in pred][:relev]
        ideal_score = 0

        i = 1
        for s in ideal: 

            ideal_score += (2**s - 1) / math.log(1 + i, 2)
        
        i += 1
        if ideal_score == 0:
            return 0
        return round(total / ideal_score , 3)




    def evaluate(self, ground_trues,predictions,k,print_scores=True):

        recall_lst = []
        precision_lst = []
        f_score_lst = []
        r_precision_lst = []
        reciprocal_rank_lst = []
        avg_precision_lst = []
        # ndcg_lst = []
        metrices = {'recall@k':recall_lst,
                    'precision@k':precision_lst,
                    'f_score@k': f_score_lst,
                    'r-precision': r_precision_lst,
                    'MRR@k':reciprocal_rank_lst,
                    'MAP@k':avg_precision_lst,

                    }
                    # 'ndcg@k':ndcg_lst}
        
    
        for i in range(len(ground_trues)):
            ground_true = ground_trues[i]
            predicted = predictions[i]
            recall_lst.append(self.recall_at_k(ground_true,predicted,k=k))
            precision_lst.append(self.precision_at_k(ground_true,predicted,k=k))
            f_score_lst.append(self.f_score(ground_true,predicted,k=k))
            r_precision_lst.append(self.r_precision(ground_true,predicted))
            reciprocal_rank_lst.append(self.reciprocal_rank_at_k(ground_true,predicted,k=k))
            avg_precision_lst.append(self.average_precision(ground_true,predicted,k=k))
    
        # ndcg_lst.append(self.ndcg_at_k(ground_true,predicted,k=k))

        if print_scores:
            for name,values in metrices.items():
                    print(name,sum(values)/len(values))

        return metrices    
    
 





