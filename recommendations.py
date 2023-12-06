import pandas as pd
import numpy as np
#from tqdm import tqdm

class Recommendations:

    def __init__(self, model, content_df: pd.DataFrame, train_valid_pairs, train_set_csr, id_to_product):
        self.products_data = content_df
        self.train_valid_pairs = train_valid_pairs
        self.train_set_csr = train_set_csr
        self.id_to_product = id_to_product
        self.model = model


    def id_to_content_df(ids: np.array, id_to_item):
        items = tuple(id_to_item[i] for i in ids)
        #print(items)
        result_df = self.products_data[
            self.products_data.product_id.isin(items)
        ]
        return result_df

    def cust_random_history_recommend(self, customer_id=0, N=10):

        # csr матрица на вход ALS
        random_history = self.train_set_csr[np.random.randint(low=0, high=self.train_set_csr.shape[0])]
        self.id_to_content_df(random_history.nonzero()[1], self.id_to_product)#??

        recommended_content, recommended_scores = self.model.recommend(
                    userid = customer_id,
                    user_items=random_history,
                    N=N,
                    filter_already_liked_items=True,
                    #recalculate_user=True
        )
        res = self.id_to_content_df(recommended_content, id_to_product)
        return res

    def top_n_recommends_random(self, all_content, order_history, N=50):
        top_n_result = np.random.choice(all_content, size=N, replace=True)
        hit = 0

        if len(order_history[1]) > 0 and np.intersect1d(order_history[1], top_n_result).size > 0:
            hit = 1

        return hit

    def top_n_recommends_popular(self, top_100_popular_items, order_history, N=40):
        top_n_result = top_100_popular_items[:N]
        hit = 0

        if len(order_history[1]) > 0 and np.intersect1d(order_history[1], top_n_result).size > 0:
            hit = 1
        return hit

    def top_n_recommends_personal(self, order_history, customer_id=0, N=50):
        top_n_result, recommended_scores = self.model.recommend(
                userid = customer_id,
                user_items=order_history[0],
                N=N,
                filter_already_liked_items=False,
                #recalculate_user=False
        )
        hit = 0
        if len(order_history[1]) > 0 and np.intersect1d(order_history[1], top_n_result).size > 0:
            hit = 1
        return hit

    def random_rec(self):
        #content_popularity = np.asarray(train_set_csr.sum(axis=0)).reshape(-1)
        all_content = np.array(list(self.id_to_product.keys())) 
        
        hits = [self.top_n_recommends_random(all_content, i) for i in self.train_valid_pairs]

        #print('Num hits %.4f from %d' % (sum(hits)/len(hits), len(hits)))
        return sum(hits)/len(hits), len(hits)

    def popularity_rec(self):

        content_popularity = np.asarray(self.train_set_csr.sum(axis=0)).reshape(-1)
        top_100_popular_items = np.argsort(-content_popularity)[:100]
        

        hits = [self.top_n_recommends_popular(top_100_popular_items, i) for i in self.train_valid_pairs]

        #print('Num hits %.4f from %d' % (sum(hits)/len(hits), len(hits)))
        return sum(hits)/len(hits), len(hits)

    def personal_rec(self):

        #testing_model = model


        print('Prediction started\n')
        hits = [self.top_n_recommends_personal(i) for i in self.train_valid_pairs]
        #print('Num hits %.4f from %d' % (sum(hits)/len(hits), len(hits)))
        return sum(hits)/len(hits), len(hits)
