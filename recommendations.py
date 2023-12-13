import pandas as pd
import numpy as np

class Recommendations:

    def __init__(self, model, content_df: pd.DataFrame, train_valid_pairs, train_set_csr, id_to_product, product_to_id, customer_to_id):
        self.products_data = content_df
        self.train_valid_pairs = train_valid_pairs
        self.train_set_csr = train_set_csr
        self.id_to_product = id_to_product
        self.product_to_id = product_to_id
        self.customer_to_id = customer_to_id
        self.model = model


    def id_to_content_df(self, ids: np.array, id_to_item):
        items = tuple(id_to_item[i] for i in ids)

        result_df = self.products_data[self.products_data.product_id.isin(items)]
        return result_df

    def cust_history_recommend(self, customer_id, N=10):

        recommended_content, recommended_scores = self.model.recommend(
                    userid = customer_id,
                    user_items=self.train_valid_pairs[customer_id][0],
                    N=N,
                    filter_already_liked_items=True,
                    #recalculate_user=True
        )
        res = self.id_to_content_df(recommended_content, self.id_to_product)

        return res

    def top_n_recommends(self, order_history, top_n_result):

        hit = 0

        if len(order_history[1]) > 0 and np.intersect1d(order_history[1], top_n_result).size > 0:
            hit = 1

        return hit


    def top_n_recommends_personal(self, order_history, customer_id, N=50):
        top_n_result, recommended_scores = self.model.recommend(
                userid=customer_id,
                user_items=order_history[0],
                N=N,
                filter_already_liked_items=False,
                recalculate_user=True
        )
        hit = 0
        if len(order_history[1]) > 0 and np.intersect1d(order_history[1], top_n_result).size > 0:
            hit = 1

        return hit

    def random_rec(self, N=10):

        all_content = np.array(list(self.id_to_product.keys())) 
        top_n_result = np.random.choice(all_content, size=N, replace=True)
        
        hits = [self.top_n_recommends(i, top_n_result) for i in self.train_valid_pairs]

        #print('Random hits %.4f from %d' % (sum(hits)/len(hits)*100, len(hits)))

        res = self.id_to_content_df(top_n_result, self.id_to_product)

        return res

    def popularity_rec(self, N=10):

        content_popularity = np.asarray(self.train_set_csr.sum(axis=0)).reshape(-1)
        top_100_popular_items = np.argsort(-content_popularity)[:100]

        top_n_result = top_100_popular_items[:N]

        hits = [self.top_n_recommends(i, top_n_result) for i in self.train_valid_pairs]

        #print('Popularity hits %.4f from %d' % (sum(hits)/len(hits)*100, len(hits)))

        res = self.id_to_content_df(top_n_result, self.id_to_product)

        return res

    def personal_rec(self, customer_id_str='f374c8c54c'):

        customer_id = self.customer_to_id[customer_id_str]

        hits = [self.top_n_recommends_personal(i, customer_id) for i in self.train_valid_pairs]
        #print('Personal hits %.4f from %d' % (sum(hits)/len(hits)*100, len(hits)))
        
        res = self.cust_history_recommend(customer_id)

        return res
