from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np

class Vectorization:

    def __init__(self,train_data, test_data, content_to_id):
        self.orders_train = train_data
        self.orders_test = test_data
        self.product_to_id = content_to_id

    def get_als_action_history_vector(self, item_to_id, action_history, binary=True) -> np.ndarray:
        """Получить историю действий для ALS

        :param item_to_id: справочник контента ALS
        :return:
        """
        als_action_history_vector = np.zeros(len(item_to_id), dtype=int)

        for iid in action_history:
        
            if iid in item_to_id.keys():

                if binary:
                    als_action_history_vector[item_to_id[iid]] = 1

        return als_action_history_vector

    def vectorize_action_history(self, action_history):
        res = self.get_als_action_history_vector(self.product_to_id, action_history)
        return res

    def df_vectors(self):

        orders_val, orders_test = self.orders_train, self.orders_test

        test_group = orders_test.groupby('customer_id').apply(lambda cust: cust.product_id.unique()).reset_index(name='product_ids')
        val_group = orders_val.groupby('customer_id').apply(lambda cust: cust.product_id.unique()).reset_index(name='product_ids')

        test_group_slice = test_group.iloc[::]
        val_group_slice = val_group.iloc[::]

        test_dataset_vectors = { row['customer_id'] : self.vectorize_action_history(row['product_ids']) for i, row in test_group_slice.iterrows() }
        
        for key in set(test_dataset_vectors.keys()):
            if sum(test_dataset_vectors[key]) == 0:
                del test_dataset_vectors[key]

        ground_truth_dataset_vectors = { row['customer_id'] : self.vectorize_action_history(row['product_ids']) for i, row in val_group_slice.iterrows() if row['customer_id'] in  test_dataset_vectors}


        return test_dataset_vectors, ground_truth_dataset_vectors

    def train_valid_union(self):
        train_valid_pairs = []

        test_dataset_vectors, ground_truth_dataset_vectors = self.df_vectors()

        for customer_id in test_dataset_vectors.keys():

            if customer_id in ground_truth_dataset_vectors.keys():
                train_valid_pairs.append((
                csr_matrix(test_dataset_vectors[customer_id]),  # csr матрица на вход ALS
                ground_truth_dataset_vectors[customer_id].nonzero()[0]
            ))

            
        return train_valid_pairs
