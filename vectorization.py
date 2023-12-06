from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
#from tqdm import tqdm

class Vectorization:

    def __init__(self, test_data, content_to_id):
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

        orders_val, orders_test = train_test_split(self.orders_test, test_size=0.25, shuffle=True)

        test_group = orders_test.groupby('customer_id').apply(lambda cust: cust.product_id.unique()).reset_index(name='product_ids')
        val_group = orders_val.groupby('customer_id').apply(lambda cust: cust.product_id.unique()).reset_index(name='product_ids')

        test_group_slice = test_group['product_ids'].iloc[:200]
        val_group_slice = val_group['product_ids'].iloc[:300]


        test_dataset_vectors = [self.vectorize_action_history(i) for i in test_group_slice]
        ground_truth_dataset_vectors = [self.vectorize_action_history(i) for i in val_group_slice]

        return test_dataset_vectors, ground_truth_dataset_vectors
        #print('test dataset length: %d', len(test_dataset_vectors))
        #print('test dataset length: %d', len(ground_truth_dataset_vectors))

    def train_valid_union(self):
        train_valid_pairs = []

        test_dataset_vectors, ground_truth_dataset_vectors = self.df_vectors()

        for test_user_id in range(len(test_dataset_vectors)):
            train_valid_pairs.append((
                csr_matrix(test_dataset_vectors[test_user_id]),  # csr матрица на вход ALS
                ground_truth_dataset_vectors[test_user_id].nonzero()[0]
            ))
        return train_valid_pairs
