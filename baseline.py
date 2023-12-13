import collections
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np


class BaseLine:

    def __init__(self, train_df, test_df):
        self.orders_train = train_df
        self.orders_test = test_df
        

    def indexing(self):
        unique_customers = self.orders_train.customer_id.unique()
        unique_products = self.orders_train.product_id.unique()

        # прямая и обратная индексация контента
        product_to_id = {j: i for i, j in enumerate(unique_products)}
        id_to_product = {j: i for i, j in product_to_id.items()}

        # прямая индексация пользователей
        customer_to_id = {j: i for i, j in enumerate(unique_customers)}
        #print('Индекс создан: %d строк %d столбцов' % (len(customer_to_id), len(product_to_id)))
        return product_to_id, id_to_product, customer_to_id

    def last_views_top(self):
        HISTORY_TOP = 50
        # применяем оконную функцию
        self.orders_train['rank'] = (self.orders_train.groupby(by=['customer_id'])['order_day'].rank(method='first', ascending=True))

        ui_slim_df = (self.orders_train[self.orders_train['rank'] < HISTORY_TOP][['customer_id', 'product_id']])


        entries = np.ones(ui_slim_df.shape[0])
        return ui_slim_df, entries

    def create_csr(self, customer_to_id, product_to_id):

        ui_slim_df, entries = self.last_views_top()

        num_rows = len(customer_to_id)
        num_cols = len(product_to_id)
        entries = np.ones(ui_slim_df.shape[0])  # проставляем ненулевые элементы - единицы
        rows = tuple(customer_to_id[i] for i in ui_slim_df.customer_id.values)
        cols = tuple(product_to_id[i] for i in ui_slim_df.product_id.values)

        train_set_csr = csr_matrix(
            (entries, (rows, cols)),
            shape=(num_rows, num_cols),
            dtype=np.float32)

        return train_set_csr

    def model_als_fit(self, train_set_csr, customer_to_id, product_to_id, factors=4, iterations=1):
        
        implict_als_params = {'factors': factors, 'alpha': 0.2, 'iterations': iterations, 'calculate_training_loss': True}
        model = AlternatingLeastSquares(**implict_als_params)

        # на вход подаём матрицу user_item
        model.fit(train_set_csr)
        return model


