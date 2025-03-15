import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np


df = pd.read_csv('amazon.csv')

# Convert 'rating' column to numeric (force errors='coerce' to handle invalid values)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df = df.dropna(subset=['rating'])
df['rating'] = df['rating'].astype(int)

# Create user-item matrix after fixing duplicates
df = df.groupby(['user_id', 'product_id']).agg({'rating': 'mean'}).reset_index()
user_item_matrix = df.pivot(index='user_id', columns='product_id', values='rating').fillna(0)
print("User-Item Matrix Shape:", user_item_matrix.shape)


sparse_matrix = csr_matrix(user_item_matrix)

knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
knn.fit(sparse_matrix)


user_index = np.random.choice(user_item_matrix.shape[0])
distances, indices = knn.kneighbors([sparse_matrix[user_index].toarray()], n_neighbors=6)
recommended_products = user_item_matrix.iloc[indices.flatten()[1:]].mean(axis=0).sort_values(ascending=False).index[:5]
print("Recommended Products:", recommended_products.tolist())
