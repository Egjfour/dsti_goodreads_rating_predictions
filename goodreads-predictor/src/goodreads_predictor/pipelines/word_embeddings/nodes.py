# nodes.py

import pandas as pd
from sentence_transformers import SentenceTransformer

def create_embeddings(books_loaded):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_parquet(books_loaded)
    df['title_embedding'] = df['title'].apply(lambda x: model.encode(x).tolist())
    df['author_embedding'] = df['authors'].apply(lambda x: model.encode(x).tolist())
    return df

