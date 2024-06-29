# nodes.py

from sentence_transformers import SentenceTransformer

def create_embeddings(books_loaded):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    books_loaded['title_embedding'] = books_loaded['title'].apply(lambda x: model.encode(x).tolist())
    books_loaded['author_embedding'] = books_loaded['authors'].apply(lambda x: model.encode(x).tolist())
    return books_loaded
