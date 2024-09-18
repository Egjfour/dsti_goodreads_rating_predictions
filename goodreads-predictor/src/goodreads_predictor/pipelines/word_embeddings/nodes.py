import pandas as pd
from sentence_transformers import SentenceTransformer # pylint: disable=E0401

def init_model() -> SentenceTransformer:
    """
    Initializes and returns a SentenceTransformer model.

    Returns:
        SentenceTransformer: The initialized SentenceTransformer model.
    """
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def create_embeddings(model: SentenceTransformer,
                      input_data: pd.DataFrame,
                      embedding_column: str,
                      key_column: str) -> pd.DataFrame:
    """
    Create embeddings for the given input data using the provided model.

    Args:
        model (SentenceTransformer): The pre-trained sentence embedding model.
        input_data (pd.DataFrame): The input data containing the text to be embedded.
        embedding_column (str): The name of the column in input_data to embed.
        key_column (str): The name of the column in input_data to use as a key.

    Returns:
        pd.DataFrame: A DataFrame containing the key column and the corresponding embedding column.
    """
    # Remove null records
    data_to_process = input_data[input_data[embedding_column].notnull()]

    # Execute the model against the chosen column -- have the model operate on a list for performance reasons
    data_to_process[f'{embedding_column}_embedding'] = model.encode(data_to_process[embedding_column].values.tolist()).tolist()

    # Return the key column and the embedding column as a lookup table
    return data_to_process[[key_column, f'{embedding_column}_embedding']]
