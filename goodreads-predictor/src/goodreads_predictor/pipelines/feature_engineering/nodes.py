"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.6
"""
from typing_extensions import Tuple, List, Union, Dict, Any
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap.umap_ as umap


def cut_variable_count_by_mean_counts(input_df: pd.DataFrame, column: str, mapping = List[str], bin_col_name: str = 'binned') -> Tuple[pd.DataFrame, float, float]:
    """
    Cuts a variable into bins based on the mean of the variable and mean of the varaible not including 1

    Args:
        input_df: source DataFrame
        column: the column to cut
        mapping: An ordered list of the
        bin_col_name: the name of the new column

    Returns:
        pd.DataFrame: A DataFrame containing the binned variable
        float: The mean of the counts
        float: The mean of the counts excluding 1
    """
    # Get the counts of the variable
    count_data = pd.DataFrame(input_df.value_counts(column))

    # Get the mean of the counts and mean of the counts excluding 1
    mean_count = count_data['count'].mean()
    mean_count_no_1 = count_data[count_data['count'] > 1]['count'].mean()

    # Apply the mapping
    output_array = np.where(count_data['count'] < mean_count, mapping[0], np.where(count_data['count'] < mean_count_no_1, mapping[1], mapping[2]))
    count_data[bin_col_name] = output_array

    return count_data.drop(columns = 'count'), mean_count, mean_count_no_1

def calculate_author_fame_levels(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the fame levels of authors based on their total review count.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.

    Returns:
        pd.DataFrame: A DataFrame containing the fame levels of authors.
                      The DataFrame has a single column named 'AuthorFameLevel'.

    Notes:
        The fame levels are mapped as follows:
        - 1: "1 - Very Low Exposure"
        - 2: "2 - Not Well-Known"
        - 3: "3 - Well-Known"
        - 4: "4 - Popular"
        - 5: "5 - Semi-Famous"
        - 6: "6 - Famous"
    """
    author_fame_levels = ...
    # Use percentiles of the total review count to get a sense of how well-known an author is
    author_fame_levels = (
        data
        # Get total review count by author
        .groupby("most_common_author")
        .agg(TotalReviews = ("ratings_count", "sum"))
        .sort_values("TotalReviews", ascending=False)

        # Get quantiles of the total review count (list comprehension to repeat the quantiles for each author)
        .assign(QuantileReviews = lambda x: x['TotalReviews'].quantile([0, 0.15, 0.25, 0.5, 0.75, 0.95]).values.repeat(len(x)).reshape(-1, len(x)).T.tolist())

        # Explode the quantiles to get one row per quantile then sort by quantile
        .explode("QuantileReviews")
        .sort_values("QuantileReviews", ascending=True)

        # Create a quantile index for each author
        .assign(QuantileRank = lambda x: x.groupby("most_common_author").cumcount().add(1))

        # Filter to where the total review count is greater than the quantile review count or we are at the lowest quantile possible
        .assign(ReviewsDiff = lambda x: x['TotalReviews'] - x['QuantileReviews'])
        .assign(MaxReviewsDiff = lambda x: x.groupby("most_common_author")["ReviewsDiff"].transform("max"))
        .assign(AbsReviewsDiff = lambda x: abs(x["ReviewsDiff"]))
        .query("ReviewsDiff > 0 or MaxReviewsDiff <= 0")

        # Get the first row for each author (the lowest quantile where the total review count is greater)
        .sort_values("AbsReviewsDiff", ascending=True)
        .groupby("most_common_author")
        .first()

        # Map the quantile rank to a string ("Unknown", "Not Well-Known", "Well-Known", "Very Well-Known", "Famous")
        .assign(AuthorFameLevel = lambda x: x["QuantileRank"].map({1: "1 - Very Low Exposure", 2: "2 - Not Well-Known", 3: "3 - Well-Known",
                                                                   4: "4 - Popular", 5:"5 - Semi-Famous", 6: "6 - Famous"}))
        .sort_values("TotalReviews", ascending=False)
        [['AuthorFameLevel']]
    )

    return author_fame_levels

def calculate_publisher_price_categories(data: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    """
    Calculate the price categories for publishers based on the median price of books.

    Args:
        data (pd.DataFrame): The input DataFrame containing book data.

    Returns:
        pd.DataFrame: A DataFrame containing the price categories for publishers.

        float: The median of median publisher prices.
    """
    # Get the median price of books for each publisher
    price_data = data.query('Price > 0').groupby('publisher').agg(MedianPrice = ('Price', 'median'))

    # Get the median of the medians to use as a cutoff point
    median_of_medians = price_data['MedianPrice'].median()

    # Categorize publishers based on their median price
    price_data['PublisherPriceCategory'] = np.where(price_data['MedianPrice'] <= median_of_medians, "1 - Low Price Publisher", "2 - High Price Publisher")
    return price_data, median_of_medians

def calculate_reviewer_engagement_level(data: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    """
    Calculate the engagement level of reviewers based on the proportion of text reviews to total reviews.
    Args:
        data (pd.DataFrame): The input DataFrame containing the necessary columns:
            - 'text_reviews_count': The number of text reviews.
            - 'ratings_count': The total number of reviews.
    Returns:
        np.ndarray: An array of engagement levels for each reviewer, where:
            - '1 - Passive Reviewers': The engagement level for reviewers with a text reviews percentage below the 25th percentile.
            - '2 - Standard Reviewers': The engagement level for reviewers with a text reviews percentage between the 25th and 50th percentiles.
            - '3 - Engaged Reviewers': The engagement level for reviewers with a text reviews percentage above the 50th percentile.
    """
    # We'll use the proportion of text reviews to total reviews as a proxy for reviewer engagement
    data['text_reviews_percentage'] = data['text_reviews_count'] / data['ratings_count']

    # Use the IQR as cutoffs for the engagement levels
    quantiles = data['text_reviews_percentage'].quantile([0.25, 0.5, 0.75])
    engagement_levels = np.where(data['text_reviews_percentage'] <= quantiles[0.25], "1 - Passive Reviewers",
                                 np.where(data['text_reviews_percentage'] <= quantiles[0.5], "2 - Standard Reviewers",
                                          np.where(data['text_reviews_percentage'] <= quantiles[0.75], "3 - Engaged Reviewes", '4 - Highly Engaged Reviewers')))
    
    return engagement_levels, quantiles

def categorize_book_length(book_lengths: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    """
    Categorizes book lengths into different categories based on the number of pages.

    Args:
        book_lengths (Union[List[float], np.ndarray, pd.Series]): A list, numpy array, or pandas Series containing the lengths of books in pages.

    Returns:
        np.ndarray: An array containing the category index for each book length.
    """
    # Define the categories based on the number of pages
    categories = ['1 - Short Stories (1-271 pages)', '2 - Standard Books (272-524 pages)',
                    '3 - Long Books (525 - 1049 pages)', '4 - Very Long Books (1050+ pages)']
    cutoff_mins = np.array([0, 271, 524, 1049])
    categories_dict = dict(zip(cutoff_mins, categories))

    # Convert input to numpy array if it's a pandas Series
    if isinstance(book_lengths, pd.Series):
        book_lengths = book_lengths.values

    # Find the largest minimum cutoff that is less than or equal to each book length
    filtered = cutoff_mins[:, None] <= book_lengths
    page_cutoffs = np.max(np.where(filtered, cutoff_mins[:, None], -np.inf), axis=0)

    # Map the cutoffs to the corresponding categories
    return np.vectorize(categories_dict.get)(page_cutoffs)

def consolidate_english_language_codes(data: pd.DataFrame) -> np.ndarray:
    """
    Consolidate the English language codes in the dataset.

    Args:
        data (pd.DataFrame): The input DataFrame containing the language codes.

    Returns:
        np.ndarray: An array containing the consolidated language codes.
    """
    return np.where(data['language_code'].isin(['eng', 'en-US', 'en-GB', 'en-CA']), 'eng', data['language_code'])

def apply_book_attributes(input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    # Calculate the fame level for the author
    author_fame_levels = calculate_author_fame_levels(input_df)

    # Calculate the price categories for publishers
    publisher_price_categories, median_publisher_price = calculate_publisher_price_categories(input_df)

    # Calculate the engagement level for reviewers
    reviewer_engagement_level, engagement_quantiles = calculate_reviewer_engagement_level(input_df)

    # Count of Books written by Author
    (author_book_counts,
     author_mean_book_counts,
     author_mean_counts_no_ones) = cut_variable_count_by_mean_counts(input_df, 'most_common_author',
                                                                     ["1 - Few Books Written", "2 - Some Books Written", "3 - Many Books Written"], 'book_count_category')
    
    # Count of Books written by Publisher
    (publisher_book_counts,
     publisher_mean_book_counts,
     publisher_mean_counts_no_ones) = cut_variable_count_by_mean_counts(input_df, 'publisher',
                                                                        ["1 - Few Books Published", "2 - Some Books Published", "3 - Many Books Published"], 'publisher_book_count_category')
    
    # Consolidate the language codes
    consolidated_language_codes = consolidate_english_language_codes(input_df)
    
    # Save a dictionary for the cutoffs to apply at inference time
    cutoff_dict = {"median_publisher_price": median_publisher_price,
                   "engagement_quantile_cutoffs": list(engagement_quantiles.values),
                   "author_mean_book_counts": author_mean_book_counts,
                   "author_mean_counts_no_ones": author_mean_counts_no_ones,
                   "publisher_mean_book_counts": publisher_mean_book_counts,
                   "publisher_mean_counts_no_ones": publisher_mean_counts_no_ones}
    
    # Create author and publisher lookup tables with the information at that level of granularity
    author_metrics = author_fame_levels.merge(author_book_counts, left_index=True, right_index=True)
    publisher_metrics = publisher_book_counts.merge(publisher_price_categories, left_index=True, right_index=True, how='left')
    publisher_metrics.drop(columns='MedianPrice', inplace=True)
    publisher_metrics['PublisherPriceCategory'] = publisher_metrics['PublisherPriceCategory'].fillna("Unknown")

    # Merge the data with the calculated metrics to the original data
    output_df = (
        input_df
        .merge(author_metrics, left_on='most_common_author', right_index=True, how='left')
        .merge(publisher_metrics, left_on='publisher', right_index=True, how='left')
        )
    
    # Add the book-level attributes
    output_df['book_length_category'] = categorize_book_length(output_df['num_pages'])
    output_df['engagement_level'] = reviewer_engagement_level
    output_df['language_code'] = consolidated_language_codes

    return output_df, cutoff_dict

# Defin ethe function to merge the output_df with the description embedding from catalog
def merge_description_embeddings(output_df: pd.DataFrame, description_embeddings: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the description embeddings with the output DataFrame.
    Args:
        output_df (pd.DataFrame): The output DataFrame.
        description_embeddings (pd.DataFrame): The description embeddings DataFrame.

    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    # Merge the description embeddings with the output DataFrame
    merged_df = pd.merge(output_df, description_embeddings, left_on='DescriptionISBN', right_on='isbn13')
    return merged_df

# Define the function to perform clustering analysis

def perform_clustering_analysis(merged_df: pd.DataFrame, n_clusters: int = 10) -> pd.DataFrame:
    # Convert the 'description_embedding' column to a numpy array if it's not already
    embeddings = np.vstack(merged_df['Description_embedding'].values)

    # Flatten the embeddings to 1D
    embeddings = embeddings.reshape(embeddings.shape[0], -1)

    # Lower the dimension of the description_embedding column using UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)  # Use 2 components for better visualization
    reduced_embeddings = umap_model.fit_transform(embeddings)

    # Perform KMeans clustering on the reduced embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    merged_df['cluster'] = kmeans.fit_predict(reduced_embeddings)

    # Calculate the silhouette score
    silhouette_avg = silhouette_score(reduced_embeddings, kmeans.labels_)
    print("Silhouette Score:", silhouette_avg)

    # Add the reduced dimensions to the DataFrame for visualization
    merged_df['UMAP1'] = reduced_embeddings[:, 0]
    merged_df['UMAP2'] = reduced_embeddings[:, 1]

    # Drop the original 'description_embedding' column
    merged_df.drop(columns=['Description_embedding'], inplace=True)

    return merged_df


