"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.6
"""
import re
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go # For the waterfall plot -- cannot easily do this in matplotlib

def join_data(books: pd.DataFrame, book_descriptions: pd.DataFrame, price_by_isbn: pd.DataFrame, book_genres: pd.DataFrame) -> pd.DataFrame:
    """
    Joins the given dataframes based on the 'isbn13' column.

    Args:
        books (pd.DataFrame): The dataframe containing book data.
        book_descriptions (pd.DataFrame): The dataframe containing book descriptions.
        price_by_isbn (pd.DataFrame): The dataframe containing price data.
        book_genres (pd.DataFrame): The dataframe containing book genres.

    Returns:
        pd.DataFrame: The merged dataframe with added price data and descriptions.
    """
    # Add in the price data
    books = books.merge(price_by_isbn, on='isbn13', how='left')

    # Add in the descriptions
    books = books.merge(book_descriptions, on='isbn13', how='left', validate="1:1")

    # Add in the genres (Merge is on the Goodreads Book ID -- This will make it more difficult to do inference later on new books outside Goodreads)
    books['bookID'] = books['bookID'].astype(str)
    books = books.merge(book_genres, left_on='bookID', right_on = '_id', how='left').rename(columns={"most_frequent_genre": "genre"})
    books['genre'] = books['genre'].fillna('no genre information')

    return books

def identify_most_common_author_by_isbn(books_input: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the most common author for each book title in the dataset.

    Args:
        books_input (pd.DataFrame): The input DataFrame containing book information.

    Returns:
        pd.DataFrame: A DataFrame with the most common author for each book title, indexed by ISBN.
    """
    # Identify how often authors are represented in the dataset overall
    author_frequency = (
        books_input[['isbn13', 'authors']]
        .assign(authors_list = lambda x: x['authors'].str.split('/'))
        .explode('authors_list')
        .assign(authors_list = lambda x: x['authors_list'].str.replace("\s+", " ").str.strip())
        .groupby('authors_list')
        .agg(
            Ct = ('isbn13', 'nunique')
        )
        .sort_values('Ct', ascending=False)
    )

    # Figure out which author is most common by ISBN
    most_common_author_by_isbn = (
        books_input[['isbn13', 'authors']]
        .assign(authors_list = lambda x: x['authors'].str.split('/'))
        .explode('authors_list')
        .assign(authors_list = lambda x: x['authors_list'].str.strip())
        .merge(author_frequency, left_on='authors_list', right_index=True, validate='m:1')
        .sort_values(['Ct', 'authors_list'], ascending=False)
        .groupby('isbn13')
        .agg(
            most_common_author = ('authors_list', 'first')
        )
    )

    return most_common_author_by_isbn

def consolidate_duplicated_titles(data: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidates duplicated titles in the given DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing the data to be consolidated.

    Returns:
        pd.DataFrame: The consolidated DataFrame with duplicated titles removed and aggregated values.

    """
    # Identify the proper description to grab
    longest_descriptions_by_title = (
        data
        .assign(DescriptionLength = lambda x: x['Description'].str.len())
        .sort_values('DescriptionLength', ascending = False)
        .groupby('title')
        .agg(
            LongestDescription = ('Description', 'first'),
            DescriptionISBN = ('isbn13', 'first') # Need this to match the book description embeddings
        )
    )

    # Aggregate at each unique title
    return (
        data
        .merge(longest_descriptions_by_title, left_on='title', right_index = True, how='left')
        .sort_values('ratings_count', ascending = False)
        .groupby('title')
        .apply(lambda g: pd.Series({
            'isbn': g['isbn'].values[0],
            'isbn13': g['isbn13'].values[0],
            'most_common_author': g['most_common_author'].mode().values[0],
            'average_rating': (g['average_rating'] * g['ratings_count']).sum() / g['ratings_count'].sum(),
            'num_pages': g['  num_pages'].mean(),
            'ratings_count': g['ratings_count'].sum(),
            'text_reviews_count': g['text_reviews_count'].sum(),
            'publisher': g['publisher'].mode().values[0],
            'Price': g[g['Price'] > 0]['Price'].mean(),
            'language_code': g['language_code'].mode().values[0],
            'Description': g['LongestDescription'].values[0],
            'DescriptionISBN': g['DescriptionISBN'].values[0],
            'genre': g['genre'].mode().values[0],
        }), include_groups = False)
        .assign(Price = lambda x: x['Price'].fillna(-999))
        .reset_index(drop = False)
    )

def get_description_from_json(record) -> str:
    """
    Get the description from the JSON record.

    Args:
        record (Dict[str, Any]): The JSON record.

    Returns:
        str: The description of the book.
    """
    description = 'No Description Found'
    if 'description' in record.keys():
        description = record['description']

        # Check if it is a dictionary
        if type(description) == dict:
            description = description.get('value', 'No Description Found')
    elif record.get('isbn_description', []) != []:
        description = record['isbn_description']
    
        if type(description) == dict:
            description = description.get('value', 'No Description Found')
    elif 'subjects' in record.keys():
        description = ', '.join(record['subjects'])
    elif record.get('isbn_subjects', []) != []:
        description = ', '.join(record['isbn_subjects'])

    description = re.sub('[^A-z0-9\.!\?,]', ' ', description).lower()
    description = re.sub('\s+', ' ', description).strip()

    return description

def create_descriptions_lookup(book_api_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a lookup table for book descriptions based on book API data.

    Args:
        book_api_data (List[Dict[str, Any]]): A list of dictionaries containing book API data.

    Returns:
        pd.DataFrame: A DataFrame with two columns: 'isbn13' and 'Description'.
                     The 'isbn13' column contains the ISBN-13 numbers of the books,
                     and the 'Description' column contains the corresponding book descriptions.
    """
    descs = [get_description_from_json(x) for x in book_api_data]
    descriptions_by_isbn = pd.DataFrame({
        'isbn13': [x.get('isbn13', 'No ISBN') for x in book_api_data],
        'Description': descs
    })

    return descriptions_by_isbn

def create_data_filters(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create data filters based on specific conditions in the input dataframe.

    Args:
        input_df (pd.DataFrame): The input dataframe containing the data to filter.

    Returns:
        pd.DataFrame: The filtered dataframe with an additional column indicating the filter reason.
    """

    # Initialize the series to hold the filters
    filt_num = 1
    input_df['data_filter_reason'] = 'Start - No Filter'
 
    # No ratings or pages
    input_df.loc[(input_df['ratings_count'] <= 0) & (input_df['data_filter_reason'] == 'Start - No Filter'),
     'data_filter_reason'] = f'{filt_num} - No Ratings Count'
    filt_num += 1

    input_df.loc[(input_df['  num_pages'] <= 0) & (input_df['data_filter_reason'] == 'Start - No Filter'),
     'data_filter_reason'] = f'{filt_num} - Zero Page Count'
    filt_num += 1

    # Extreme tails of the avg_ratings (looked at histogram to determine values)
    input_df.loc[(input_df['average_rating'] < 3) & (input_df['data_filter_reason'] == 'Start - No Filter'),
     'data_filter_reason'] = f'{filt_num} - Low Rating Outlier'
    filt_num += 1

    input_df.loc[(input_df['average_rating'] >4.95) & (input_df['data_filter_reason'] == 'Start - No Filter'),
     'data_filter_reason'] = f'{filt_num} - High Rating Outlier'
    filt_num += 1

    # No Genre
    input_df.loc[(input_df['genre'] == 'no genre information') & (input_df['data_filter_reason'] == 'Start - No Filter'),
                 'data_filter_reason'] = f'{filt_num} - No Genre Data'
    filt_num += 1

    # No description
    input_df.loc[(input_df['Description'] == 'no description found') & (input_df['data_filter_reason'] == 'Start - No Filter'),
                 'data_filter_reason'] = f'{filt_num} - No Description'
    filt_num += 1

    # Duplicated titles
    input_df = (
        input_df
        .merge(
            input_df.groupby('title').agg(
                isbn13 = ('isbn13', 'first')
            ).rename(columns={'isbn13': 'isbn13_grouped'}),
            left_on='title', right_index=True, how = 'left', validate = "m:1"
            )
        .assign(non_duplicate = lambda X: X['isbn13'] == X['isbn13_grouped'])
    )
    input_df.loc[(input_df['non_duplicate'] == False) & (input_df['data_filter_reason'] == 'Start - No Filter'),
                  'data_filter_reason'] = f'{filt_num} - Duplicate Title'
    filt_num += 1

    # Final data -- what's not filtered (we want this to be last for the waterfall chart)
    input_df.loc[input_df['data_filter_reason'] == 'Start - No Filter', 'data_filter_reason'] = f'{filt_num} - Included in Model'
    
    return input_df

def aggregate_exclusions_data(inputs_filters: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the exclusion data based on the input filters.

    Args:
        inputs_filters (pd.DataFrame): The input filters dataframe.

    Returns:
        filters_agg (pd.DataFrame): The aggregated exclusion data at the reason level.
        walk_data (pd.DataFrame): The walk data showing the count of books excluded for each reason.
    """
    # Aggregate at the exclusion reason level and get a count of the books excluded for each reason
    filters_agg = inputs_filters.groupby('data_filter_reason', as_index=False).agg(
        Ct = ('bookID', 'count')
    )

    # Append the add data field to create a walk from starting input to final filtered dataset
    walk_data = pd.concat([pd.DataFrame({'data_filter_reason': 'All Data', 'Ct': filters_agg['Ct'].sum()}, index = [0]), filters_agg], axis = 0).reset_index(drop=True)
    walk_data['Ct'] = np.where((walk_data.index > 0) & (walk_data.index < walk_data.index.max()),  walk_data['Ct'] * -1, walk_data['Ct'])

    return filters_agg, walk_data

def create_filters_waterfall_plot(walk_data: pd.DataFrame,
                                  decreasing_color: str,
                                  increasing_color: str,
                                  totals_color: str,
                                  background_color: str) -> go.Figure:
    """
    Create a waterfall plot based on the provided data.

    Args:
        walk_data (pd.DataFrame): The data used to create the waterfall plot.
        decreasing_color (str): The color for the decreasing values in the plot.
        increasing_color (str): The color for the increasing values in the plot.
        totals_color (str): The color for the total values in the plot.
        background_color (str): The background color of the plot.

    Returns:
        go.Figure: The created waterfall plot.
    """
    
    # Create dynamic fields based on data for the waterfall plot
    dynamic_measures = np.where((walk_data.index > 0) & (walk_data.index < walk_data.index.max()),  'relative', 'absolute')

    text_values = np.where(walk_data['Ct'].values > 1000,
                            [x + ' k' for x in (walk_data['Ct'].values / 1000).round(1).astype(str)],
                            walk_data['Ct'].astype(str).values)

    # Create the waterfall plot base using the data
    fig = go.Figure(go.Waterfall(
        name='Goodreads Ratings Prediction Model Data Exclusions',
        orientation='v',
        measure=dynamic_measures,
        x=walk_data['data_filter_reason'].values,
        textposition='outside',
        text=text_values,
        y=walk_data['Ct'].values,
        connector=dict(line=dict(color='rgb(63, 63, 63)')),
        decreasing=dict(marker=dict(color=decreasing_color)),
        increasing=dict(marker=dict(color=increasing_color)),
        totals=dict(marker=dict(color=totals_color))
    ))

    # Update the styling of the plot
    fig.update_layout(
        font_family = 'Helvetica',
        title='Goodreads Ratings Prediction Model Scope Walk',
        title_x=0.5,
        showlegend=False,
        plot_bgcolor=background_color,
        title_font_family='Lato',
        title_font_size=20
    )
    
    # Update axis titles and styling
    fig.update_yaxes(range=[-1,walk_data['Ct'].values.max() * 1.1], title_text='Number of Records', title_font_family='Lato')
    fig.update_xaxes(title_text='Data Filter Reason', title_font_family='Lato')

    return fig

def apply_filters_and_consolidate(books_filters: pd.DataFrame) -> pd.DataFrame:
    """
    Apply filters to the books and consolidate duplicated titles.

    Args:
        books_filters (pd.DataFrame): DataFrame containing the books and their filters.

    Returns:
        pd.DataFrame: DataFrame with the filtered and consolidated books.
    """
    # Apply the filters and consolidate duplicated titles
    books_filtered = books_filters[(books_filters['data_filter_reason'].str.contains('Included in Model')) |
                                (books_filters['data_filter_reason'].str.contains('Duplicate Title'))]

    books_filtered = books_filtered.merge(identify_most_common_author_by_isbn(books_filtered), left_on='isbn13', right_index=True, how='left')
    books_filtered = consolidate_duplicated_titles(books_filtered)
    return books_filtered

def apply_publisher_consolidation(books_filtered: pd.DataFrame, publishers: pd.DataFrame) -> pd.DataFrame:
    """
    Apply publisher consolidation to the filtered books DataFrame.

    Args:
        books_filtered (pd.DataFrame): The filtered books DataFrame.
        publishers (pd.DataFrame): The DataFrame containing manually grouped publishers.

    Returns:
        pd.DataFrame: The updated books DataFrame with consolidated publishers.
    """

    # Merge in the manually grouped publishers
    books_filtered = books_filtered.merge(publishers, on='publisher', how='left')

    # Coalesce to fill in the original publisher where there is no grouped publisher
    books_filtered['publisher_grouped'] = np.where(books_filtered['publisher_grouped'].isnull(),
                                                   books_filtered['publisher'], books_filtered['publisher_grouped'])
    
    return books_filtered
