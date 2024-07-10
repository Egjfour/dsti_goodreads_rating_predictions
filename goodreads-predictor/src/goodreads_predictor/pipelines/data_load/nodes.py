"""
This is a boilerplate pipeline 'data_load'
generated using Kedro 0.19.6
"""
import re
import urllib.parse
from typing import Tuple
import json
import pandas as pd
import numpy as np
import requests

def copy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Copy the input DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame to be copied.

    Returns:
        pd.DataFrame: The copied DataFrame.
    """
    return data

# SECTION - EXTERNAL DATA - SLUGBOOKS PRICING
# This section is used to get the pricing off of slugbooks.com which we will use as an additional feature

def make_request_slugbooks(isbn: str, action: str, do_amazon_check: bool, book_page: int, force_ajax: bool = False):
    """
    Perform an HTTP request to get the pricing from the webpage refresh from slugbooks.com.

    Args:
        isbn (str): The ISBN of the book.
        action (str): The action to perform.
        do_amazon_check (bool): A flag indicating whether to perform an Amazon check.
        book_page (int): The page number of the book.
        force_ajax (bool, optional): A flag indicating whether to force an Ajax search. Defaults to False.

    Returns:
        Tuple[requests.Response, bool]: A tuple containing the HTTP response and a flag indicating whether the response is in HTML format.
    """
    # Perform an initial request to see if we can parse the HTML
    is_html = True
    init_request = requests.get(f"https://www.slugbooks.com/searchAjax.php?action=SearchUrl&search={isbn}", timeout=5)
    if "ngSearchType = 'title'" in init_request.text or force_ajax:
        # If we didn't find an HTML page to ping, do the Ajax search
        url = ("https://www.slugbooks.com/doAjaxBookUpdate.php?"
            "action="
            + action
            + "&bookEAN="
            + str(isbn)
            + "&doAmaz="
            + str(do_amazon_check)
            + "&bookPage="
            + str(book_page))
        is_html = False
    else:
        url = re.sub("gSearchUrl = '", "", init_request.text.split("\n")[0])[:-2]

    return requests.request("GET", url, timeout=10), is_html

def request_slugbooks_isbn_by_title(uri_title: str):
    """
    Perform an API request to retrieve the ISBN of a book by its title from Slugbooks.

    Args:
        uri_title (str): The title of the book to search for.

    Returns:
        str: The ISBN of the book found on Slugbooks.

    Raises:
        requests.exceptions.RequestException: If there is an error during the API request.
    """
    # Get the actual URL with search results by name
    init_request = requests.get(f"https://www.slugbooks.com/searchAjax.php?action=SearchUrl&search={uri_title}&country=US&inMulti=1", timeout=10)
    url = re.sub("gSearchUrl = '", "", init_request.text.split("\n")[0])[:-2]

    # Search using the book name to return many results
    res = requests.request("GET", url, timeout=10).text

    # Take the first search result and get the ISBN Slugbooks uses
    new_isbn = re.sub("ISBN13: </span> ", "", re.search('ISBN13: </span> [0-9]+', res).group())
    return new_isbn

def search_for_list_price(response_text: str, is_html: bool):
    """
    Parse the response (HTML or API) to find the list price.

    Args:
        response_text (str): The response text to be parsed.
        is_html (bool): A flag indicating whether the response is in HTML format.

    Returns:
        re.Match or None: A match object containing the list price if found, or None if not found.
    """
    # Extract the list price (find the list_price key in regex because it is not a valid json)
    pattern = r"List Price:</span>(.*?)</li>" if is_html else r"'list_price':'(.*?)'"

    # Search for the pattern in the data
    match = re.search(pattern, response_text)

    return match

def get_list_price(isbn, action, book_title: str) -> Tuple[float, str]:
    """
    Attempts to extract the list price for a book given its ISBN, action, and title.

    Args:
        isbn (str): The ISBN of the book.
        action: The action to perform.
        book_title (str): The title of the book.

    Returns:
        Tuple[float, str]: A tuple containing the extracted list price and the strategy used to obtain it.

    Notes:
        This function attempts to extract the list price of a book by making requests to a website and parsing the response.
        If the list price is not available, it calculates the average price for all available buy options.
    """
    # Get the price from the listing page
    response, is_html = make_request_slugbooks(isbn, action, False, 0)

    first_match = search_for_list_price(response.text, is_html)

    # Extract the list price if found. Else try to lookup by name
    if first_match:
        list_price = first_match.group(1)
    else:
        title_uri = urllib.parse.quote(book_title)
        search_by_name_isbn = request_slugbooks_isbn_by_title(title_uri)
        response, is_html = make_request_slugbooks(search_by_name_isbn, action, False, 0)
        name_match = search_for_list_price(response.text, is_html)
        if name_match:
            list_price = name_match.group(1)
        else:
            list_price = None

    # Check if there is a valid list price
    try:
        # Replace the dollar sign with an empty string
        list_price = list_price.replace("$", "")

        # Transform the sup tags into dots
        list_price = list_price.replace("<span class=\"sup\">", ".")
        list_price = list_price.replace("</span>", "")
        list_price = list_price.replace("</li>", "")

        list_price = float(list_price)
        strategy = "LIST_PRICE"

        return list_price, strategy
    except ValueError:
        # return np.nan, 'INVALID'
        pass
    except AttributeError:
        # return np.nan, 'INVALID'
        pass

    # Perform a second query with different options
    response, is_html = make_request_slugbooks(isbn, action, True, 1, True)

    # Extract the buyList array
    pattern = r"'buyList':\[(.*?)\]"
    match = re.search(pattern, response.text)

    if match:
        buy_list = match.group(1)
    else:
        buy_list = None

    # Extract the prices from the buyList
    matches = re.findall(r"'price':'(.*?)'", buy_list)

    if not matches:
        strategy = "FAILURE"
        return [np.nan, strategy]
    else:
        strategy = "AVERAGE_PRICE"

    # Convert the prices to floats
    prices = [float(price.replace("$", "")) for price in matches]

    # Calculate the average price
    real_prices = [price for price in prices if price > 0]
    average_price = sum(real_prices) / (1 if len(real_prices) <= 0 else len(real_prices))

    return [average_price, strategy]

def query_slugbooks_price_data(input_data: pd.DataFrame, current_mapping: pd.DataFrame = None) -> pd.DataFrame:
    """
    Queries the slugbooks API to get pricing information for a given list of books.

    Args:
        input_data (pd.DataFrame): The input data containing ISBN and title information.
        current_mapping (pd.DataFrame, optional): The current mapping of ISBN to price data. Defaults to None.

    Returns:
        pd.DataFrame: The updated mapping of ISBN to price data.
    """
    # Filter out any ISBN values that we have already queried and stored
    if current_mapping is not None:
        filt_data = pd.merge(input_data, current_mapping, on='isbn13', how='left')
        filt_data = filt_data[filt_data['PriceStrategy'].isna()].reset_index(drop=True)
    else:
        filt_data = input_data

    # Loop through each dataframe row and determine the prices and strategies
    all_results = []
    for isbn, title in zip(filt_data['isbn13'], filt_data['title']):
        try:
            info = get_list_price(isbn, "getPriceUpdate", title)
            all_results.append({"isbn13": isbn, "Price": info[0], "PriceStrategy": info[1]})
        except Exception as e:
            print(f"{e} for ISBN {isbn}")
            # We will still append since we want to keep track of ALL the ISBNs we have queried
            all_results.append({"isbn13": isbn, "Price": -999, "PriceStrategy": 'FAILURE'})

    if current_mapping is not None:
        new_mapping = pd.concat([current_mapping, pd.DataFrame(all_results)], axis=0)
    else:
        new_mapping = pd.DataFrame(all_results)

    return new_mapping

def download_huggingface_book_info() -> pd.DataFrame:
    """
    Download book information from Hugging Face dataset.

    Returns:
        pd.DataFrame: DataFrame containing book information.
    """
    # Get the URL for the actual dataset
    data_url = json.loads(requests.get("https://huggingface.co/api/datasets/Eitanli/goodreads/parquet/default/train", timeout=30).text)

    # Load the dataset using the retrieved URL
    book_descs = pd.read_parquet(data_url[0])

    # Extract the goodreads_id from the URL
    book_descs['goodreads_id'] = [int(re.sub("show/","",re.findall('show/[0-9]+', url)[0])) for url in book_descs['URL']]
    
    return book_descs[['Book', 'Author', 'goodreads_id', 'Description', 'Genres']]
