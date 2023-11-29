import requests
from bs4 import BeautifulSoup

def symbol_exists_yahoo_finance(symbol):
    url = f"https://finance.yahoo.com/quote/{symbol}"
    response = requests.get(url)
    
    # Check if the page exists (status code 200)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.title.string

        # If the title contains the symbol, it exists
        if symbol in title:
            return True
        else:
            return False
    else:
        return False

# Example usage
symbol_to_check = "KALPOW.NS"  # Replace with the symbol you want to check
if symbol_exists_yahoo_finance(symbol_to_check):
    print(f"{symbol_to_check} is present in Yahoo Finance.")
else:
    print(f"{symbol_to_check} is not found in Yahoo Finance.")
