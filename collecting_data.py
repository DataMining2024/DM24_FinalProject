# Import Library
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import requests, tqdm
from bs4 import BeautifulSoup
import time
import FinanceDataReader as fdr
import yfinance as yf
yf.pdr_override()
import os

# Download Ticker
## SP500
sp500 = fdr.StockListing('S&P500')
sp500_ticker = sp500['Symbol']
## Shanghai 
sse = fdr.StockListing('SSE')
sse_ticker = sse['Symbol']
## Shenzhen
szse = fdr.StockListing('SZSE')
szse_ticker = szse['Symbol']

# Crawling Stock Price Function
def download_stocks(ticker_li, start_date, end_date, file_path):
    cnt = 0
    for index, row in ticker_li.iterrows():
        # SP500 Exchange Market
        if row['Exchange'] == 'SP500':
            ticker = row['Symbol']
            stock_df = fdr.DataReader(ticker, start_date, end_date).reset_index()
            if stock_df.empty:
                print(f"No data available for {ticker}. Skipping...")
                continue
            else:
                cnt += 1
                file_name = f'{cnt:03d}_SP500_raw_{ticker}.csv'
                stock_df.to_csv(file_path + file_name, index=False)
            print(f"Data downloaded for {ticker} and saved to {file_name}")

        # Shanghai Exchange Market
        elif row['Exchange'] == 'Shanghai Stock Exchange':
            ticker = (str(row['Ticker']).zfill(6)) + '.SS'
            stock_df = yf.download(ticker, start_date, end_date).reset_index()
            if stock_df.empty:
                print(f"No data available for {ticker}. Skipping...")
                continue
            else:
                cnt += 1
                file_name = f'{cnt:03d}_CSI800_raw_{ticker}.csv'
                stock_df.to_csv(file_path + file_name, index=False)
            print(f"Data downloaded for {ticker} and saved to {file_name}")

        # Shenzhen Exchange Market
        elif row['Exchange'] == 'Shenzhen Stock Exchange':
            ticker = (str(row['Ticker']).zfill(6)) + '.SZ'
            stock_df = yf.download(ticker, start_date, end_date).reset_index()
            if stock_df.empty:
                print(f"No data available for {ticker}. Skipping...")
                continue
            else:
                cnt += 1
                file_name = f'{cnt:03d}_CSI800_raw_{ticker}.csv'
                stock_df.to_csv(file_path + file_name, index=False)
            print(f"Data downloaded for {ticker} and saved to {file_name}")

# Crawling Single Stock News Function
def crawling_single_stock_news(keyword, start_date, end_date, max_page):
    # parameter
    start_date_str = str(datetime(*start_date))[:10]
    end_date_str = str(datetime(*end_date))[:10]
    cd_min = start_date_str[6:7] + '/' + start_date_str[8:10] + '/' + start_date_str[:4]
    cd_max = end_date_str[6:7] + '/' + end_date_str[8:10] + '/' + end_date_str[:4]
    tbs = f'cdr:1,cd_min:{cd_min},cd_max:{cd_max},sbd:1'  # sbd: sort by date
    title_li = []
    date_li = []

    # Loop through pages
    for page in tqdm(range(max_page)):
        page_str = str(10 * page)
        params = {'q': keyword, 'hl': 'en', 'tbm': 'nws', 'tbs': tbs, 'start': page_str}
        # header = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'}
        header = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"}
        cookie = {'CONSENT':'YES'}
        response = requests.get('https://www.google.com/search?1', params=params, headers=header, cookies=cookie)
        soup = BeautifulSoup(response.text, 'lxml')

        # Extract titles
        tmp_title = []
        title_raw_li = soup.find_all('div', 'n0jPhd ynAwRc MBeuO nDgy9d')
        for title in title_raw_li:
            tmp_title.append(title.get_text())
        title_li.extend(tmp_title)

        # Extract dates
        tmp_date = []
        date_raw_li = soup.find_all('div', 'OSrXXb rbYSKb LfVVr')
        for date in date_raw_li:
            date = date.get_text()
            date_formats = ['%B %d, %Y', '%b %d, %Y']
            date_obj = None
            for fmt in date_formats:
                try:
                    date_obj = datetime.strptime(date, fmt)
                    break
                except ValueError:
                    continue
            if date_obj:
                formatted_date = date_obj.strftime('%Y-%m-%d')
            else:
                print("Unvalid Date Fromat")
            tmp_date.append(formatted_date)
        date_li.extend(tmp_date)
        time.sleep(1)

    # Create DataFrame
    df = pd.DataFrame({'keyword': keyword, 'date': date_li, 'title': title_li})
    df = df.sort_values(by='date', ascending=False).reset_index(inplace=False).drop(columns='index')
    return df


# Crawling Multiple Stock News Function
def crawling_multiple_stock_news(dataset, ticker_li,start_date,end_date,max_page):
    cnt = 100 # sp500_ticker_200
    for ticker in ticker_li: 
        df = crawling_single_stock_news(ticker,start_date,end_date,max_page)
        if df.empty:
            print(f"No data found for {ticker}. Stopping the process.")
            break
        # Save Data
        cnt += 1
        file_path = f'../data/Google_News/Google_News_{dataset}/' 
        file_name = f'{cnt:03d}_{dataset}_google_news_{ticker}.csv'
        df.to_csv(file_path+file_name,index=False)
        print(f"Data downloaded for {ticker} and saved to {file_name}")



# Crawling SP500 Stock Price
start_date = '2010-01-01'
end_date = '2024-05-01'
file_path = 'sp500_stock/sp500_stock_raw_price/'
download_stocks(sp500_ticker, start_date, end_date, file_path)

# Crawling CSI300 Stock Price
start_date = '2010-01-01'
end_date = '2024-05-01'
csi300_ticker = pd.read_excel('../data/csi300_stock_list.xlsx')
download_stocks(csi300_ticker, start_date, end_date, file_path)

# Crawling Multiple Stock News
dataset = 'SP500'
ticker_li = sp500_ticker
start_date = (2016,1,1)  
end_date = (2022,5,1)  
max_page = 35
crawling_multiple_stock_news(dataset, ticker_li,start_date,end_date,max_page)