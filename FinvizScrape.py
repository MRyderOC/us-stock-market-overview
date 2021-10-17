import time
import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup as bs
import scrapy
from scrapy.crawler import CrawlerProcess


TODAY = str(datetime.datetime.today()).split()[0]

def finviz_last_page_number() -> int:
    # Get the url and make a BeautifulSoup object
    url = 'https://finviz.com/screener.ashx?v=111'
    source = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
    soup = bs(source, 'lxml')
    # Find the last page and total records
    return int(soup.find_all('a', class_='screener-pages')[-1].text)


def finviz_total_records() -> int:
    # Get the url and make a BeautifulSoup object
    url = 'https://finviz.com/screener.ashx?v=111'
    source = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
    soup = bs(source, 'lxml')
    # Find the last page and total records
    total_records = soup.find(
        'td',
        attrs={
            "class": "count-text",
            "width": "140",
            "valign": "bottom",
            "align": "left",
        }
    ).text[7:-2]
    return int(total_records.strip())


class FinvizSpider(scrapy.Spider):
    name = 'scrape_finviz'

    last_page = finviz_last_page_number()
    start_urls = [
        f'https://finviz.com/screener.ashx?v=111&r={(20 * (i - 1)) + 1}'
        for i in range(1, last_page+1)
    ]

    def parse(self, response):
        row_headers = [
            'No.',
            'Ticker',
            'Company',
            'Sector',
            'Industry',
            'Country',
            'Market Cap',
            'P/E',
            'Price',
            'Change',
            'Volume',
        ]
        try:
            table = response.css('table[bgcolor="#d3d3d3"]')[0]
        except IndexError:
            # Do it again
            yield scrapy.Request(url=response.url, callback=self.parse, dont_filter=True)
        except Exception as e:
            print(f'Exception occured for url: {response.url}\n\tException is: {e}')
            time.sleep(1)
            # Do it again
            yield scrapy.Request(url=response.url, callback=self.parse, dont_filter=True)
        rows = table.css('tr')[1:]
        for row in rows:
            texts = row.css('a *::text').getall()
            yield {
                header: data
                for header, data in zip(row_headers, texts)
            }


def scraping_finviz(delay: float = None):
    settings = {
        'BOT_NAME': '',
        'USER_AGENT': 'Mozilla/5.0',
        'ROBOTSTXT_OBEY': False,
        'COOKIES_ENABLED': False,
        'FEED_FORMAT': "json",
        'FEED_URI': f'./db/{TODAY}.json',
        'LOG_ENABLED': False,
    }
    if delay:
        settings.update({'DOWNLOAD_DELAY': delay})
    process = CrawlerProcess(settings=settings)
    process.crawl(FinvizSpider)
    process.start()


def write_data():
    scraping_finviz()

    df = pd.read_json(f'./db/{TODAY}.json', orient='records')
    df.sort_values(by='No.', inplace=True)
    df.set_index('No.', inplace=True)

    total_records = finviz_total_records()
    if len(df) != total_records:
        raise Exception("The program didn't scrape the whole data.")

    df.to_csv(f'./db/{TODAY}_raw.csv')



if __name__ == '__main__':
    s = time.time()
    write_data()
    print(f'{"*"*30}\nTime is: {time.time() - s}\n{"*"*30}')