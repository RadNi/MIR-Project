# from selenium import webdriver
# from selenium.webdriver.common.keys import Keys
import json
import os
import sys
from multiprocessing import Process
from queue import Queue

from scrapy import crawler
from scrapy.crawler import CrawlerRunner, CrawlerProcess
from twisted.internet import reactor
from scrapy.utils.log import configure_logging

from Code.constants import CRAWL_START_PAGES, CRAWLER_PREFIX, CRAWLER_OUTPUT_JSON, DESIRED_PAGES_NUM, CRAWLER_DUMP_FILE_ADDR
from time import sleep, time
import scrapy


class Wrapper:
    CRAWL_START_PAGES = CRAWL_START_PAGES
    COUNTER = 0


# BASE_URLS = Wrapper.CRAWL_START_PAGES
CRAWLED_PAGES = []

try:
    with open(CRAWLER_DUMP_FILE_ADDR, "r") as f:
        CRAWLED_PAGES, Wrapper.CRAWL_START_PAGES, Wrapper.COUNTER = eval(f.read())
        print(len(Wrapper.CRAWL_START_PAGES))
    # eval("DataSet/phase3/crawler_dump.py")
except Exception as e:
    print(str(e))
    print("Starting crawler from scratch")
print(Wrapper.CRAWL_START_PAGES)
with open(CRAWLER_OUTPUT_JSON) as json_file:
    JSON_OUTPUT = json.load(json_file)


class Crawler(scrapy.Spider):
    name = "semantic_scholar"
    urls = Wrapper.CRAWL_START_PAGES

    def start_requests(self):
        # urls = [
        #     'http://quotes.toscrape.com/page/1/',
        #     'http://quotes.toscrape.com/page/2/',
        # ]

        Crawler.urls = Wrapper.CRAWL_START_PAGES
        # input(Wrapper.CRAWL_START_PAGES)

        for query in Crawler.urls:
            url = CRAWLER_PREFIX + query
            # print(url)
            # input()
            if query in JSON_OUTPUT:
                continue
            if len(CRAWLED_PAGES) < DESIRED_PAGES_NUM:
                yield scrapy.Request(url=url, callback=self.parse)
            # print(self.urls)

    def _update_urls(self, new_batch):
        for url in new_batch:
            if CRAWLER_PREFIX + url not in CRAWLED_PAGES:
                Wrapper.CRAWL_START_PAGES.append(url)

    def parse(self, response):
        page_id = response.request.url
        if response.request.url.startswith('https://www.semanticscholar.org'):
            page_id = page_id[len('https://www.semanticscholar.org'):]
        page_json = {
            "id": page_id,
            "title": response.selector.xpath("//*[@id=\"paper-header\"]/h1/text()").get(),
            "abstract": response.selector.xpath("//*[@id=\"paper-header\"]/div[1]/div/text()").get(),
            "date": response.selector.xpath("//*[@id=\"paper-header\"]/ul[1]/li[2]/span[2]/span/span/text()").get(),
            "authors": response.selector.xpath("//*[@id=\"paper-header\"]/ul[2]/li/span/span/a/span/span/text()").extract(),
            "references": response.selector.xpath("//*[@id=\"references\"]/div[2]/div/div[2]/div[*]/div[1]/h2/a/@href").extract()
        }
        # print(page_json)
        print(page_id, "is crawled", Wrapper.COUNTER)
        Wrapper.COUNTER += 1
        if page_json["id"] not in Wrapper.CRAWL_START_PAGES:
            print(Wrapper.CRAWL_START_PAGES)
            print(page_json["id"])
            input()
        else:
            Wrapper.CRAWL_START_PAGES.remove(page_json["id"])

        # try:
        #     BASE_URLS.remove(page_json["id"])
        # except:
        #     print(BASE_URLS, response.request.url[len("https://www.semanticscholar.org"):])
        #     exit(0)
        self._update_urls(page_json["references"])
        CRAWLED_PAGES.append(page_json["id"])
        JSON_OUTPUT[page_json["id"]] = page_json
        # if JSON_OUTPUT[page_json["id"]]:
        #     input("Exists")


process = CrawlerProcess(settings={
    'FEED_FORMAT': 'json',
    'FEED_URI': 'items.json',
    'LOG_LEVEL': 'INFO',
    'USER_AGENT': 'Mozilla/5.0',
    'DOWNLOAD_DELAY': 1/5
})
process.crawl(Crawler)
process.start()
with open(CRAWLER_OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(JSON_OUTPUT, f, ensure_ascii=False, indent=4)

with open(CRAWLER_DUMP_FILE_ADDR, "w") as f:
    f.write(str(CRAWLED_PAGES) + ', ')
    f.write(str(Wrapper.CRAWL_START_PAGES) + ', ')
    f.write(str(Wrapper.COUNTER) + '\n')

exit(CRAWLED_PAGES)