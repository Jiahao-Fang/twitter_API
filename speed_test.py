import pandas as pd
from datetime import date,timedelta,datetime
import snscrape.modules.twitter as twitterScraper
import time
from threading import Thread
from snscrape_search import twitter_scrape




class Speedtest(object):
    def __init__(self):
        self.keyword='crypto'
        self.startdate='2022-06-01'
        self.enddate='2022-10-01'
    def Speedtest1(self):
        print('Would you like to test the time of requesting 10000 tweets in 120 days(120 threads)')
        t=input("input Y/y for yes, any key for no")
        if t=='y' or t=='Y':
            maximum_threads=1000
            total_tweets=10000
            t1=time.time()
            spi = twitter_scrape(self.keyword,maximum_threads,total_tweets,self.startdate,self.enddate)
            spi.start()
            t2=time.time()
            print('it used {:.2} seconds to scrape the data'.format(t2-t1))
        return 
    def Speedtest2(self):
        print('Would you like to compare the scraping speed ofsingle thread and multi thread')
        t=input("input Y/y for yes, any key for no")
        if t=='y' or t=='Y':
            maximum_threads=1
            total_tweets=1000
            t1=time.time()
            spi = twitter_scrape(self.keyword,maximum_threads,total_tweets,self.startdate,self.enddate)
            spi.start()
            t2=time.time()
            print('it used {:.2} seconds for a single thread to scrape 1000 tweets'.format(t2-t1))
            maximum_threads=100
            t1=time.time()
            spi = twitter_scrape(self.keyword,maximum_threads,total_tweets,self.startdate,self.enddate)
            spi.start()
            t2=time.time()
            print('it used {:.2} seconds for 100 threads to scrape 1000 tweets'.format(t2-t1))
        return 

if __name__ == '__main__':
    test=Speedtest()
    test.Speedtest1()
    test.Speedtest2()


