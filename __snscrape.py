import pandas as pd
from datetime import date,timedelta,datetime
import snscrape.modules.twitter as twitterScraper
import time
from threading import Thread

class twitter_scrape(object):

    def __init__(self,keyword,max_threads=100,total_tweets_request=10000,start_date='2022-08-01',end_date='2022-10-01'):
        self.keyword = keyword
        self.max_threads = max_threads
        self.total_tweets=total_tweets_request
        self.start_date= start_date
        self.end_date=end_date

    def _multi_thread_queries(self):
        '''
        divide the query date into short date period, 
        each divided query will use one thread in the 
        scraper function;
        '''
        try:
            dt=datetime.strptime(self.end_date,'%Y-%m-%d')-datetime.strptime(self.start_date,'%Y-%m-%d')
        except ValueError:
            print("please input valid date format:YYYY-MM-DD")
            return False
        if (dt.days<=0):
            print("End date is earlier than start date")
            return False
        Quotient=dt.days//self.max_threads
        Remainder=dt.days%self.max_threads
        tmp_date=datetime.strptime(self.start_date,'%Y-%m-%d')
        tmp=self.max_threads
        queries=[]
        while tmp!=0:
            quote=self.keyword+' since:'+tmp_date.strftime("%Y-%m-%d")+' until:'
            if(Quotient==0 and Remainder==0):
                break
            tmp_date=tmp_date+timedelta(days=(Quotient+min(Remainder,1)))
            quote=quote+tmp_date.strftime("%Y-%m-%d")
            queries.append(quote)
            if Remainder!=0:
                Remainder-=1
            tmp-=1
        self.queries=queries
        return True
    
    def _spider(self,query):
        '''
        process a query and scraper the data from twitter web,
        preprocess the content by replacing the symbol to space,
        store the tweets into a single file
        '''
        f = open(self.keyword+'.csv','a')
        scraper = twitterScraper.TwitterSearchScraper(query)
        for i, tweet in enumerate(scraper.get_items(), start = 1):
            if i >self.mean_tweets:
                break
            if tweet.retweetedTweet!=None:
                continue
            tmp=tweet.content
            tmp=tmp.replace('\n',' ')
            tmp=tmp.replace('\r',' ')
            tmp=tmp.replace('\r\n',' ')
            tmp=tmp.replace(',',' ')
            f.write(f"{tweet.date},{tweet.user},{tmp},{tweet.likeCount}\n")
        f.close()

    def start(self):
        '''
        scrape the tweets by using multiple threads
        '''
        if self._multi_thread_queries():
            print('create multi-thread queries succesffuly')
        else:
            print('fail to create multi-thread queries')
            return 
        f = open(self.keyword+'.csv','a')
        f.write('post time,user_id, tweets content,likecount,\n')
        f.close()
        self.mean_tweets=self.total_tweets//len(self.queries)
        t_list = []
        for query in self.queries:
            print(len(t_list))
            t = Thread(target=self._spider,args=(query,))
            t_list.append(t)
            t.start()

if __name__ == '__main__':
    Keyword = 'crypto'
    startdate='2022-08-01'
    enddate='2022-10-01'
    maximum_threads=100
    total_tweets=1000
    spi = twitter_scrape(Keyword,maximum_threads,total_tweets,startdate,enddate)
    spi.start()