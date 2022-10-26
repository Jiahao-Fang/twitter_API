# Twitter scraper

This is the sample code for tweets data scraping and real time tweets streaming

## Requirements

The project requires Python 3.8 or higher

## installation

    pip install -r requirements.txt

## Usage

`snscrape_search` is for data scraping by multi-threads using snscraper pacakage, user can change the parameter including `Keyword`, `startdate`, `enddate`, `maximum_threads` and `total_tweets`. It will store the tweets data into a csv file including its contents, post user, likecounts. User can also request more information of tweets by changing the parameter.

`speed_test` is a speed test for tweets search function

`tweepy_setup` is authentication class for Twitter API, the key in it now is my own developer account with V2 elevated access.

`tweepy_stream` is sample code for streaming the real time tweets.

More function is coming soon

## 10.25 Update
After looking into the source code of snscrape, I realized snscrape package scrape the tweets by accessing the url:https://api.twitter.com/2/search/adaptive.json, which is a unsupported and unpublished API from Twitter's developer platform. It has potential risk of using this API including a block of IP address. Please use it carefully.

