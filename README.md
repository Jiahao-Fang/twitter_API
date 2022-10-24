# Twitter scraper

This is the sample code for tweets data scraping and real time tweets streaming

## Requirements

The project requires Python 3.8 or higher

## installation

    pip install -r requirements.txt

## Usage

`__snscrape` is for data scraping by multi-threads using snscraper pacakage, user can change the parameter including `Keyword`, `startdate`, `enddate`, `maximum_threads` and `total_tweets`.

`tweepy_setup` is authentication class for Twitter API, the key in it now is my own developer account with V2 elevated access.

`tweepy_stream` is sample code for streaming the real time tweets.

More function is coming soon

