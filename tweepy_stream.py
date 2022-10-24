import time
import tweepy
from tweepy_setup import userAuth

search_terms =["python","programming","coding"]
class MyStream(tweepy.StreamingClient):
    '''
    sample stream code,
    print the tweets in the terminal each 0.1 second
    '''
    def on_connect(self):
        print("connected")
    def on_tweet(self, tweet):
        if tweet.referenced_tweets == None:
            print(tweet.text)
            time.sleep(0.1)

if __name__ == '__main__':
    usr=userAuth()
    stream =MyStream(bearer_token=usr.bearer_token)
    search_terms =["crypto"]
    for term in search_terms:
        stream.add_rules(tweepy.StreamRule(term))
    #stream.add_rules(tweepy.StreamRule('from:sam__loker'))
    
    stream.filter(tweet_fields="referenced_tweets")
    #stream.disconnect()

    