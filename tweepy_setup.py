import tweepy


class userAuth:
    def __init__(self):
        self.api_key = "g89OFazB0Odw8oYvE6UWG2x3z"
        self.api_secret = "tOK9x4TojIpHXITd9eWnV76qQ5Gyd9yRJghvVBLkEIwZqI4YHT"
        self.bearer_token = "AAAAAAAAAAAAAAAAAAAAAOWNiQEAAAAAl35aWc0ki7jabW7utYRcTva6JcU%3D6beYa3kkBftyo7UZqaPwW81Hi4q0M4CCDM81s59gQEcYd4RJyL"
        self.access_token = "1583110572027117568-C9VQM5eLYr5I1h8RAFrnjXHmppmYpq"
        self.access_token_secret = "I0oDMwQQOotjiV8Bqv2rv2ir5UMzHuc1pKcVadhRBnQm3"
    def authenticate(self):
        client = tweepy.Client(self.bearer_token,self.api_key,self.api_secret,self.access_token,self.access_token_secret)
        auth= tweepy.OAuth1UserHandler(self.api_key,self.api_secret,self.access_token,self.access_token_secret)
        self.api =tweepy.API(auth)
