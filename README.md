

```python
#dependencies
import tweepy
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
import numpy as np
from config import consumer_key,consumer_secret,access_token,access_token_secret

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()





```


```python
# Twitter API Keys
consumer_key = consumer_key
consumer_secret = consumer_secret
access_token = access_token
access_token_secret = access_token_secret

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
#api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
#News Stations
news_stations = ('@BBCWorld','@CBSNews','@CNN','@FoxNews','@nytimes')



#Variable for holding sentiments
sentiments = []

# Loop through each user
for station in news_stations:
    count = 1
    # Loop through 5 pages of tweets (total 100 tweets)
#    for page in range(5):
    for page in tweepy.Cursor(api.user_timeline,id = station).pages(5):

        # Get all tweets from home feed
#        public_tweets = api.user_timeline(station,page = page)
        public_tweets = page        
        # Loop through all tweets
        for tweet in public_tweets:
            tweet = tweet._json
            #print(tweet._json)
            #get date and tweet text
            date = tweet['created_at']
            tweet_text = tweet['text']

            # Run Vader Analysis on each tweet
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]

            # create DataFrame
            sentiments.append({'Date':date,
                               'Station': station,
                               'Tweets':tweet_text,
                               'Compound':compound,
                               'Positive':pos,
                               'Neutral':neu,
                               'Negative':neg,
                               'Tweet_Counter':count})
            count = count + 1
            
       
    

```


```python
sentiments_pd = pd.DataFrame.from_dict(sentiments)
sentiments_pd.to_csv('stations.CSV')
sentiments_pd

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Station</th>
      <th>Tweet_Counter</th>
      <th>Tweets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>Sat Mar 24 06:53:52 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>1</td>
      <td>Flying firsts: Aviation milestones throughout ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0000</td>
      <td>Sat Mar 24 05:18:00 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>2</td>
      <td>French police 'hero' dies of wounds https://t....</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.4019</td>
      <td>Sat Mar 24 05:02:57 +0000 2018</td>
      <td>0.278</td>
      <td>0.722</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>3</td>
      <td>Spain Catalonia: Clashes after separatist lead...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.3818</td>
      <td>Sat Mar 24 03:37:39 +0000 2018</td>
      <td>0.342</td>
      <td>0.658</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>4</td>
      <td>Skripal 'regretted being double agent' https:/...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>Sat Mar 24 02:17:19 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>5</td>
      <td>The woman who brought skull watches back to li...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0000</td>
      <td>Sat Mar 24 02:01:47 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>6</td>
      <td>Florida school shooting: Pennsylvania students...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0000</td>
      <td>Sat Mar 24 01:22:13 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>7</td>
      <td>DR Congo to shun its own donor conference in G...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0000</td>
      <td>Sat Mar 24 00:42:53 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>8</td>
      <td>A day in the life of India's 'tuberculosis war...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0000</td>
      <td>Sat Mar 24 00:39:01 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>9</td>
      <td>The evolution of UK-Australia travel into a si...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.5994</td>
      <td>Fri Mar 23 23:33:03 +0000 2018</td>
      <td>0.302</td>
      <td>0.698</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>10</td>
      <td>Hamelin Bay: Nearly 150 beached whales die in ...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0000</td>
      <td>Fri Mar 23 22:18:47 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>11</td>
      <td>Eye-witness says French hostage-taker ran afte...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.8271</td>
      <td>Fri Mar 23 21:59:00 +0000 2018</td>
      <td>0.251</td>
      <td>0.749</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>12</td>
      <td>RT @BBCNews: A look at the war of words betwee...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.5994</td>
      <td>Fri Mar 23 21:48:32 +0000 2018</td>
      <td>0.234</td>
      <td>0.766</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>13</td>
      <td>RT @bbcworldservice: Yemen's desperate mothers...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-0.3612</td>
      <td>Fri Mar 23 21:14:42 +0000 2018</td>
      <td>0.238</td>
      <td>0.762</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>14</td>
      <td>France shooting: Hostage swap officer 'fightin...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-0.7783</td>
      <td>Fri Mar 23 20:40:11 +0000 2018</td>
      <td>0.576</td>
      <td>0.424</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>15</td>
      <td>Catalonia crisis: Five separatist leaders deta...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.8316</td>
      <td>Fri Mar 23 20:26:03 +0000 2018</td>
      <td>0.494</td>
      <td>0.506</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>16</td>
      <td>France hostage crisis: Moment police closed in...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.4939</td>
      <td>Fri Mar 23 19:16:28 +0000 2018</td>
      <td>0.262</td>
      <td>0.738</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>17</td>
      <td>Mother jailed in France for drowning five newb...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0000</td>
      <td>Fri Mar 23 18:44:32 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>18</td>
      <td>RT @BBCtrending: Guns, beer and cigarettes: he...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0000</td>
      <td>Fri Mar 23 18:23:40 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>19</td>
      <td>Half of African species 'face extinction' http...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0000</td>
      <td>Fri Mar 23 18:23:40 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>20</td>
      <td>Elon Musk pulls brands from Facebook https://t...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-0.5719</td>
      <td>Fri Mar 23 16:37:49 +0000 2018</td>
      <td>0.316</td>
      <td>0.684</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>21</td>
      <td>US sanctions Iranian hackers for 'stealing uni...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-0.4939</td>
      <td>Fri Mar 23 16:14:43 +0000 2018</td>
      <td>0.286</td>
      <td>0.714</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>22</td>
      <td>Car bomb targets spectators at Afghanistan wre...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.0000</td>
      <td>Fri Mar 23 15:37:32 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>23</td>
      <td>China cracks down on video parodies https://t....</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-0.7579</td>
      <td>Fri Mar 23 13:58:28 +0000 2018</td>
      <td>0.520</td>
      <td>0.480</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>24</td>
      <td>France hostage crisis: Police shoot supermarke...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>-0.3400</td>
      <td>Fri Mar 23 13:56:57 +0000 2018</td>
      <td>0.112</td>
      <td>0.888</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>25</td>
      <td>Donald Trump firing a senior member of his tea...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>-0.3182</td>
      <td>Fri Mar 23 13:43:24 +0000 2018</td>
      <td>0.320</td>
      <td>0.680</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>26</td>
      <td>Poland abortion: Protests against bill imposin...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-0.4588</td>
      <td>Fri Mar 23 13:43:24 +0000 2018</td>
      <td>0.273</td>
      <td>0.727</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>27</td>
      <td>Reddit: Guns, beer and tobacco transactions no...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-0.8176</td>
      <td>Fri Mar 23 13:09:12 +0000 2018</td>
      <td>0.273</td>
      <td>0.727</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>28</td>
      <td>RT @BBCBreaking: At least two people now feare...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-0.5574</td>
      <td>Fri Mar 23 12:27:17 +0000 2018</td>
      <td>0.340</td>
      <td>0.660</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>29</td>
      <td>Trade wars, Trump tariffs and protectionism ex...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.0000</td>
      <td>Fri Mar 23 12:14:43 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>30</td>
      <td>#Trèbes: The latest reports from supermarket i...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>0.0000</td>
      <td>Fri Mar 23 16:22:14 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>71</td>
      <td>We met Elsa and Anna as they took a break for ...</td>
    </tr>
    <tr>
      <th>471</th>
      <td>0.4588</td>
      <td>Fri Mar 23 16:20:15 +0000 2018</td>
      <td>0.000</td>
      <td>0.850</td>
      <td>0.150</td>
      <td>@nytimes</td>
      <td>72</td>
      <td>Welcome to the New 42nd Street Studios, where ...</td>
    </tr>
    <tr>
      <th>472</th>
      <td>-0.6705</td>
      <td>Fri Mar 23 16:10:13 +0000 2018</td>
      <td>0.176</td>
      <td>0.824</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>73</td>
      <td>An armed man who killed 3 people in a supermar...</td>
    </tr>
    <tr>
      <th>473</th>
      <td>0.0000</td>
      <td>Fri Mar 23 15:55:06 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>74</td>
      <td>Make Mark Bittman's spaghetti pie https://t.co...</td>
    </tr>
    <tr>
      <th>474</th>
      <td>0.3400</td>
      <td>Fri Mar 23 15:40:06 +0000 2018</td>
      <td>0.000</td>
      <td>0.876</td>
      <td>0.124</td>
      <td>@nytimes</td>
      <td>75</td>
      <td>A PAC led by John Bolton, the next national se...</td>
    </tr>
    <tr>
      <th>475</th>
      <td>0.3182</td>
      <td>Fri Mar 23 15:25:34 +0000 2018</td>
      <td>0.000</td>
      <td>0.859</td>
      <td>0.141</td>
      <td>@nytimes</td>
      <td>76</td>
      <td>40% of Americans were obese in 2015 and 2016, ...</td>
    </tr>
    <tr>
      <th>476</th>
      <td>-0.2960</td>
      <td>Fri Mar 23 15:11:43 +0000 2018</td>
      <td>0.196</td>
      <td>0.804</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>77</td>
      <td>RT @karenyourish: All the major Trump firings ...</td>
    </tr>
    <tr>
      <th>477</th>
      <td>-0.5106</td>
      <td>Fri Mar 23 15:00:08 +0000 2018</td>
      <td>0.185</td>
      <td>0.815</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>78</td>
      <td>Over 800 protests for stricter gun laws are pl...</td>
    </tr>
    <tr>
      <th>478</th>
      <td>0.3832</td>
      <td>Fri Mar 23 14:45:02 +0000 2018</td>
      <td>0.088</td>
      <td>0.701</td>
      <td>0.210</td>
      <td>@nytimes</td>
      <td>79</td>
      <td>John Bolton's ascension to national security a...</td>
    </tr>
    <tr>
      <th>479</th>
      <td>0.0000</td>
      <td>Fri Mar 23 14:31:04 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>80</td>
      <td>Blue passports were meant to be a symbol of po...</td>
    </tr>
    <tr>
      <th>480</th>
      <td>0.3612</td>
      <td>Fri Mar 23 14:15:09 +0000 2018</td>
      <td>0.000</td>
      <td>0.737</td>
      <td>0.263</td>
      <td>@nytimes</td>
      <td>81</td>
      <td>10 new books we recommend this week https://t....</td>
    </tr>
    <tr>
      <th>481</th>
      <td>0.0000</td>
      <td>Fri Mar 23 13:59:03 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>82</td>
      <td>John Bolton's political committee was one of t...</td>
    </tr>
    <tr>
      <th>482</th>
      <td>0.7096</td>
      <td>Fri Mar 23 13:45:04 +0000 2018</td>
      <td>0.000</td>
      <td>0.742</td>
      <td>0.258</td>
      <td>@nytimes</td>
      <td>83</td>
      <td>RT @nytimesworld: Intellectual property, which...</td>
    </tr>
    <tr>
      <th>483</th>
      <td>-0.5267</td>
      <td>Fri Mar 23 13:30:15 +0000 2018</td>
      <td>0.167</td>
      <td>0.833</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>84</td>
      <td>Breaking News: President Trump said he may vet...</td>
    </tr>
    <tr>
      <th>484</th>
      <td>0.0000</td>
      <td>Fri Mar 23 13:29:04 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>85</td>
      <td>"This is just what America needs: The thrilla ...</td>
    </tr>
    <tr>
      <th>485</th>
      <td>-0.3400</td>
      <td>Fri Mar 23 13:08:42 +0000 2018</td>
      <td>0.107</td>
      <td>0.893</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>86</td>
      <td>Breaking News: An armed man opened fire and to...</td>
    </tr>
    <tr>
      <th>486</th>
      <td>0.0000</td>
      <td>Fri Mar 23 13:00:31 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>87</td>
      <td>Morning Briefing: Here's what you need to know...</td>
    </tr>
    <tr>
      <th>487</th>
      <td>0.0000</td>
      <td>Fri Mar 23 12:51:19 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>88</td>
      <td>"Hamilton," "Frozen" and many Broadway shows a...</td>
    </tr>
    <tr>
      <th>488</th>
      <td>-0.8360</td>
      <td>Fri Mar 23 12:41:03 +0000 2018</td>
      <td>0.357</td>
      <td>0.643</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>89</td>
      <td>A firefighter was killed and two others were s...</td>
    </tr>
    <tr>
      <th>489</th>
      <td>-0.6908</td>
      <td>Fri Mar 23 12:35:00 +0000 2018</td>
      <td>0.198</td>
      <td>0.802</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>90</td>
      <td>"All of the information we have at the moment ...</td>
    </tr>
    <tr>
      <th>490</th>
      <td>-0.5994</td>
      <td>Fri Mar 23 12:30:02 +0000 2018</td>
      <td>0.329</td>
      <td>0.549</td>
      <td>0.122</td>
      <td>@nytimes</td>
      <td>91</td>
      <td>European leaders issued a relatively tough sta...</td>
    </tr>
    <tr>
      <th>491</th>
      <td>0.0000</td>
      <td>Fri Mar 23 12:20:05 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>92</td>
      <td>In an NYT interview in 2002, John Bolton was a...</td>
    </tr>
    <tr>
      <th>492</th>
      <td>-0.3612</td>
      <td>Fri Mar 23 12:11:01 +0000 2018</td>
      <td>0.160</td>
      <td>0.751</td>
      <td>0.089</td>
      <td>@nytimes</td>
      <td>93</td>
      <td>Your daily @DealBook Briefing:\n\n• China is t...</td>
    </tr>
    <tr>
      <th>493</th>
      <td>-0.5574</td>
      <td>Fri Mar 23 11:58:45 +0000 2018</td>
      <td>0.159</td>
      <td>0.841</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>94</td>
      <td>A police operation is underway in Trèbes, in t...</td>
    </tr>
    <tr>
      <th>494</th>
      <td>0.0000</td>
      <td>Fri Mar 23 11:46:04 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>95</td>
      <td>The Austin bombings have stoked the raw racial...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>-0.7579</td>
      <td>Fri Mar 23 11:30:09 +0000 2018</td>
      <td>0.265</td>
      <td>0.735</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>96</td>
      <td>More than 150 whales became stranded on a West...</td>
    </tr>
    <tr>
      <th>496</th>
      <td>-0.2732</td>
      <td>Fri Mar 23 11:20:02 +0000 2018</td>
      <td>0.180</td>
      <td>0.691</td>
      <td>0.129</td>
      <td>@nytimes</td>
      <td>97</td>
      <td>A brewing trade war between China and the Unit...</td>
    </tr>
    <tr>
      <th>497</th>
      <td>0.6369</td>
      <td>Fri Mar 23 11:10:06 +0000 2018</td>
      <td>0.000</td>
      <td>0.826</td>
      <td>0.174</td>
      <td>@nytimes</td>
      <td>98</td>
      <td>For decades, Americans have believed that the ...</td>
    </tr>
    <tr>
      <th>498</th>
      <td>-0.5574</td>
      <td>Fri Mar 23 11:00:08 +0000 2018</td>
      <td>0.167</td>
      <td>0.833</td>
      <td>0.000</td>
      <td>@nytimes</td>
      <td>99</td>
      <td>A firefighter died after responding to a blaze...</td>
    </tr>
    <tr>
      <th>499</th>
      <td>-0.0772</td>
      <td>Fri Mar 23 10:50:04 +0000 2018</td>
      <td>0.109</td>
      <td>0.796</td>
      <td>0.095</td>
      <td>@nytimes</td>
      <td>100</td>
      <td>China announced that it would impose tariffs o...</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 8 columns</p>
</div>




```python

#Dataframe filter for each station
BBC_pd = sentiments_pd.loc[sentiments_pd['Station'] == '@BBCWorld']
CBS_pd = sentiments_pd.loc[sentiments_pd['Station'] == '@CBSNews']
CNN_pd = sentiments_pd.loc[sentiments_pd['Station'] == '@CNN']
FOX_pd = sentiments_pd.loc[sentiments_pd['Station'] == '@FoxNews']
NYT_pd = sentiments_pd.loc[sentiments_pd['Station'] == '@nytimes']
BBC_pd
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Station</th>
      <th>Tweet_Counter</th>
      <th>Tweets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>Sat Mar 24 06:53:52 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>1</td>
      <td>Flying firsts: Aviation milestones throughout ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0000</td>
      <td>Sat Mar 24 05:18:00 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>2</td>
      <td>French police 'hero' dies of wounds https://t....</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.4019</td>
      <td>Sat Mar 24 05:02:57 +0000 2018</td>
      <td>0.278</td>
      <td>0.722</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>3</td>
      <td>Spain Catalonia: Clashes after separatist lead...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.3818</td>
      <td>Sat Mar 24 03:37:39 +0000 2018</td>
      <td>0.342</td>
      <td>0.658</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>4</td>
      <td>Skripal 'regretted being double agent' https:/...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0000</td>
      <td>Sat Mar 24 02:17:19 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>5</td>
      <td>The woman who brought skull watches back to li...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0000</td>
      <td>Sat Mar 24 02:01:47 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>6</td>
      <td>Florida school shooting: Pennsylvania students...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0000</td>
      <td>Sat Mar 24 01:22:13 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>7</td>
      <td>DR Congo to shun its own donor conference in G...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0000</td>
      <td>Sat Mar 24 00:42:53 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>8</td>
      <td>A day in the life of India's 'tuberculosis war...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0000</td>
      <td>Sat Mar 24 00:39:01 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>9</td>
      <td>The evolution of UK-Australia travel into a si...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.5994</td>
      <td>Fri Mar 23 23:33:03 +0000 2018</td>
      <td>0.302</td>
      <td>0.698</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>10</td>
      <td>Hamelin Bay: Nearly 150 beached whales die in ...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0000</td>
      <td>Fri Mar 23 22:18:47 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>11</td>
      <td>Eye-witness says French hostage-taker ran afte...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>-0.8271</td>
      <td>Fri Mar 23 21:59:00 +0000 2018</td>
      <td>0.251</td>
      <td>0.749</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>12</td>
      <td>RT @BBCNews: A look at the war of words betwee...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>-0.5994</td>
      <td>Fri Mar 23 21:48:32 +0000 2018</td>
      <td>0.234</td>
      <td>0.766</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>13</td>
      <td>RT @bbcworldservice: Yemen's desperate mothers...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>-0.3612</td>
      <td>Fri Mar 23 21:14:42 +0000 2018</td>
      <td>0.238</td>
      <td>0.762</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>14</td>
      <td>France shooting: Hostage swap officer 'fightin...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>-0.7783</td>
      <td>Fri Mar 23 20:40:11 +0000 2018</td>
      <td>0.576</td>
      <td>0.424</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>15</td>
      <td>Catalonia crisis: Five separatist leaders deta...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>-0.8316</td>
      <td>Fri Mar 23 20:26:03 +0000 2018</td>
      <td>0.494</td>
      <td>0.506</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>16</td>
      <td>France hostage crisis: Moment police closed in...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>-0.4939</td>
      <td>Fri Mar 23 19:16:28 +0000 2018</td>
      <td>0.262</td>
      <td>0.738</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>17</td>
      <td>Mother jailed in France for drowning five newb...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0000</td>
      <td>Fri Mar 23 18:44:32 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>18</td>
      <td>RT @BBCtrending: Guns, beer and cigarettes: he...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0000</td>
      <td>Fri Mar 23 18:23:40 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>19</td>
      <td>Half of African species 'face extinction' http...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0000</td>
      <td>Fri Mar 23 18:23:40 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>20</td>
      <td>Elon Musk pulls brands from Facebook https://t...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>-0.5719</td>
      <td>Fri Mar 23 16:37:49 +0000 2018</td>
      <td>0.316</td>
      <td>0.684</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>21</td>
      <td>US sanctions Iranian hackers for 'stealing uni...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>-0.4939</td>
      <td>Fri Mar 23 16:14:43 +0000 2018</td>
      <td>0.286</td>
      <td>0.714</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>22</td>
      <td>Car bomb targets spectators at Afghanistan wre...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.0000</td>
      <td>Fri Mar 23 15:37:32 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>23</td>
      <td>China cracks down on video parodies https://t....</td>
    </tr>
    <tr>
      <th>23</th>
      <td>-0.7579</td>
      <td>Fri Mar 23 13:58:28 +0000 2018</td>
      <td>0.520</td>
      <td>0.480</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>24</td>
      <td>France hostage crisis: Police shoot supermarke...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>-0.3400</td>
      <td>Fri Mar 23 13:56:57 +0000 2018</td>
      <td>0.112</td>
      <td>0.888</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>25</td>
      <td>Donald Trump firing a senior member of his tea...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>-0.3182</td>
      <td>Fri Mar 23 13:43:24 +0000 2018</td>
      <td>0.320</td>
      <td>0.680</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>26</td>
      <td>Poland abortion: Protests against bill imposin...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>-0.4588</td>
      <td>Fri Mar 23 13:43:24 +0000 2018</td>
      <td>0.273</td>
      <td>0.727</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>27</td>
      <td>Reddit: Guns, beer and tobacco transactions no...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>-0.8176</td>
      <td>Fri Mar 23 13:09:12 +0000 2018</td>
      <td>0.273</td>
      <td>0.727</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>28</td>
      <td>RT @BBCBreaking: At least two people now feare...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>-0.5574</td>
      <td>Fri Mar 23 12:27:17 +0000 2018</td>
      <td>0.340</td>
      <td>0.660</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>29</td>
      <td>Trade wars, Trump tariffs and protectionism ex...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.0000</td>
      <td>Fri Mar 23 12:14:43 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>30</td>
      <td>#Trèbes: The latest reports from supermarket i...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>-0.1280</td>
      <td>Thu Mar 22 15:50:31 +0000 2018</td>
      <td>0.207</td>
      <td>0.631</td>
      <td>0.162</td>
      <td>@BBCWorld</td>
      <td>71</td>
      <td>Donald Trump's top Russia lawyer John Dowd res...</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.0000</td>
      <td>Thu Mar 22 15:40:24 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>72</td>
      <td>Skating Lake Baikal, the world's deepest lake ...</td>
    </tr>
    <tr>
      <th>72</th>
      <td>0.0000</td>
      <td>Thu Mar 22 14:55:08 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>73</td>
      <td>Cambridge Analytica taken to court over data s...</td>
    </tr>
    <tr>
      <th>73</th>
      <td>-0.6705</td>
      <td>Thu Mar 22 14:54:46 +0000 2018</td>
      <td>0.290</td>
      <td>0.539</td>
      <td>0.172</td>
      <td>@BBCWorld</td>
      <td>74</td>
      <td>Serving six years for killing her abusive ex-p...</td>
    </tr>
    <tr>
      <th>74</th>
      <td>0.1779</td>
      <td>Thu Mar 22 14:33:39 +0000 2018</td>
      <td>0.215</td>
      <td>0.519</td>
      <td>0.267</td>
      <td>@BBCWorld</td>
      <td>75</td>
      <td>Ukraine arrests pilot hero Savchenko over 'cou...</td>
    </tr>
    <tr>
      <th>75</th>
      <td>-0.5106</td>
      <td>Thu Mar 22 14:27:54 +0000 2018</td>
      <td>0.248</td>
      <td>0.752</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>76</td>
      <td>Dapchi kidnap: Nigerian father's pain as daugh...</td>
    </tr>
    <tr>
      <th>76</th>
      <td>-0.7184</td>
      <td>Thu Mar 22 14:22:27 +0000 2018</td>
      <td>0.500</td>
      <td>0.500</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>77</td>
      <td>YouTube gun ban drives bloggers to PornHub htt...</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.0000</td>
      <td>Thu Mar 22 14:17:41 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>78</td>
      <td>Dutch parliament halted as man jumps from publ...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>0.0000</td>
      <td>Thu Mar 22 14:17:41 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>79</td>
      <td>Cynthia Nixon and 10 other celebrities who ent...</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.0000</td>
      <td>Thu Mar 22 13:49:56 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>80</td>
      <td>Nigerian girls tell of kidnap ordeal https://t...</td>
    </tr>
    <tr>
      <th>80</th>
      <td>-0.6249</td>
      <td>Thu Mar 22 13:37:39 +0000 2018</td>
      <td>0.254</td>
      <td>0.746</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>81</td>
      <td>Meet Louis - the gorilla who walks upright to ...</td>
    </tr>
    <tr>
      <th>81</th>
      <td>-0.1531</td>
      <td>Thu Mar 22 13:37:10 +0000 2018</td>
      <td>0.167</td>
      <td>0.833</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>82</td>
      <td>Miss Venezuela to close temporarily over corru...</td>
    </tr>
    <tr>
      <th>82</th>
      <td>0.0000</td>
      <td>Thu Mar 22 13:24:26 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>83</td>
      <td>Austria recalls diplomat from Israel over Nazi...</td>
    </tr>
    <tr>
      <th>83</th>
      <td>0.4854</td>
      <td>Thu Mar 22 13:21:44 +0000 2018</td>
      <td>0.000</td>
      <td>0.813</td>
      <td>0.187</td>
      <td>@BBCWorld</td>
      <td>84</td>
      <td>RT @BBC_HaveYourSay: Iran Supreme Leader Ali K...</td>
    </tr>
    <tr>
      <th>84</th>
      <td>-0.3182</td>
      <td>Thu Mar 22 13:16:07 +0000 2018</td>
      <td>0.247</td>
      <td>0.753</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>85</td>
      <td>Iranian officials mocked for using foreign pro...</td>
    </tr>
    <tr>
      <th>85</th>
      <td>0.2263</td>
      <td>Thu Mar 22 12:44:28 +0000 2018</td>
      <td>0.000</td>
      <td>0.808</td>
      <td>0.192</td>
      <td>@BBCWorld</td>
      <td>86</td>
      <td>Tennessee grants $1 million to wrongly convict...</td>
    </tr>
    <tr>
      <th>86</th>
      <td>-0.1779</td>
      <td>Thu Mar 22 12:28:36 +0000 2018</td>
      <td>0.141</td>
      <td>0.743</td>
      <td>0.115</td>
      <td>@BBCWorld</td>
      <td>87</td>
      <td>RT @BBCSteveR: Commenting on the Skripal poiso...</td>
    </tr>
    <tr>
      <th>87</th>
      <td>-0.2960</td>
      <td>Thu Mar 22 11:44:32 +0000 2018</td>
      <td>0.167</td>
      <td>0.833</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>88</td>
      <td>South Korea to shut off computers to stop peop...</td>
    </tr>
    <tr>
      <th>88</th>
      <td>0.3818</td>
      <td>Thu Mar 22 11:20:53 +0000 2018</td>
      <td>0.000</td>
      <td>0.776</td>
      <td>0.224</td>
      <td>@BBCWorld</td>
      <td>89</td>
      <td>Florida school shooting students told to wear ...</td>
    </tr>
    <tr>
      <th>89</th>
      <td>-0.6908</td>
      <td>Thu Mar 22 10:59:32 +0000 2018</td>
      <td>0.416</td>
      <td>0.584</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>90</td>
      <td>Syria war: Rebels 'begin leaving key Eastern G...</td>
    </tr>
    <tr>
      <th>90</th>
      <td>0.0000</td>
      <td>Thu Mar 22 10:37:31 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>91</td>
      <td>Deadly blast at Czech chemical plant https://t...</td>
    </tr>
    <tr>
      <th>91</th>
      <td>-0.4549</td>
      <td>Thu Mar 22 10:29:12 +0000 2018</td>
      <td>0.271</td>
      <td>0.729</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>92</td>
      <td>Uber self-driving crash: Footage shows moment ...</td>
    </tr>
    <tr>
      <th>92</th>
      <td>-0.8360</td>
      <td>Thu Mar 22 09:54:21 +0000 2018</td>
      <td>0.530</td>
      <td>0.470</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>93</td>
      <td>Freiburg murder: Afghan jailed for life in Ger...</td>
    </tr>
    <tr>
      <th>93</th>
      <td>0.0000</td>
      <td>Thu Mar 22 09:50:00 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>94</td>
      <td>Curiosity rover: 2,000 days on Mars https://t....</td>
    </tr>
    <tr>
      <th>94</th>
      <td>0.0000</td>
      <td>Thu Mar 22 09:30:17 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>95</td>
      <td>Dutch referendum: Spy tapping powers vote too ...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>-0.8360</td>
      <td>Thu Mar 22 09:25:17 +0000 2018</td>
      <td>0.530</td>
      <td>0.470</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>96</td>
      <td>Eleven jailed for life over India 'beef' murde...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0.0000</td>
      <td>Thu Mar 22 09:25:15 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>97</td>
      <td>UK passports 'to be made in France after Brexi...</td>
    </tr>
    <tr>
      <th>97</th>
      <td>-0.6249</td>
      <td>Thu Mar 22 09:25:15 +0000 2018</td>
      <td>0.362</td>
      <td>0.638</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>98</td>
      <td>Rebel Wilson: Australia media firms fail in de...</td>
    </tr>
    <tr>
      <th>98</th>
      <td>-0.5859</td>
      <td>Thu Mar 22 09:25:15 +0000 2018</td>
      <td>0.324</td>
      <td>0.676</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>99</td>
      <td>Theresa May to warn EU leaders of Russian thre...</td>
    </tr>
    <tr>
      <th>99</th>
      <td>0.0000</td>
      <td>Thu Mar 22 08:14:29 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>@BBCWorld</td>
      <td>100</td>
      <td>Nicolas Sarkozy: French ex-president says fund...</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 8 columns</p>
</div>




```python
#Scatterplot
#Scatter different colors

a = plt.scatter(BBC_pd['Tweet_Counter'],BBC_pd['Compound'], c = 'red',label = 'BBC')
b = plt.scatter(CBS_pd['Tweet_Counter'],CBS_pd['Compound'], c = 'blue', label = 'CBS')
c = plt.scatter(CNN_pd['Tweet_Counter'],CNN_pd['Compound'], c = 'green', label = 'CNN')
d = plt.scatter(FOX_pd['Tweet_Counter'],FOX_pd['Compound'], c = 'yellow', label = 'FOX')
e = plt.scatter(NYT_pd['Tweet_Counter'],NYT_pd['Compound'], c = 'orange', label = 'NYT')

legend =[a,b,c,d,e]
legend_names = ['BBC','CBS','CNN','FOX','NYT']


#Other scatterplot items
plt.grid(True)
plt.legend(legend,legend_names)
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
plt.title("Sentiment Analysis of Media Tweets 3/23")
plt.savefig('TwitterScatterPlot.png')
plt.show()

```


![png](output_5_0.png)



```python
#Barplot
#Overall sentiment - mean compound value for each station
BBC = BBC_pd['Compound'].mean()
CBS = CBS_pd['Compound'].mean()
CNN = CNN_pd['Compound'].mean()
FOX = FOX_pd['Compound'].mean()
NYT = NYT_pd['Compound'].mean()
colors = ['red','blue','green','yellow','orange']

station_value = [BBC,CBS,CNN,FOX,NYT]
#position
station_len = np.arange(5)
station_names = ['BBC','CBS','CNN','FOX','NYT']
width = 0.5

plt.bar(station_len, station_value, max(station_value) + 0.5, color = colors)

plt.xticks(station_len,station_names)
plt.title("Overall Media Sentiment Based on Twitter 3/23")
plt.xlabel("Stations")
plt.ylabel("Tweet Polarity")
plt.savefig('TwitterBarPlot.png')
plt.show()

```


![png](output_6_0.png)



```python
print(BBC)
print(CBS)
print(CNN)
print(FOX)
print(NYT)
```

    -0.16529000000000005
    -0.16159800000000005
    -0.045325000000000004
    -0.011947000000000001
    -0.09036399999999997
    
