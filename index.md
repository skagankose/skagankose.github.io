# Introduction

Social media usage is significantly increased in recent years. According to [Smart Insights](http://www.smartinsights.com/social-media-marketing/social-media-strategy/new-global-social-media-research/), 482 million people joined social networks in the last year. There is an enormous amount of data provided by these individuals and this data creates a unique opportunity to analyze societies as never before.

Within the field of Natural Language Processing, there are highly effective techniques to extract meaningful informations from piles of data. There are terrific sources to learn more about NLP.

- [Introduction to Natural Language Processing](http://blog.algorithmia.com/introduction-natural-language-processing-nlp/) by Algorithmia
- [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) by Stanford

I want to emphasize that deep learning has become a crucial part of NLP studies, especially after the paper [Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398). Consequently, deep learning is (almost) a pre-requisite for studying NLP. Here are some of the outstanding deep learning courses.

- [Introduction to Deep Learning](http://introtodeeplearning.com) by MIT
- [Tensorflow for Deep Learning Research](http://web.stanford.edu/class/cs20si/) by Stanford

After an extensive introduction, I now, proceed to details of my research.

My aim in this research is to analyze the Twitter network of Istanbul Sehir Univesity using various NLP methods. First, I constructed the Sehir Tweeter Network (STN) and then, proceed to NLP analysis. The methods I utilized through the analysis are as follows (in order).

- Determine what are the most popular topics within STN using *latent dirichlet allocation**
- Construct a label prediction model to categorize tweets using *fasttext** and *basic neural networks*
- Test the coherency (performance) of our prediction model using *LSTM networks*
- Assign a label to each tweet then, analyze the distribution of categories within STN

In the last part of this post, I will mention certain points that can be improved to take this analyzes into an higher level.

# Network Construction

I started with construction of the Sehir University's Twitter network (STN). I use snowball sampling meaning that I collected the friends of my friends. A list of my friends can be found [here](https://github.com/skagankose/sehirTweets/blob/master/data/coreUsers.txt). By saying friends, I mean followers and followees. [Here](https://github.com/skagankose/sehirTweets/blob/master/collectUsers.py) is the code I wrote.

The network, in this stage, had many users that are not (actually) belong to Sehir University. I determined that the users with at least 3 followers and 3 followees, within the current network, are realted to Sehir University. So, all the users who had less than 3 followers and 3 followings, are removed.

The resulting network composed of 1,382 nodes (users) and 27,118 directed edges (according to followers and followees). Here is the graph.

![STN Graph](skagankose.github.io/images/pageRank.png)
*Figure 1: Sehir University's Twitter network with PageRank based representation (where big nodes mean high popularity)*

Note that the color and the size of the nodes are according to [PageRank](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf) centrality score.

# Natural Language Processing Analysis

After retrieving tweets, I moved onto the analysis.

## Topic Determination

To assign a certain topic distribution to each user, first, we labeled each of their tweets. The problem here though, we didn't know which label to use. To this respect, first, we analyzed the STN using a topic modeling algorithm, more specifically, latent dirichlet allocation (LDA). We won't go into details of LDA for the sake of simplicity. To earn more about LDA see [2]. By using LDA, we were able extract four main topics popular among Sehir members.

## Label prediction

## Text Generation to Test the Prediction Model

1. Test text generation model
2. Test label prediction model

## Network Graph Analysis

# Future Work
