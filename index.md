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

- Determine what are the most popular topics within STN using *latent dirichlet allocation*
- Construct a label prediction model to categorize tweets using *fasttext* and *basic neural networks*
- Test the coherency (performance) of our prediction model using *LSTM networks*
- Assign a label to each tweet then, analyze the distribution of categories within STN

In the last part of this post, I will mention certain points that can be improved to take this analyzes into an higher level.

# 1 Network Construction

I started with construction of the Sehir University's Twitter network (STN). I use snowball sampling meaning that I collected the friends of my friends. [Here](https://github.com/skagankose/sehirTweets/blob/master/collectUsers.py) is the code for that. And, a list of my friends can be found [here](https://github.com/skagankose/sehirTweets/blob/master/data/coreUsers.txt). By saying friends, I mean followers and followees.

> Note that to retrieve tweets from Twitter, you have to have a Twitter developer account.
> Then to use API, it is required to create an application.
> You may access more information from their [official website](https://dev.twitter.com).

The data, in this stage, had many users that are not (actually) belong to Sehir University. I see that the users with at least 3 followers and 3 followees, within the current network, are (actually) realted to Sehir University. So, all the users who had less than 3 followers and 3 followees, are removed. Then, the network is constructed with remaining users. The code for that is [here](https://github.com/skagankose/sehirTweets/blob/master/createGraph.py).

The resulting network composed of 1,353 nodes (users) and 26,439 directed edges (according to followers and followees). Here is the graph.

![STN Graph](skagankose.github.io/images/pageRank.png)

*Figure 1: Sehir University's Twitter network with PageRank based representation (where big nodes mean high popularity).*

Note that colors and sizes of nodes are arranged according to [PageRank](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf) centrality score.

> I want to reserve couple of sentences to mention the program that I used for drawing graphs which is [Gephi](https://gephi.org).
> It is one of the leading visualization and exploration software for all kinds of graphs and networks.
> Moreover, it is open-source and free.

# 2 Natural Language Processing Analysis

After retrieving tweets, I moved onto the analysis part.

## 2.1 Topic Determination

To assign a certain topic distribution to each user, I used their tweets. I assign a label to each tweet then, calculated frequency of categories. The first obstacle was that I didn't know which label to use meaning that I didn't know how to categorize tweets within Sehir University. To solve that, I zsed a topic modeling algorithm, more specifically, [latent dirichlet allocation](http://ai.stanford.edu/~ang/papers/nips01-lda.pdf) (LDA). I won't go into details of LDA but I can simply state that it clusters similar words together for given documents. By using LDA, I were able extract the most popular (four) topics in Sehir University.

First, I retrieved last 200 tweets of each user within STN and excluded the tweets posted before 2017 for the sake of up-to-dateness. I consider the collection of tweets that belongs to a certain user, as a single document and run the LDA model accordingly. Frequent words, within each resulting clusters, are presented in the Table 1. I decided that four is an appropriate number for distinct clusters. This means that there are four distinguishable popular topics within STN. Code for [retrieving tweets](https://github.com/skagankose/sehirTweets/blob/master/fetchAndClean.py). and [LDA](https://github.com/skagankose/sehirTweets/blob/master/customizedLDA.py) can be accessed.

> I wanted to point out that the code for fetching tweets utilizes two other code files.
> The [first file](https://github.com/skagankose/sehirTweets/blob/master/tweetDumper.py) is for retrieving recent tweets of users within STN.
> And retrieved tweets are cleaned (e.g. stop words are removed) using the [second file](https://github.com/skagankose/sehirTweets/blob/master/tweetCleaner.py).

![STN Graph](skagankose.github.io/images/frequentWords.png)

*Table 1: Frequent words belonging to clusters found using LDA with appropriate titles assigned to them.*

By examining the most frequent words belonging to clusters, I assign a title to each of them. For the rest of the study, I consider these categories as four (plausible) labels for tweets.

## 2.2 Label prediction

I used a (nearly) deep neural network model to predict the labels of tweets. As a vector embedding, I used the [Turkish pre-trained model]((https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)) created using [FastText](https://arxiv.org/abs/1607.01759) algorithm.

To train the model, I retrieved tweets from different accounts related to topics that we found. We will give more detail about these process in part 3. By using the trained model, we assign label to each tweet within STN. By using tagged tweets, we determined the distribution of four topics for each user. For build the model in this part, we used Keras which is an open source neural network library. (We have used the version backed by TensorFlow.)

## 2.3 Text Generation to Test the Prediction Model

- Test text generation model
- Test label prediction model

# 3 Network Graph Analysis

# Conclusion

# Future Work
