# Introduction

Social media usage is significantly increased in recent years. According to [Smart Insights](http://www.smartinsights.com/social-media-marketing/social-media-strategy/new-global-social-media-research/), 482 million people joined social networks in the last year. There is an enormous amount of data provided by these individuals and this data creates a unique opportunity to analyze societies as never before.

Within the field of Natural Language Processing, there are highly effective techniques to extract meaningful informations from piles of data. There are terrific sources to learn more about NLP.

- [Introduction to Natural Language Processing](http://blog.algorithmia.com/introduction-natural-language-processing-nlp/) by Algorithmia
- [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) by Stanford

I want to emphasize that deep learning has become a crucial part of NLP studies, especially after the paper [Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398). Consequently, deep learning is (almost) a pre-requisite for studying NLP. Here are some of the outstanding deep learning courses.

- [Introduction to Deep Learning](http://introtodeeplearning.com) by MIT
- [Tensorflow for Deep Learning Research](http://web.stanford.edu/class/cs20si/) by Stanford

After an extensive prolog, her is the details of my research.

My aim in this research is to analyze the Twitter network of Istanbul Sehir Univesity using various NLP methods. First, I constructed the Sehir Tweeter Network (STN) and then, proceed to the analysis. The methods I utilized through the analysis are as follows (in order).

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

The data, in this stage, had many users that are not (actually) belong to Sehir University. I determined that the users with at least 3 followers and 3 followees, within the current network, are (actually) related to Sehir University. So, all the users who had less than 3 followers and 3 followees, are removed. Then, the network is constructed with remaining users. The code for that is [here](https://github.com/skagankose/sehirTweets/blob/master/createGraph.py).

The resulting network composed of 1,353 nodes (users) and 26,439 directed edges (according to followers and followees). Here is the graph.

![STN Graph](skagankose.github.io/images/pageRank.png)

*Figure 1: Sehir University's Twitter network with PageRank based representation (where big nodes mean high popularity).*

Note that colors and sizes of nodes are arranged according to [PageRank](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf) centrality score.

> I want to reserve couple of sentences to mention the program that I used for drawing graphs which is [Gephi](https://gephi.org).
> It is one of the leading visualization and exploration software for all kinds of graphs and networks.

# 2 Natural Language Processing Analysis

After retrieving tweets and constructing the network, I moved onto the analysis part.

## 2.1 Topic Determination

To determine a certain topic distribution to each user, I used their tweets. First, a label is assigned to each tweet then, label frequency is calculated for each user. The first challenge was that I didn't know which label to use, meaning that I didn't know how to categorize tweets within Sehir University. To solve that, I used a topic modeling algorithm, more specifically, [latent dirichlet allocation](http://ai.stanford.edu/~ang/papers/nips01-lda.pdf) (LDA). I won't go into details of LDA but I can basically say that it clusters similar words together for given documents. By using LDA, I were able extract the most popular (four) topics in Sehir University.

First, I retrieved last 200 tweets of each user within STN by excluding the tweets posted before 2017 for the sake of up-to-dateness. I consider the collection of tweets that belongs to a certain user, as a single document and run the LDA model accordingly. Frequent words, within each resulting clusters, are presented in the Table 1. By empirical analysis, I determined that four is an appropriate for distinct number of clusters, meaning that there are four distinguishable popular topics within STN. Here are the codes for [retrieving tweets](https://github.com/skagankose/sehirTweets/blob/master/fetchAndClean.py). and [LDA](https://github.com/skagankose/sehirTweets/blob/master/customizedLDA.py).

> I wanted to point out that the code for fetching tweets utilizes two other code files.
> The [first file](https://github.com/skagankose/sehirTweets/blob/master/tweetDumper.py) is for retrieving recent tweets of users within STN.
> And retrieved tweets are cleaned (e.g. stop words are removed) using the [second file](https://github.com/skagankose/sehirTweets/blob/master/tweetCleaner.py).

![Word Clusters](skagankose.github.io/images/frequentWords.png)

*Table 1: Frequent words belonging to clusters found using LDA with appropriate titles assigned to them.*

By examining the most frequent words belonging to clusters, I assign a title to each of them. For the rest of the study, I consider these categories as four (plausible) labels for tweets.

## 2.2 Label prediction

I used a (nearly) deep neural network model to assign a label to each tweet. As vector embeddings, I used the [Turkish pre-trained model]((https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md)) created using [FastText](https://arxiv.org/abs/1607.01759) algorithm. I used bag-of-words method to gather the words in a sentence.

To train the prediction model, I used 3 domain-specific Twitter accounts each of which corresponds to a category in Table 1.
Last 3,000 tweets retrieved from each account (which makes 9,000 tweets for each category) then, the model is trained using these tweets by setting labels according to the belonging account. (For example, we label all the tweets retrieved from [NTV Kültür Sanat](https://twitter.com/ntvkultursanat) as "art".) Again, I wrote this [code](https://github.com/skagankose/sehirTweets/blob/master/tweetDumper.py) to retrieve tweets.

To test the accuracy of the model, I split the data into %80 and %20 as train and test respectively. As the comparison metric, I used accuracy. The performance of the system is indicated in Table 2. By the way, the code for the prediction model is [here](https://github.com/skagankose/sehirTweets/blob/master/extendedPredictor.ipynb) (just the part with the title "Prediction Model"). Since, the actual outputs are too long, I included only portions of the outputs for representation purposes.

![Prediction Model Performance](skagankose.github.io/images/predictionModel.png)

*Table 2: Performance of the prediction model using FastText as word embeddings.*

%94 accuracy is reasonably high where there are four different labels. By trusting these results, I consider the model applicative for its purpose. I used this model for the rest of the study to predict labels.

> I want to mention that the prediction model is created using [TensorFlow](https://www.tensorflow.org).
> It is an open source software library for numerical computation using data flow graphs made by Google.
> I also want to indicate that in the context of this project, all the models related to deep neural networks are created using TensorFlow.

I used trained model to label each tweet within STN. Then, I found distribution of categories that a particular user tweets about by simply,  calculating the frequency of label within tweets belonging to a particular user. For each user, I calculated four different values each of which represents the frequency of a certain category. Then, I saved them to a CSV file to use them with Gephi. The code for that part is [here](https://github.com/skagankose/sehirTweets/blob/master/extendedPredictor.ipynb) (under the title "Predicting Labels").

## 2.3 Text Generation to Test the Prediction Model

Next, I wanted to check that the labels assigned by the prediction model is coherent. I decided to try a bit of an unusual technique. First, I divided users into four groups according to topic distribution of their tweets (by considering the most frequent category among the tweets of a particular user, as the group of that user). Then, I collected all the tweets together for users belonging to the same category. The code until here can be found in [here](https://github.com/skagankose/sehirTweets/blob/master/extendedPredictor.ipynb) (at the part titled as "Creating Training Text").

Using these tweets, four different LSTM models are trained using [this code](https://github.com/skagankose/sehirTweets/blob/master/trainLSTM.ipynb). Then, by using these models, four different categories of texts are generated using [this code](https://github.com/skagankose/sehirTweets/blob/master/generateText.ipynb). I examined these (generated) texts to see whether they are coherent with the related topics. Some of the resulting (generated) texts are as follows.

![Prediction Model Performance](skagankose.github.io/images/generatedTexts.png)

*Table(s) 3: Examples of LSTM-generated tweets belonging to four different categories: Politics, Daily, Art and Charity (respectively from top to bottom and two examples for each category).*

As far as I can comprehend, each text is coherent with the related category. By examining these sentences it can be validated that the model had a fine job predicting labels.

# 3 Network Graph Analysis

After making sure that the model is working properly, I moved onto analyzing the STN graph itself. First, I analyze the graph with respect to four different categories separately. The graphs according to category distributions are presented in the Figure 2. Also frequencies of categories in STN are shown in Table 4.

![Politics and Daily Graph](skagankose.github.io/images/politicsDaily.png)

*Figure 2.1: STN graphs according to categories Politics (left) and Daily (right).*

![Art and Charity Graph](skagankose.github.io/images/artCharity.png)

*Figure 2.2: STN graphs according to categories Art (left) and Charity (right).*

![Category Frequencies](skagankose.github.io/images/categoryFrequency.png)

*Table 4: The (percentage-wise) frequency of each category within STN.*

I want to start interpreting the results by emphasizing that half of the tweets within STN are just daily conversations (which can also be called as chit chat). The frequency for the rest of the categories are close to each other.

There are certain findings that are worth mentioning about these graphs. First, users talking about the politics extensively are not in the center, (almost) all the of the bigger nodes in the Politics graph are periphery. The nodes in the center are no bigger than mediocre. Therefore, we can reason that talking extensively about politics is not the best way to gain popularity within Sehir University.

Second, large nodes in the Art graph are concentrated around the center, meaning that users talking about art & culture have more followers within Sehir University.

Moreover, in the Charity graph, it is seen that people who tweets about Charity are not concentrated in any specific region. Then, we can make the comment that there might be charitable people at all walks of Sehir University's society.

> Before finishing the interpretation of the graphs, I want to emphasize that daily conversations are evenly distributed across STN.
> Although, this results is expected, I wanted to mention it for the sake of completeness.

# Conclusion

I merged various algorithms and developed a methodology to categorize tweets and revel topic-distribution of a social network. By following this methodology, I analyzed Sehir University's Twitter Network in this study.

To wrap up, let me restate my steps and findings during the project.

- First, I tried to retrieve Twitter users and their tweets, that are members of Sehir University by using snowball sampling.
- Then, I determined the most popular four topics among retrieved tweets by using LDA.
- Considering each popular topic as a label, I created a model to predict labels of tweets. FastText is used to represent word embedding and a deep neural network model used for prediction.
- To test the coherence of this label-predictor model, I constructed an LSTM network to generate texts using tweets in each category. I found that the prediction model is working properly.
- Finally, I draw four different category-specific graphs by representing users according to the topics distribution of their tweets. Significant findings from these graphs can be gathered in one sentence as "people talking less about politics and more about art & culture tend to be more popular in Sehir University".

# Future Work

There are couple of points I can think of to improve this research. The obvious way to improve the analysis is to collect more tweets. I were able to collect only last 200 tweets of each user, however, collecting more tweets will reveal more about characteristic of users.

Another improvement might be using recursive neural network instead of bag-of-words for gathering word embeddings before feeding the neural network model. Although the overall performance of our prediction model is fine, I believe that, use of recursive neural network can increase the performance considerably.

The final point I wanted to draw attention to is that the STN graph constructed above does not indicate the real friendship relation between users. The edge are constructed using follower and followees and cannot be considered as indicating friendships. I believe that constructing a graph that shows (real) friendships relations might be reveal more information about the members of Sehir University.
