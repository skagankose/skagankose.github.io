# Introduction

Social media usage is significantly increased in recent years. According to [Smart Insights](http://www.smartinsights.com/social-media-marketing/social-media-strategy/new-global-social-media-research/), 482 million people joined social networks since the last year. There is an enormous amount of data created by these individuals. And, this data provides an unique opportunity to analyze societies.

Within the field of Natural Language Processing, there are highly effective techniques to extract meaningful informations from piles of data. There are terrific sources to learn more about NLP.

- [Introduction to Natural Language Processing](http://blog.algorithmia.com/introduction-natural-language-processing-nlp/) by Algorithmia
- [Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) by Stanford

I want to emphasize that deep learning has become a crucial part of NLP studies, especially after the paper [Natural Language Processing (almost) from Scratch](https://arxiv.org/abs/1103.0398). Consequently, deep learning is (almost) a pre-requisite for studying NLP. Here are some of the outstanding deep learning courses.

- [Introduction to Deep Learning](http://introtodeeplearning.com) by MIT
- [Tensorflow for Deep Learning Research](http://web.stanford.edu/class/cs20si/) by Stanford

After an extensive introduction, I now, proceed to details of my research.

My aim in this research is to analyze the Twitter network of Istanbul Sehir Univesity using various NLP methods. First, I constructed the Sehir Tweeter Network (STN) then, proceed analysis. The methods I utilized throughout the analysis are as follows (in order).

* Determine what are the most popular topics within STN using **latent dirichlet allocation**
* Construct a label prediction model to categorize tweets using **fasttext and basic neural networks**
* Test the coherency (performance) of our prediction model using **LSTM networks**
* Assign a label to each tweet then, analyze the distribution of categories within STN

In the last part of this post, I mentioned certain points that can be improved to take this analyzes into an higher level.

# Network Construction

You can use the [editor on GitHub](https://github.com/skagankose/skagankose.github.io/edit/master/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/skagankose/skagankose.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
