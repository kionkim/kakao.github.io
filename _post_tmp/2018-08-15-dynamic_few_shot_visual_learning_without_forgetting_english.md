---
layout: post
title: 'Dynamic Few-Shot Visual Learning without Forgetting'
author: kion.kim
date: 2018-07-26 17:00
tags: [deeplearning, nlp, attention, skip connection, NMT]
---

# Introduction

Recently, we are interested in meta learning to cope with the cases where we don't have enough observations for classification. This happens pretty much commonly in business situation as there are always brand new products released in the market. Whenever a new product is introduced in the market, managements eager to know who migiht be interested in new product and want to attack them more efficiently with target marketing. In modeling terminology, we face extremely imbalanced classification problem everyday.


Few shot learning is the one that we would like to do exactly. After showing a couple of examples, we want the algorithm quickly recognize new category. Few show learning was popular in a couple of years ago but there were not many researchers who adress that issue for a couple of years. After MAML and SNAIL from BAIR group, it's getting more popularity and few shot learning is one of the hottest topics in deep learning area. 

Today we will going to go over brand new paper on few shot learning, "Dynamic Few-Shot Visual Learning without Forgetting", which is presented in CVPR, 2018

# Few shot learning

Few shot learning is a trial to mimic human behavior in recognizing brand new concept after exposing to just a few examples, like children easily figure out bicycle at their first sight. This paper proposed an architecture to tackle this problem, focusing on 

(a) speed 
(b) keeping accucary for existing categories, 

which were not emphasized enough from the previous algorithms.

They proposed an algorithm called, 'dynamic few-shot learning without forgetting', trying to use information from base categories(the categories used for training step) for updating weights for novel categories(the categories only shown in test time) without losing prediction power for base castegories. They believe their algorithm mimic human's learning process more naturally compared to the previously proposed methods. Two major techniques involved here are

* Few-shot classification-weight generator based on attention
* Cosine-similarity based ConvNet recognition model

## Few-shot classification-weight generator based on attention

As classifiers are defined via weights, weights for novel categories should be decided to be able to learn how to classify novel categories. To train weights for those rare categories, they introduced few-shot classification weight generator, which is another neurla network. While estimating weights for novel categories, they utilize information acquired from base categories via attention mechanism as if we learn knowledge from previous experience. They say that it boosted the performance when there is only one example avaialble for novel category

## Cosine-similarity based ConvNet recognition model

When merging two sets of weights, one for base categories and the other for novel categories, must be carefully incorporated not to ruin performance for base categories. As is done for ordinary fully connected network, simple dot product between weights and features won't work correctly as weight scale can be different. To address this issue they employ cosine-similarity instead of dot-product when they apply weights to features.


# Methodology

Base dataset, defined as,

$$ D_{train} = \bigcup_{b=1}^{K_{base} } \left\{ x_{b,i} \right\}_{i=1}^{N_b}$$

will be used to generate base category classifier. After that we want to modify the base model a little bit using novel dataset, defined by,

$$ D_{novel} = \bigcup_{n=1}^{K_{novel} } \left\{ x'_{n,i} \right\}_{i=1}^{N'_n}$$


## ConvNet-base recognition model

The ConvNet-based recognition model is not that much special. It is just a regular neural network classifier for $K_{base}$ categories. It consists of two component: feature extractor, $F(\cdot | \theta)$, where $\theta$ being learnable parameters and classifier $C(\cdot | W^*)$ where $W^* = \{ w_k^* \in \mathbb R^d \}_{k= 1}^{K^*}$, the $K^*$ many sets of learnable classification weights of size $d$. I.e., Classifier has $K^*$ many classification vectors. The classifier get feature representation $z$ as input and results in the score vector of size $K^*$, namely, $p = C(z|W^*)$.

For the easeness of understanding, you can think of the network right before the output layer as feature extractor and the part after feature extractor can be considered as classifier. As I said before, it is just a regular network with somewhat different notation.

For the single traing phase, it can be said that we are searching for the optimial paraemter $\theta$ for $W^* = W_{base}$.

## Few-shot classification weight generator

During test time, this step modifiy weights from recognition model to be able to recognize newly introduced novel categories by assimilating them from base categories. For each category $n \in [1, N_{novel}]$, the few-shot classification weight generator $G(.,.|\phi)$ gets $Z'_n = \{z'_{n, i}\}_{i=1}^{N'_n}$, where $z'_{n, i} = F(x'_{n,i} | \theta)$ to generator weights for novel categories. Here $\phi$ is a set of learnable parameters.

> Note that novel and base data share the same feature extractor, $F(\cdot | \theta)$.

Weight generator generates the weights for novel category using information from extracted feature from novel categories and weights from base categories as 

$$w'_n = G(Z'_n, W_{base}|\phi)$$

Therefore if we deonte $W_{novel} = \{ w'_n\}_{n=1}^{K_{novel}}$ as weight vector of size $d$ for novel categoreis, the classifier can be written,

$$C(\cdot | W^*), W^* = W_{base} \bigcup W_{novel}$$

By appending weights for novel categories, which borrows information from base category classification task, to weights for base case, we can quickly classify novel categories from the base one without losing classification power for the base one.

The following figure depicts all components of the model


![dynamic_few_shot_learning](/assets/dynamic_few_shot_learning.png)



## Cosine-similarity based recognition model

ConvNet only differs from standard neural networks in the way calculating final score for the classifier. Suppoisng that an example with extracted feature, $z$, regular neural netwoks calculate the score for the $k$ the category first by $s_k = z^Tw_k^*$, where $w^*_k$ is the $k$-th classification weight vector in $W^*$ and then $p_k = softmax(s)$.

But convnet may differ the scale of $w_k$'s for the novel categories, i.e., $w_k \in W_{novel}$, which calculated separately. This is because base learner involves so many data to train with and those parameter evolves very slowly and smoothly with small SGD steps over the course of their training, but the novel classification weights are dynamically predicted by weight generator based on input feature vectors that may vary a lot.

To overcome this issue, they do the folloiwng

$$ x_k = \tau \cdot cos(z, w_k^*) = \tau \cdot \bar z^T \bar w_k^*, $$

where $\bar z = \frac z {||z||}$ and $\bar w_k^* = \frac{w_k^*}{||w_k^*||}$.