---
layout: post
title: 'Attention mechanism'
author: kion.kim
date: 2018-06-21 17:00
tags: [deeplearning, nlp, sentence representation, relation network, skip-gram]
---
## Self attention mechanism


Recently, attention mechanism is one of the core techniques in deep learning throughout variety of application fields after its huge success in neural machine translation[Cho et al., 2015 and references therein]. It improved existing recurrent neural network(RNN) based NMT algorithms significantly by relaxing the assumption that all the information from input sentences should be extracted in a single hidden vector.

![Bahdandau et al., 2015](/assets/Bahdandau_attention.png)

In the above, RNNsearch-50 is the result form the model with soft attention mechanism and we can see that BLEU score does not drop as input sentence gets longer and the authors believe that attention mechanism helped conveying huge information that cannot be retained with relatively small hidden vector.
Stated that target tokens(words) are generated sequentially using the information from input sentence and previously generated tokens in neural machine translation, RNN based models utilize the identical sentence representation regardless of the words that is currently generated. 

![https://towardsdatascience.com/sequence-to-sequence-tutorial-4fde3ee798d8](/assets/seq_to_seq.png)

 Different from those models, attention mechanism allows the input sentence representation can vary across the word currently being generated through context vector. With attention mechanism, we can let the algorithm decide which part of the input sentence has more impact on a word on output sentence.

![https://medium.com/datalogue/attention-in-keras-1892773a4f22](/assets/seq_seq_att_example.png)

Attention mechanism also can be thought of as a great reconcile between CNN based methods focusing on a part of sentence and those based on RNN that  retrieve information from the entire input sentence.