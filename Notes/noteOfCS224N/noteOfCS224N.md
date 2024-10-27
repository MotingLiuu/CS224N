# Natural Language Processing with Deep Learning

## Language Models
$$
p(w_1,...,w_m)=\prod_{i=1}^{i=m}P(w_i|w_1,...,w_{i-1})\approx\prod_{i=1}^{i=m}P(w_i|w_{i-n},...,w_{i-1})
$$

### n-gram Language Models
$$
p(w_3|w_1,w_2)=\frac{count(w_1,w_2,w_3)}{count(w_1,w_2)}
$$
**Two main issues**
1. Sparsity problems 
e.g If $w_1,w_2$ and $w_3$ never appear together in the corpus, the probability of $w_3$ is 0. To solve this, a small $\delta$ should be added to the count for each word in the vocabulary.
If $w_1$ and $w_2$ never occurred together in the corpus, then no probability can be calculated for $w_3$
数据集有限，导致很可能不能为所有的组合计算出一个概率。

2. Storage problems with n-gram Language models
we need to store the count for all n-grams we saw in the corpus.
需要存储所有n-1个单词同时出现和n和单词同时出现的次数。模型随着n呈指数型增长

### Window-based Neural Language Model
Word2vec

## Recurrent Neural Networds(RNN)
![alt text](image.png)
Each vertical rectangular box is a hidden layer at a time-step $t$. 
$$
h_t=\sigma(W^{(hh)}h_{t-1}+W^{(hx)}x_t)
$$
$$
\hat{y}_t=softmax(W^{(S)}h_t)
$$
The parameter in the network
* x_1,...,x_{t-1},x_t,x_{t+1},...,x_T: the word vectors corresponding to a corpus with T words.