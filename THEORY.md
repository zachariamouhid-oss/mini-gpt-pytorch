# Mathematics and fundamental concepts for generative AI

## Introduction 

This file can be considered as a introductory course to get a first step in deep learning and generative AI. Who is it for ? It is mainly made for students who wants to build things on their own, and especially for ones with some mathematical background. As I described in **README.md**, an undergraduate level and and Python basis are required to be able to understand concepts covered here. It will also contain the explanation/justification of the different choices I had to make during the building of my GPT, such as architecture decisions or parameter defitions.

Before digging in, let's present the goal : create a **GPT**, a *G*enerative *P*re-trained *T*ransformer. Already three terms to explain. Generative means that the model is able to create a content (an image, text...), but without copying something that already exists. A transformer is an architecture of neural networks which allows current LLMs to understand sentences by affecting meaning to words in function of their relative position. Finally, pre-trained just signifies that the neural network has already seen a (big) quantity of data to be more accurate for generation. This is what you will be able to build through this file, and in particular focus on the maths modelisation behind the hood.

All the following is written to be read in order, and every references can be found at the end of this paper. 

## Table of contents

- Deep learning fundamentals
- Vector representation of texts
- Transformer architecture
- Learning and training of a model
- Text generation
- Conclusion

## 1) Deep learning fundamentals

In this section, we will more precisely focus on deep generative learning. The deep learning is an artificial intelligence method (AI) in which computers learn to treat data like humans. With a generative aspect, we add a creation aspect to our model.

### 1) Introduction to perceptrons and neural networks

The perceptron, introduced by Rosenblatt in 1958, is one of the earliest mathematical models of an artificial neuron. Given an input vector $(x \in \mathbb{R}^d)$, it computes a weighted sum of the inputs and applies a step function to produce a binary output:
$$
f(x) =
\begin{cases}
1 & \text{if } w^T x + b \geq 0,\\
0 & \text{otherwise},
\end{cases}
$$
where $(w \in \mathbb{R}^d)$ is the weight vector and $(b \in \mathbb{R})$ is the bias. The perceptron can therefore be interpreted as a binary linear classifier.

Modern artificial neurons generalize this idea by replacing the step function with an activation function $(\phi)$, typically nonlinear. A neuron can thus be defined as a function
$$
f : \mathbb{R}^d \rightarrow \mathbb{R},
\qquad
f(x) = \phi(w^T x + b).
$$
By combining multiple neurons into layers, neural networks (NN) can represent increasingly complex functions.

### 2) From a Single Neuron to Multilayer Perceptrons (MLPs)
#### 2) 1) The need for non-linearity

The use of a nonlinear activation function is essential when building a neural network. Indeed, if the activation function were linear, such that $(\phi(x)=x)$, then a layer would simply compute
$$
h = Wx+b.
$$
Stacking multiple such layers would not increase the expressive power of the network. For example, consider two consecutive layers:
$$
h = W_1x+b_1,
\qquad
y = W_2h+b_2.
$$
Substituting the first equation into the second gives
$$
y = W_2(W_1x+b_1)+b_2
  = (W_2W_1)x + (W_2b_1+b_2),
$$
which is again an affine transformation of the input. Therefore, regardless of the number of layers, the entire network could be reduced to a single linear layer.

A nonlinear activation function breaks this property by introducing nonlinear transformations between layers. This allows neural networks to model complex, nonlinear relationships in the data. This is proven by the Universal approximation theorem (see [Wikipedia](https://en.wikipedia.org/wiki/Universal_approximation_theorem)).

Thus let's present some of the most common and easiest activation functions :

- Heaviside function : $H(x) = \mathbb{1}_{\R_+}$
- ReLU (Rectified Linear Unit) : $ReLU(x) = max(0,x)$
- Sigmoid : $\sigma(x) = \frac{1}{1+e^{-x}}$
- tanh : $tanh(x) = \frac{e^{2x}-1}{e^{2x}+1}$
- SiLU ((Sigmoid Linear Unit) : $\text{SiLU}(x) = x \cdot \sigma(x)$

The main differences between those functions is their differentiability properties. We will discuss it later in this section.

#### 2) 2) MLPs

A single neuron can only compute a relatively simple function of its inputs. To increase the expressive power of the model, multiple neurons can be organized into layers. Given an input $x$ in $\mathbb{R}^{d_0}$, a fully connected layer with $d_1$ neurons computes $h = \phi(Wx+b)$, where $W \in \mathbb{R}^{d_1 \times d_0}$, $b \in \mathbb{R}^{d_1}$, and $\phi$ is applied element-wise. Each row of $W$ therefore corresponds to the weights of one neuron in the layer.

By stacking such several layers, we obtain a Multi-layer Perceptron. For a network with $n$ layers, the computation can be written as :
$$
h^{(0)} = x,\
h^{(k)} = \phi\left(W^{(k)}h^{(k-1)} + b^{(k)}\right),
\ k=1,\ldots,n.
$$

An MLP is thus a feed-forward achitecture, which means that computations are made sequentially, layer after another, without any loop or feed-back connections. Moreover, every neuron of a layer is connected to all neurons of the next one. It is a particularity of MLPs.

#### 2) 3) Training and backward propagation

When first initiliazed (most of the time randomly), a neural network will not give accurate predictions, because its parameters are completely arbritrary. The goal of training is to update the weights $(W^{(k)}_{ij})$ and biais $(b^{(k)}_{i})$ so that for a new given entry, the neural networks will better predict the right output. 

#### 2) 3) 1) The loss function

To mesure how good (or bad) a prediction $\hat{y}$ is compared to the correct answer $y$, we use a loss function $\mathcal{L}(y,\hat{y})$. More precisely, this function must measure the error between a predicted value and the correct one. Training then becomes an optimization problem, it consists on the minimization of the loss function.

#### 2) 3) 2) Backward propagation

To minimize the empirical loss over a dataset, we optimize the network parameters $\theta = \{W^{(k)}, b^{(k)}\}_{k=1}^n$ using gradient descent. The standard update rule with learning rate $\eta > 0$ is:

$$\theta \leftarrow \theta - \eta \, \nabla_\theta \mathcal{L}(y, \hat{y}).$$

Computing these gradients efficiently relies on the chain rule, implemented via the backpropagation algorithm. Let us denote the pre-activation vector at layer $k$ by $z^{(k)} = W^{(k)} h^{(k-1)} + b^{(k)}$, so that $h^{(k)} = \phi(z^{(k)})$. We define the error vector (or sensitivity) at layer $k$ as:

$$
\delta^{(k)} = \frac{\partial \mathcal{L}}{\partial z^{(k)}} \in \mathbb{R}^{d_k}.
$$

1. Base case: Output layer ($k = n$)

At the final layer, $h^{(n)} = \hat{y}$. Applying the chain rule component-wise:
$$
\delta^{(n)}_i = \frac{\partial \mathcal{L}}{\partial z^{(n)}_i} = \frac{\partial \mathcal{L}}{\partial \hat{y}_i} \cdot \phi'(z^{(n)}_i),$$
which vectorizes as:
$$
\delta^{(n)} = \nabla_{\hat{y}} \mathcal{L} \odot \phi'(z^{(n)}),
$$
where $\odot$ denotes the Hadamard product.

2. Backward recurrence ($k < n$)

Using the multivariate chain rule, the sensitivity at layer $k$ can be expressed in terms of the next layer's sensitivity $\delta^{(k+1)}$:

$$
\delta^{(k)}_j = \sum_{i=1}^{d_{k+1}} \frac{\partial \mathcal{L}}{\partial z^{(k+1)}_i} \frac{\partial z^{(k+1)}_i}{\partial h^{(k)}_j} \frac{\partial h^{(k)}_j}{\partial z^{(k)}_j}.
$$

Since $z^{(k+1)}_i = \sum_{j} W^{(k+1)}_{ij} h^{(k)}_j + b^{(k+1)}_i$, we have $\frac{\partial z^{(k+1)}_i}{\partial h^{(k)}_j} = W^{(k+1)}_{ij}$. Thus:

$$
\delta^{(k)} = \left( (W^{(k+1)})^T \delta^{(k+1)} \right) \odot \phi'(z^{(k)}).
$$

3. Parameter gradients

Once the error terms $\delta^{(k)}$ are computed for all layers, the partial derivatives with respect to the weights and biases are obtained immediately:
$$
\frac{\partial \mathcal{L}}{\partial W^{(k)}} = \delta^{(k)} (h^{(k-1)})^T \in \mathbb{R}^{d_k \times d_{k-1}},
$$
$$
\frac{\partial \mathcal{L}}{\partial b^{(k)}} = \delta^{(k)} \in \mathbb{R}^{d_k}
$$

This backward recurrence allows us to compute all parameter gradients in a single backward pass with a computational cost proportional to the forward pass, avoiding redundant computations.

#### 2) 3) 3) Regularization

Commonly, to prevent overfitting (which is when a model fits too closely the exact patterns and noises of the training set that do not generalize), we often add a penalty term in the loss function. The goal shifts to the minimization of 

$$
\mathcal{L}_{total} = \mathcal{L} + \lambda \Omega(\theta)
$$
where $\Omega$ is called the penalty function, and $\lambda > 0$ the regularization coefficient, whose value will accentuate or reduce the penalty.

#### L2 regularization

An example of penalty function is the L2 regularization 

$$
\Omega(\theta) = \frac{1}{2}\|\theta\|_2^2
$$
and the gradient descent becomes 

$$
\theta \leftarrow (1-\eta \lambda) \theta - \eta \nabla_\theta \mathcal{L}
$$

Thanks to this additionnal term, larger values of $\theta_i$ are penalized and shrink the weights toward zero, and it can improve generalization, although this is not guaranteed.

## 2) Vector representation of texts

### 1) Autoregressive formulation

We search, as an output of our model, a probabillity distribution to try to predict the following word of a sequence. Let us denote $\mathcal{V}$ the finite set representing our dictionnary (it contains our vocabulary) and the length of a given sequence. Let us represent a sequence of words as the realization of random variables $(X_1,\ldots, X_T)$. The goal is to estimate $\mathbb{P}(X_1=x_1, \ldots, X_T=x_T)$, simply noted here $\mathbb{P}(x_1, \ldots, x_T)$.

To compute $\mathbb{P}(x_1, \ldots, x_T)$, we use the chain rule of probability, which allows us to write : 
$$
\mathbb{P}(x_1, \ldots, x_T) = \mathbb{P}(x_1)\prod_{t=2}^{T} \mathbb{P}(x_t \mid x_1 \ldots, x_{t-1})
$$

Then we understand the meaning behind an "autoregressive" model, which describes a process in which the future value depends on the past values. To be concrete, suppose "couch" and "plane" are in a small vocabulary set $\mathcal{V}$. For the sequence "The cat is sleeping on the", our model should predict that the probability of "The cat is sleeping on the couch" is almost one, unlike "The cat is sleeping on the plane". The role of the model $\theta$ is then to estimate $$P_\theta(v \mid x_{<T}) \quad \text{for all } v \in V$$


### 2) Negative Log likelihood or Cross Entropy

To train our model, given the sentence "The capital of France is", we now need to maximize $\mathbb{P}("Paris" \mid x_{<T})$. We are then looking for a model $\theta^*$ which satisfies :
$$
\theta^* = \arg\max_\theta \prod_{t=1}^N P_\theta(X_t = x_t^{\text{observed}} \mid x_{<t})
$$

To do so, we take the $log$, as it is strictly increasing, we have :
$$
\arg\max_\theta \prod_{t=1}^T P_\theta(x_t \mid x_{<t}) = \arg\max_\theta \sum_{t=1}^T \ln P_\theta(x_t \mid x_{<t})
$$
And current algorithm are engineered to minimize a loss, we take the opposite :
$$\arg\min_\theta \underbrace{\left( - \sum_{t=1}^T \ln P_\theta(x_t \mid x_{<t}) \right)}_{\text{Negative Log-Likelihood (NLL) or Cross Entropy loss}}$$

If a model predicts the correct word with $P = 1$, then $-\ln(1) = 0$ (no loss). However, if the model gives probability $P \to 0$ to the right answer, then $-\ln(P) \to +\infty$ (infinite loss).

### 3) Tokenization and Byte-Pair Encoding 

As you may know, we won't be working with strings directly in the model. Of course, this is because we want to do calculations, and strings don't allow this. Then we want to transform a sentence into numbers, named tokens, i.e. we want a map $\mathcal{V}$ that contains the way we encoded our text into tokens.

A first strategy would be to make a dictionnary $\mathcal{V}$ with all the possible letters in every languages, and associate each ones to a unique integer. It the main issue is that texts would be too long to be treated because we would have to consider letters one after another; which implies to much computing power and generating a single word would need too many operations.

Then, one could think that encoding word by word will work. The issue here is that the vocabulary size $|V|$ would be enormous and the model would be incapable to deal with typos, and also complete words that he never saw during his training.

Finally, the solution presented here (and that is used in modern LLMs) is the Byte Pair Encoding (BPE). We can see it as an in-between of the two solutions presented before. The principles is as follows :

1. Set a vocabulary size $|V|$ 
2. Consider the whole chain of characters as n-adjacent letters
3. Find the most frequent pairs of this chain
4. Merge all those pairs with a new token 
5. Repeat from step 2 until vocabulary size is reached 

For example, suppose we are encoding "aaabdaaabac", with a specified vocabulary size of $|V| = 6$, then it would first be encoded as $"0, 0, 0, 1, 2, 0, 0, 0, 1, 0, 3"$ with a vocabulary of $"a=0,\ b=1,\ d=2,\ c=3"$. After 2 loop in the algorithm, we would obtain $"4, 5, 2, 4, 5, 0, 3"$ with a vocabulary of $"a=0,\ b=1,\ d=2,\ c=3,\ aa=4,\ ab=5"$.

In the above example, the output of the BPE is a vocabulary, which can be used to encode any text that is written with the letters "abcd". It will not be able to encode text containing other symbols, such as "e". The byte-level converts the text into UTF-8 first, and treat it as a stream of bytes. This guarantees that any text encoded in UTF-8 can be therefor encoded by the BPE.

#### Byte-level initial vocabulary ($\vert{}\mathcal{V}_{\text{base}}\vert{} = 256$)

To avoid the out-of-vocabulary problem entirely, byte-level BPE initializes the vocabulary with all 256 possible byte values ($0 \dots 255$):
$$\mathcal{V}_0 = \{0x00, 0x01, \dots, 0xFF\}$$
Any text serialized via UTF-8 is fundamentally a sequence of bytes $b \in \{0, \dots, 255\}$. Consequently:
* **No unknown tokens**: Every valid UTF-8 sequence can be parsed, eliminating the need for an `<unk>` fallback token.
* **Bounded vocabulary size**: Starting at $256$, the algorithm greedily adds merge tokens up to a fixed threshold $\vert{}\mathcal{V}\vert{}$.

In addition to the learned byte merges, language models rely on reserved control tokens. The most essential is the end-of-sequence token <|endoftext|>, placed at the terminal index of the vocabulary ($\vert{}\mathcal{V}\vert{} - 1$). It serves as an absorbing boundary condition between independent documents in the training corpus and signals the model to halt inference during generation.

### 4) The embedding table

An embedding table, represented as a matrix $W_e \in \R^{|V| \times d_{model}}$, is the object that contains all the meaning of a token. How it works ?
It is a matrix in which each line corresponds to the vector representation $x_t$ of a token from se. By having a big $d_{model}$ (for my GPT, it is set to $512$), it allows vectors to capture the meaning of tokens during training.

#### One-Hot Representation and Embedding Lookup

Let $t \in \{0, \dots, \vert{}\mathcal{V}\vert{}-1\}$ be a token identifier. Its canonical algebraic representation is the one-hot indicator vector $\mathbf{e}_t \in \{0, 1\}^{\vert{}\mathcal{V}\vert{}}$, defined by:$$(\mathbf{e}_t)_i = \begin{cases} 1 & \text{if } i = t, \\ 0 & \text{otherwise}. \end{cases}$$We introduce a trainable embedding matrix $\mathbf{W}_e \in \mathbb{R}^{\vert{}\mathcal{V}\vert{} \times d_{\text{model}}}$. The continuous vector representation $\mathbf{x}_t \in \mathbb{R}^{d_{\text{model}}}$ corresponding to token $t$ is computed via:$$\mathbf{x}_t = \mathbf{e}_t^T \mathbf{W}_e.$$In practice, rather than computing a sparse matrix-vector multiplication of dimension $\vert{}\mathcal{V}\vert{}$, this operation is evaluated as a constant-time table lookup selecting the $t$-th row of $\mathbf{W}_e$:$$\mathbf{x}_t = \mathbf{W}_e[t, :].$$For an entire input sequence of length $T$, stacking the embeddings row-wise yields the sequence tensor:$$\mathbf{X} = \begin{pmatrix} \mathbf{x}_1 \\ \vdots \\ \mathbf{x}_T \end{pmatrix} \in \mathbb{R}^{T \times d_{\text{model}}}.$$

## 3) Transformer architecture

Suppose now that an input is no longer a text, but a tensor $\mathbf{X} = \begin{pmatrix} \mathbf{x}_1 \\ \vdots \\ \mathbf{x}_T \end{pmatrix} \in \mathbb{R}^{T \times d_{\text{model}}}$, with $x_i$ the vector representation of token $i$ in a sequence of $T$ tokens. This is the input of the model that we are about to describe. 

### 1) Attention mechanism

Introduced by *Google* in 2017 in the paper *Attention is all you need*, this is the mechanism behind every modern generative AI. First, we project $\mathbf{X}$ into three learned vectors 
$$
{Q} = \mathbf{X} {W}_Q, \qquad {K} = \mathbf{X} {W}_K, \qquad {V} = \mathbf{X} {W}_V
$$

Those tensors $Q,\ K, \ V$ are the center of the attention mechanism. We interpret them as follows :

- Queries ($Q$): Represent what each token is looking for.
- Keys ($K$): Represent what each token represents.
- Values (${V}$): Represent the actual content, the semantic of a token.

The interaction between the $i$-th and $j$-th tokens is quantified with the inner product $s_{ij} = \langle q_i, k_j \rangle = q_i k_j^T$, where $q_i$ is the query of token $i$, and $k_j$ the key of token $j$. Stacking all pairs yields the raw affinity matrix:
$$
S =QK^T \in \R^{T \times T}
$$ 
This inner product is proportionnal to the angle between those two vectors. Let's visualize this through an example. Consider the sentence: "The cat ate the mouse because **he** was hungry". Let token $i$ correspond to "he", and tokens $j_1, j_2$ correspond to "cat" and "mouse" ($j_1, j_2 < i$) : 

- Query formulation (${q}_{\text{he}}$) : The representation of the ambiguous pronoun "he" projects into a query asking:$$\mathbf{q}_{\text{he}} \approx \text{"Find an antecedent noun that is an agent/subject capable of hunger."}$$

- Key responses (${k}_{\text{cat}}, {k}_{\text{mouse}}$) : The noun "cat" (subject of the clause) emits :

$$
{k}_{\text{cat}} \approx \text{"Feline animal, active subject of 'ate'."}
$$ 

The noun "mouse" (direct object) emits :

$$
{k}_{\text{mouse}} \approx \text{"Small rodent, patient/object of 'ate'."}
$$

- Inner product evaluation : 

Because the feature vectors ${q}_{\text{he}}$ and ${k}_{\text{cat}}$ align strongly in the latent space (their representations are nearly collinear), their inner product is high:$$\langle {q}_{\text{he}}, {k}_{\text{cat}} \rangle \gg \langle {q}_{\text{he}}, {k}_{\text{mouse}} \rangle.$$


Then, the full causal self-attention operation is defined by the following expression: 

$$\text{Attention}({Q}, {K}, {V}) = \text{Softmax}\left( \frac{{Q} {K}^T}{\sqrt{d_{\text{head}}}} + {M} \right) {V}$$
Where Softmax is defined as $(\vec{x} \in \R^n \mapsto \frac{\exp(\mathbf{\vec{x}})}{\sum_{j=1}^n \exp(x_j)} \in \R^n)$

Let us break down each component of this equation step by step.

1. The scaling by $\sqrt{d_{\text{head}}}$ : $d_{\text{head}}$ represents the dimension of the latent space where the attention mechanism takes place. Experiments showed that without this term, the inner product tends to diverge to infinity, then Softmax exponiantially tends toward zeros, which causes gradient to be null, and learning slows too much (see the [Vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)).
2. Causal masking : because language generation is autoregressiv, we want that a token at the $i$-th position can't see the "future", which means token $i$ must never interact with $j \gt i$. Thus we add $M$ a matrix defined by :
$$
M_{ij} = \begin{cases}  0 & \text{if } j \le i, \\  -\infty & \text{if } j > i. \end{cases}
$$
So that when evaluating the Softmax across row $i$:
$$
\text{Softmax}\left( \frac{{q}_i {K}^T}{\sqrt{d_{\text{head}}}} + {M}_{i, :} \right)_j = \frac{\exp\left( \frac{s_{ij}}{\sqrt{d_{\text{head}}}} + M_{ij} \right)}{\sum_{k=1}^T \exp\left( \frac{s_{ik}}{\sqrt{d_{\text{head}}}} + M_{ik} \right)}.$$
For any future position $j > i$, $M_{ij} = -\infty$, which yields:
$$
\exp(-\infty) = 0 \implies A_{ij} = 0.
$$

In facts, we use multi-head attention running in parrallel, that we can see as an extension of self attention mechanism. We usually divide $d_\text{model}$ in multiple heads (in my code, you will notice $\frac{d_\text{model}}{d_\text{heads}}=\frac{512}{64}= 8$ heads noted $H$) so that each one will be specialized on one specific task (such as the grammar, syntactic strutures...).
Then, for each head, we compute $\text{Attention}({Q_h}, {K_h}, {V_h})$ for all $h \in \{1, \dots, H\}$ : 
$$
Q_h = \mathbf{X} W_Q^{(h)}, \quad K_h = \mathbf{X} W_K^{(h)}, \quad V_h = \mathbf{X} W_V^{(h)} \quad \text{avec } W^{(h)} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}
$$

The heads are then concatenated along the feature dimension and projected back to the original model dimension through an output matrix $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$:
$$
\text{MHA}(\mathbf{X}) = \text{Concat}\left(\text{head}_1, \dots, \text{head}_H\right) W_O.
$$
This factorization allows the model to attend simultaneously to diverse linguistic relationships (e.g., local syntactic agreement on one head, long-range semantic coreference on another).

### 2)  Positional embedding

According to the attention formula, the mechanism is currently agnostic to the relative and absolute positions of tokens. Indeed, the sentences 'The cat ate the mouse.' and 'The mouse ate the cat.' (or even "The the ate cat mouse") would yield the same representation for the same words. Indeed, consider $P$ a permutation matrix $(PP^T=I_n)$ which changes the order of words, i.e. ${Q'}=P\mathbf{X}W_{Q}=PQ$, ${K'}=P\mathbf{X}W_{K}=PK$, ${V'}=P\mathbf{X}W_{V}=PV$. Without any causal mask and any positional information :

$$\text{Attention}({Q'}, {K'}, {V'}) = \text{Softmax}\left( \frac{{PQ} ({PK})^T}{\sqrt{d_{\text{head}}}}\right) {PV} = \text{Softmax}\left( \frac{{PQ} K^TP^T}{\sqrt{d_{\text{head}}}}\right) {PV}=P\ \text{Softmax}\left( \frac{{Q} K^T}{\sqrt{d_{\text{head}}}}\right) {V}
$$

Then :
$$
\text{Attention}({Q'}, {K'}, {V'}) = P\ \text{Attention}({Q}, {K}, {V})
$$

In this case, the internal dynamic remains the same, and the model won't see any difference for all permutations. Then, let's introduce RoPE, the rotary positional encoding.


#### Rotary positional encoding

Instead of adding an absolute position vector at a position $m \in \{0, \dots, T-1\}$, i.e. $x'_m = x_m + p_m$, we are looking for an operation such that the inner product between $q_i$ and $k_j$ only depends on $i-j$ :
$$\langle R({q}_i, i), R({k}_j, j) \rangle = g({q}_i, {k}_j, i-j)$$

To do so, let's first get into a 2D representation. Let us denote the rotation matrix
$$\mathbf{R}_{\theta, m} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \in \text{SO}(2)$$
Then we have : $$\langle \mathbf{R}_{\theta, m} {q}_m, \mathbf{R}_{\theta, n} {k}_n \rangle = {q}_m^T \mathbf{R}_{\theta, m}^T \mathbf{R}_{\theta, n} {k}_n = {q}_m^T \mathbf{R}_{\theta, n - m} {k}_n = {q}_m^T \mathbf{R}_{\theta, -(m - n)} {k}_n$$

With this operation, the spatial representation between two tokens in positions $m$ and $n$ only depends on their relative positions in the sequence $m-n$.


#### Generalization to Dimension $d_{\text{head}}$

In our architecture, the attention vectors reside in a latent space of dimension $d_{\text{head}}$ (with $d_{\text{head}}$ even, e.g., $d_{\text{head}} = 64$). We decompose $\mathbb{R}^{d_{\text{head}}}$ into an orthogonal direct sum of $d_{\text{head}} / 2$ two-dimensional subspaces:
$$
\mathbb{R}^{d_{\text{head}}} = \bigoplus_{i=0}^{d_{\text{head}}/2 - 1} \mathbb{R}^2.
$$

We define a block-diagonal rotation matrix $R_{\Theta, m} \in \mathbb{R}^{d_{\text{head}} \times d_{\text{head}}}$:
$$
R_{\Theta, m} = \text{diag}\left( \mathbf{R}_{\theta_0, m}, \, \mathbf{R}_{\theta_1, m}, \, \dots, \, \mathbf{R}_{\theta_{d_{\text{head}}/2 - 1}, m} \right),
$$
where each frequency $\theta_i$ decreases geometrically according to:
$$
\theta_i = \theta_{\text{base}}^{-2i / d_{\text{head}}}, \qquad i \in \left\{0, 1, \dots, \frac{d_{\text{head}}}{2} - 1\right\},
$$
with standard base hyperparameter $\theta_{\text{base}} = 10\,000$.

By block-orthogonality, the inner product between rotated vectors preserves the relative distance property across all dimensions:
$$
\langle R_{\Theta, m} \mathbf{q}_m, R_{\Theta, n} \mathbf{k}_n \rangle = \mathbf{q}_m^T R_{\Theta, m}^T R_{\Theta, n} \mathbf{k}_n = \mathbf{q}_m^T R_{\Theta, n - m} \mathbf{k}_n.
$$

#### Efficient Vectorized Implementation

Instantiating an explicit sparse matrix $R_{\Theta, m} \in \mathbb{R}^{d_{\text{head}} \times d_{\text{head}}}$ for every position $m$ is computationally inefficient due to the extreme sparsity of the matrix. Instead, we vectorize the operation using element-wise Hadamard products.

A 2D rotation of a coordinate pair $(x_1, x_2)$ can be expressed as:
$$
\begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} \cos(m\theta) + \begin{pmatrix} -x_2 \\ x_1 \end{pmatrix} \sin(m\theta).
$$

In our tensor implementation, rather than rotating consecutive pairs $(x_{2i}, x_{2i+1})$, the vector $\mathbf{x} \in \mathbb{R}^{d_{\text{head}}}$ is split into two halves of dimension $d_{\text{head}}/2$:
$$
\mathbf{x} = \begin{pmatrix} \mathbf{x}_{\text{first}} \\ \mathbf{x}_{\text{second}} \end{pmatrix}.
$$
We define the transformed vector $\text{rotate\_half}(\mathbf{x})$:
$$
\text{rotate\_half}(\mathbf{x}) = \begin{pmatrix} -\mathbf{x}_{\text{second}} \\ \mathbf{x}_{\text{first}} \end{pmatrix}.
$$
The rotation of tensor $\mathbf{x}$ at position $m$ simplifies to:
$$
R_{\Theta, m} \mathbf{x} = (\mathbf{x} \odot \cos_m) + (\text{rotate\_half}(\mathbf{x}) \odot \sin_m),
$$
where $\cos_m$ and $\sin_m$ are precomputed broadcasted tensors containing duplicated angles across both halves:

$$
\cos_m = \text{Concat}(\cos(m \Theta), \cos(m \Theta)) \in \mathbb{R}^{d_{\text{head}}}.
$$

This vectorization requires $\mathcal{O}(d_{\text{head}})$ operations per token without dynamic matrix allocations.


3) Normalization: From LayerNorm to RMSNorm

As neural networks grow deeper, the distributions of intermediate activations shift across layers during optimization. This phenomenon destabilizes learning dynamics, requiring exceedingly small learning rates. Normalization layers enforce scale invariance and stabilize the variance of activations.

#### Layer normalization
The standard Layer Normalization (LayerNorm), introduced by Ba et al. (2016), normalizes an activation vector $\mathbf{x} \in \mathbb{R}^{d}$ across its feature dimension by centering it around its empirical mean $\mu$ and scaling by its standard deviation :

$$\mu = \frac{1}{d} \sum_{i=1}^d x_i, \qquad \sigma = \sqrt{\frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2 + \epsilon},$$
$$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sigma} \odot \mathbf{g} + \mathbf{b},$$

where $\mathbf{g}, \mathbf{b} \in \mathbb{R}^d$ are learnable affine gain and bias parameters, and $\epsilon > 0$ prevents division by zero.

#### Root Mean Square Normalization (RMSNorm)

Zhang and Sennrich (2019) demonstrated that the computational benefits of LayerNorm stem predominantly from its scaling invariance rather than its mean-centering property. Shift-invariance via $\mathbf{x} - \mu$ contributes negligible regularization while introducing significant memory bandwidth overhead.

RMSNorm discards mean centering and normalizes activations strictly by their root-mean-square statistic :

$$
\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon} = \frac{\Vert{}\mathbf{x}\Vert{}_2}{\sqrt{d}} + \mathcal{O}(\epsilon).
$$

The normalized output is scaled by a single learnable weight parameter $\mathbf{g} \in \mathbb{R}^d$ (with $\mathbf{b} = \mathbf{0}$):

$$
\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \mathbf{g}.
$$

Mathematical Properties:
- Scale Invariance: For any scalar $\alpha > 0$, $\text{RMSNorm}(\alpha \mathbf{x}) = \text{RMSNorm}(\mathbf{x})$. The magnitude of activations cannot explode or vanish across successive layers.
- Computational Efficiency: Calculating $\text{RMS}(\mathbf{x})$ requires a single accumulation of squared entries without intermediate passes to compute and subtract $\mu$, saving memory read/write cycles on GPU kernels.

4) Residual Connections and Pre-LN Architecture

In deep architectures, stacking non-linear transformations impairs backward gradient flow : gradients passing through multiple Jacobian matrices $\prod_{l} \frac{\partial \mathbf{h}^{(l)}}{\partial \mathbf{h}^{(l-1)}}$ decay exponentially.To resolve this, He et al. (2016) introduced residual skip connections:

$$
\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + \mathcal{F}(\mathbf{x}^{(l)}).
$$

The gradient with respect to an earlier activation $\mathbf{x}^{(l)}$ decomposes into:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(l)}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{(l+1)}} \left( \mathbf{I} + \frac{\partial \mathcal{F}(\mathbf{x}^{(l)})}{\partial \mathbf{x}^{(l)}} \right).$$

The identity matrix $\mathbf{I}$ creates an uninterrupted gradient highway, ensuring that even if $\frac{\partial \mathcal{F}}{\partial \mathbf{x}^{(l)}} \approx 0$, gradients flow back directly without vanishing.

#### Post-LN vs. Pre-LN
The original Transformer placed normalization layers after the residual addition (Post-LN):
$$
\mathbf{x}^{(l+1)} = \text{Norm}\left(\mathbf{x}^{(l)} + \mathcal{F}(\mathbf{x}^{(l)})\right).
$$

In Post-LN, gradients passing through the residual connection are repeatedly scaled by the derivative of the normalization operator, destabilizing the initial optimization phases and requiring strict warmup schedules. Modern architectures (LLaMA, and our implementation) adopt the Pre-LN configuration, applying normalization strictly to the branch inputs prior to sub-layer transformations:
$$\mathbf{x}^{(l+1/2)} = \mathbf{x}^{(l)} + \text{Attention}\left(\text{RMSNorm}_1(\mathbf{x}^{(l)})\right),$$

$$\mathbf{x}^{(l+1)} = \mathbf{x}^{(l+1/2)} + \text{FFN}\left(\text{RMSNorm}_2(\mathbf{x}^{(l+1/2)})\right)$$

Because the identity path remains unnormalized, signals propagate unhindered across all $N$ layers, significantly enhancing training stability.

### 5) The Feed-Forward Network (FFN)

While the multi-head self-attention module allows tokens to route information dynamically across the sequence dimension, it computes linear mixtures across token positions. The Feed-Forward Network (FFN) provides token-wise non-linear transformations in the feature space $\mathbb{R}^{d_{\text{model}}}$.
#### Classical MLP Block
The standard Transformer FFN expands the hidden dimension from $d_{\text{model}}$ to an intermediate projection dimension $d_{\text{ff}} = 4 \times d_{\text{model}}$, applies an activation function, and projects back down:
$$\text{FFN}(\mathbf{x}) = W_{\text{down}} \, \phi\left(W_{\text{up}} \mathbf{x}\right),$$
where $W_{\text{up}} \in \mathbb{R}^{4 d_{\text{model}} \times d_{\text{model}}}$ and $W_{\text{down}} \in \mathbb{R}^{d_{\text{model}} \times 4 d_{\text{model}}}$ (biases are omitted in modern configurations).

#### Smooth Activations: The SiLU / Swish Function
Classical architectures relied on the non-smooth $\text{ReLU}(x) = \max(0, x)$. Modern architectures favor smooth, non-monotonic activation functions such as SiLU (Sigmoid Linear Unit, also known as Swish):

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}.$$

Mathematical Advantages over ReLU:
- Continuous Differentiability: The derivative of $\text{SiLU}$ is everywhere continuous: $$\frac{d}{dx} \text{SiLU}(x) = \sigma(x) + x \sigma(x)(1 - \sigma(x)) = \sigma(x)\left(1 + x(1 - \sigma(x))\right).$$ Unlike $\text{ReLU}$, which has an abrupt gradient discontinuity at $x=0$, $\text{SiLU}$ provides a smooth loss landscape.
- Non-Monotonicity and Negative Saturation : For small negative values ($x \in [-1.28, 0]$), $\text{SiLU}(x) < 0$. This smooth lower bound allows small negative signals to propagate through backpropagation without producing completely deactivated "dead" neurons as observed in $\text{ReLU}$.

#### 6) Full Transformer Block Assembly
A single decoder block combines Multi-Head Attention and the Feed-Forward network through Pre-RMSNorm residual layers. Given an input representation $\mathbf{X}^{(l)} \in \mathbb{R}^{T \times d_{\text{model}}}$ at layer $l$ and precomputed RoPE buffers $(\cos, \sin)$:

$$\mathbf{X}_{\text{norm1}} = \text{RMSNorm}_1\left(\mathbf{X}^{(l)}\right) \text,$$

$$\mathbf{X}^{(l+1/2)} = \mathbf{X}^{(l)} + \text{Attention}\left(\mathbf{X}_{\text{norm1}}, \, \cos, \, \sin\right),$$
$$\mathbf{X}_{\text{norm2}} = \text{RMSNorm}_2\left(\mathbf{X}^{(l+1/2)}\right),$$
$$\mathbf{X}^{(l+1)} = \mathbf{X}^{(l+1/2)} + \text{FFN}\left(\mathbf{X}_{\text{norm2}}\right) .$$

Stacking $L$ such blocks forms the backbone of the model. A final $\text{RMSNorm}$ normalizes the output of the terminal block before projecting activations onto vocabulary logits through the tied language modeling head:

$$
\mathbf{Z} = \text{RMSNorm}_{\text{final}}\left(\mathbf{X}^{(L)}\right) \mathbf{W}_e^T \in \mathbb{R}^{T \times \vert{}\mathcal{V}\vert{}}
$$

## 4) Learning and training

Now that our model is built and suppose that we have a training set available, we need an algorithm to train our model. We already presented in the previous section the loss function we are going to use (the cross-entropy loss), the backpropagation and regularization principles.

### 1) Adam algorithm

Adam stands for Adaptive Moment Estimation. It's an algorithm whose goal is to upgrade the stochastic gradient descent (SGD). The main issue of SGD is that each parameters has the same global learning rate $\eta$. Thus, a variable associated with a rare event in the training set, it may not be learnt fast enough compared to more frequent events. Mathematically, if the loss function exhibits steep gradients along one coordinate $\theta_i$ but lies on a flat plateau along another coordinate $\theta_j$, choosing a single $\eta$ leads to an unavoidable dilemma: 
- If $\eta$ is large enough to make progress along $\theta_j$, it causes $\theta_i$ to oscillate or diverge.
- If $\eta$ is small enough to stabilize $\theta_i$, training along $\theta_j$ stalls.

Adam resolves this by assigning an individual, dynamically adjusted step size $\eta_{t, i}$ to every single scalar parameter $\theta_i$.The update rule coordinate by coordinate becomes:
$$
\theta_{t, i} = \theta_{t-1, i} - \underbrace{\left( \frac{\eta}{\sqrt{\hat{v}_{t, i}} + \epsilon} \right)}_{\eta_{t, i} \text{ (effective learning rate)}} \hat{m}_{t, i}
$$

Let us explain all new terms. First, let's introduce those two components :
- $m_{t, i}$, the first moment momentum. It is defined as an exponentially decaying average of past gradients. With $g_{t,i}$ the gradient of the loss at step t: $$m_{t,i} = \beta_1 m_{t-1,i} + (1 - \beta_1) g_{t,i}$$ $\beta_1$ is an hyperparameter, generally equals $0.9$. This momentum accelerates the gradient descent by using past values of gradients to adapt each steps. Thus, it follows coherent direction of the gradient to generaly use more optimal paths towards the minimum for $\theta$ compared to classic SGD.
-  $v_{t, i}$, the second moment momentum, defined as follows : $$v_{t,i} = \beta_2 v_{t-1,i} + (1 - \beta_2) g_{t,i}^2$$ It avoids abnormal values of the gradient by keeping in memomry its past values, and adapts the learning rate.
- $\epsilon = 10^{-8}$, prevents dividing by $0$

Then, because those two momentums are initialized to zero, it exists a bias for both of them :
$$
   \mathbb{E}[m_t] = \mathbb{E}\left[ (1 - \beta_1) \sum_{k=1}^t \beta_1^{t-k} {g}_k \right] = (1 - \beta_1^t) \mathbb{E}[{g}_t],
$$
$$
   \mathbb{E}[v_t] = \mathbb{E}\left[ (1 - \beta_2) \sum_{k=1}^t \beta_2^{t-k} {g}_k^2 \right] = (1 - \beta_2^t) \mathbb{E}[{g}_t^2].
$$


We then use $\hat{m}_{t,i} = \frac{m_{t,i}}{1-\beta_1^t}$ and  $\hat{v}_{t,i} = \frac{v_{t,i}}{1-\beta_2^t}$ to correct this bias.

### 2) Decoupled Weight Decay: From Adam to AdamW

To prevent parameters from growing excessively large and overfitting the training corpus, optimization objectives incorporate regularization penalties:
$$
\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}(\theta) + \frac{\lambda}{2} \|\theta\|_2^2, \qquad \lambda > 0.
$$

#### The Failure of $L_2$ Regularization in Adam
Under standard SGD, computing the gradient of $\mathcal{L}_{\text{total}}$ yields $\nabla_\theta \mathcal{L}_{\text{total}} = \nabla_\theta \mathcal{L} + \lambda \theta$. The update step expands as:
$$
\theta_t = \theta_{t-1} - \eta (\nabla_\theta \mathcal{L} + \lambda \theta_{t-1}) = (1 - \eta \lambda) \theta_{t-1} - \eta \nabla_\theta \mathcal{L}.
$$
In SGD, $L_2$ regularization is mathematically equivalent to **weight decay**: scaling down the weights by a constant contraction factor $(1 - \eta \lambda)$ at every iteration before applying the gradient step.

However, Loshchilov & Hutter (2017) highlighted that this equivalence completely breaks down when applied to adaptive optimizers like Adam. If the $L_2$ penalty is added directly to the loss, the regularized gradient ${g}_t + \lambda \theta_{t-1}$ is fed into the variance tracker ${v}_t$. The resulting update step becomes:
$$
\theta_{t, i} = \theta_{t-1, i} - \frac{\eta}{\sqrt{\hat{v}_{t, i}} + \epsilon} \left( \hat{m}_{t, i} + \lambda \theta_{t-1, i} \right).
$$
This leads to an unintended distortion:
- For parameters with large historical gradients, $\hat{v}_{t, i}$ is large, which artificially suppresses the regularization penalty $\frac{\lambda \theta}{\sqrt{\hat{v}}}$. Weights that need regularization the most are penalized the least.
- Conversely, parameters with sparse or small gradients experience an amplified weight decay penalty.

#### The AdamW Solution
AdamW (*Adam with Decoupled Weight Decay*) restores true weight decay by decoupling the shrinkage step from gradient momentum tracking:
$$
\theta_t = (1 - \eta \lambda) \theta_{t-1} - \frac{\eta}{\sqrt{\hat{{v}}_t} + \epsilon}  \hat{{m}}_t.
$$
Under AdamW, weights shrink uniformly at an absolute rate proportional to their current magnitude, independently of gradient variance. This simple algebraic correction drastically improves out-of-distribution generalization in deep Transformers.

### 3) Learning Rate Scheduling: Linear Warmup and Cosine Decay

Rather than maintaining a constant learning rate $\eta$ across all optimization steps $t \in \{1, \dots, T_{\text{total}}\}$, language model pre-training dynamically modulates $\eta_t$ using a compound schedule: a linear warmup phase followed by a cosine annealing decay.

#### 1. Linear Warmup Phase ($t \le T_{\text{warmup}}$)
At initialization ($t = 0$), model weights are randomly distributed, attention patterns are uncalibrated, and the AdamW momentum accumulators are empty (${m}_0 = {0}, {v}_0 = {0}$). Early stochastic mini-batch gradients are noisy and divergent. 

Applying the peak learning rate $\eta_{\max}$ immediately causes large parameter updates that disrupt the initial weight distribution and destabilize early training. To mitigate this, the learning rate increases linearly from $0$ to $\eta_{\max}$ over the first $T_{\text{warmup}}$ steps (e.g., $T_{\text{warmup}} = 200$):
$$
\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}, \qquad \forall \, t \le T_{\text{warmup}}.
$$
This warmup period allows the moving averages ${m}_t$ and ${v}_t$ to accumulate reliable directional statistics before aggressive parameter updates begin.

#### 2. Cosine Annealing Decay ($t > T_{\text{warmup}}$)
Once peak learning rate is reached, $\eta_t$ decays along a half-period cosine curve toward zero (or a positive minimum floor $\eta_{\min} \ge 0$):
$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left( 1 + \cos\left( \pi \, \frac{t - T_{\text{warmup}}}{T_{\text{total}} - T_{\text{warmup}}} \right) \right).
$$

#### Why is decay necessary if AdamW already adapts step sizes?
While AdamW rescales updates per coordinate via $\frac{\hat{{m}}_t}{\sqrt{\hat{{v}}_t} + \epsilon}$, this ratio normalizes relative magnitudes but does not shrink the absolute step size to zero. Near convergence, stochastic gradient fluctuations around a flat minimum maintain:
$$
\frac{\hat{m}_{t, i}}{\sqrt{\hat{v}_{t, i}} + \epsilon} \approx \frac{g}{|g|} = \pm 1.
$$
Without an external decay schedule, AdamW continues taking discrete steps of scale $\eta_{\max}$, causing weights to bounce around the minimum rather than settling in it. Decaying $\eta_t$ reduces the variance of this random walk, allowing the optimizer to settle into flat, generalizable minima.

---

### 4) Gradient Norm Clipping

Because training corpora contain heterogeneous documents with occasional outliers or ill-conditioned token combinations, individual mini-batches can produce anomalously large gradients ${g}_t$. Unchecked, such gradient can push weights into non-recoverable regions of the loss surface.

To guarantee bounded update trajectories, we enforce global $L_2$-norm gradient clipping (Pascanu et al., 2013) prior to updating parameters:
$$
\|{g}\|_2 = \sqrt{\sum_{i=1}^{|\Theta|} g_i^2}.
$$
Given a maximum allowable threshold $M > 0$ (set to $M = 1.0$ in our training loop), the gradient tensor is rescaled as:
$$
{g} \leftarrow \begin{cases} 
{g} & \text{if } \|{g}\|_2 \le M, \\ 
\frac{M}{\|g\|_2} {g} & \text{if } \|{g}\|_2 > M. 
\end{cases}
$$

##### Geometric Interpretation:
Gradient clipping acts as a geometric speed limiter. It preserves the exact direction of steepest ascent ($\frac{{g}}{\|{g}\|_2}$ remains invariant) while bounding the maximum step length, ensuring stability in regions of high curvature.

---

### 5) Weight Initialization

Proper parameter initialization preserves activation variance across forward passes and gradient variance across backward passes.

For all linear layers ($W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$) and the token embedding table ($W_e \in \mathbb{R}^{|\mathcal{V}| \times d_{\text{model}}}$), weights are drawn from an uncentered Gaussian distribution scaled by the model dimension:
$$
W_{ij} \sim \mathcal{N}\left(0, \, \sigma^2\right), \qquad \sigma = \frac{1}{\sqrt{d_{\text{model}}}}.
$$
All bias vectors (where present) are initialized to zero: ${b} = {0}$, while RMSNorm scaling gains are initialized to unity: ${g} = {1}$.

#### Variance Preservation Rationale:
Consider a linear projection ${y} = W {x}$ where ${x} \in \mathbb{R}^{d_{\text{in}}}$ has independent zero-mean components with unit variance $\text{Var}(x_j) = 1$. Assuming independent initial weights $W_{ij} \sim \mathcal{N}(0, \sigma^2)$:
$$
\text{Var}(y_i) = \text{Var}\left( \sum_{j=1}^{d_{\text{in}}} W_{ij} x_j \right) = \sum_{j=1}^{d_{\text{in}}} \text{Var}(W_{ij}) \text{Var}(x_j) = d_{\text{in}} \sigma^2.
$$
Setting $\sigma = \frac{1}{\sqrt{d_{\text{in}}}}$ preserves activation variance ($\text{Var}(y_i) = 1$), preventing signals from vanishing or exploding as they traverse multiple layers.

## 5) Text Generation and Sampling

Once pre-trained, the model generates text sequentially through an autoregressive loop. Unlike training, inference generates one token at a time, appending each predicted token to the prompt context.

---

### 1) The Autoregressive Inference Loop

Given an initial prompt token sequence ${x} = (x_1, \dots, x_k)$, generation proceeds iteratively:

1. **Context Window Truncation**: If the sequence length exceeds the maximum context capacity $T_{\max}$ (e.g., $T_{\max} = 256$), the sequence is truncated to its trailing window:
   $$\tilde{{x}} = (x_{k - T_{\max} + 1}, \dots, x_k).$$
2. **Forward Evaluation**: The truncated tensor passes through the network to yield the logits matrix ${Z} \in \mathbb{R}^{T \times \vert{}\mathcal{V}\vert{}}$.
3. **Terminal Logit Extraction**: We extract the final row ${z}_T \in \mathbb{R}^{\vert{}\mathcal{V}\vert{}}$, which contains the unnormalized scores predicting the next token $x_{k+1}$.
4. **Appension and Termination**: A token $x_{k+1}$ is sampled from the derived probability distribution and appended to the context:
   $${x} \leftarrow (x_1, \dots, x_k, x_{k+1}).$$
   The process repeats until reaching "max_new_tokens" or upon emitting the termination token $x_{k+1} = \text{"<|endoftext|>"}$.

---

### 2) Temperature Scaling

Before converting the final logits ${z} \in \mathbb{R}^{\vert{}\mathcal{V}\vert{}}$ into probabilities, we divide them by a strictly positive scalar $\tau > 0$, known as the **temperature**:
$$p_i = \text{Softmax}\left( \frac{{z}}{\tau} \right)_i = \frac{\exp(z_i / \tau)}{\sum_{j=1}^{\vert{}\mathcal{V}\vert{}} \exp(z_j / \tau)}.$$
This parameter directly alters the entropy of the predicted categorical distribution:

* **Low Temperature ($\tau \to 0^+$)**: Amplifies logit discrepancies. The distribution collapses toward a Dirac delta on the argmax:
  $$\lim_{\tau \to 0^+} p_i = \begin{cases} 1 & \text{if } z_i = \max_j z_j, \\  0 & \text{otherwise}. \end{cases}$$
  The model becomes strictly deterministic, conservative, and repetitive.
* **Standard Temperature ($\tau = 1.0$)**: Preserves the exact categorical distribution learned during pre-training.
* **High Temperature ($\tau \to +\infty$)**: Flattens the distribution toward a uniform prior across all tokens:
  $$\lim_{\tau \to +\infty} p_i = \frac{1}{\vert{}\mathcal{V}\vert{}}.$$
  Sampling degenerates into random noise, breaking syntactic and grammatical coherence.

---

### 3) Decoding Strategies: Greedy vs. Multinomial Sampling

Once probabilities ${p} \in [0, 1]^{\vert{}\mathcal{V}\vert{}}$ are computed, two primary strategies determine token selection:

#### 1. Greedy Decoding
Selects the mode of the distribution:
$$x_{k+1} = \arg\max_{i \in \{0, \dots, \vert{}\mathcal{V}\vert{}-1\}} p_i.$$
While computationally straightforward, greedy decoding end more easily to degenerated loops (e.g., repeatedly generating identical tokens).

#### 2. Multinomial Categorical Sampling
Draws $x_{k+1}$ as a random variable according to the categorical distribution:
$$x_{k+1} \sim \text{Categorical}(p_1, \dots, p_{\vert{}\mathcal{V}\vert{}}).$$
Sampling introduces controlled stochasticity, allowing plausible lower-probability tokens to be explored, which prevents repetitive loops and produces more natural text completions.


## 6) Conclusion

Congratulations ! You have been through a journey in the world of generative AI. Thanks a lot for reading this text and I hope you understood it. Sorry for those who expected more, but I couldn't go deeper as I am still almost a begginner. I hope it will be useful for you. The next step is to try to reverse engineer (as I did thanks to open-source LLMs but mostly with Karpathy's videos) the code written. Try to build your own GPT now, and you should go further with fine-tuning to create your own assistant !
