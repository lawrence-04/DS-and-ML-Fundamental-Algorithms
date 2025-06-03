# Transformer
The transformer is one of the most powerful architectures in deep learning and is found in almost all state-of-the-art models in the current ML era.

The wiki is [here](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)).

## Theory
Fundamentally, the transformer takes in a sequence and maps it to an output. We can also have a transformer as a decoder, where the output is included in the input, and the model is used recurrently. This is how popular chatbot models like ChatGPT work. One of the aspects that makes the transformer especially powerful is that the decoder can be trained in parallel, unlike similar methods like RNNs.

The architecture of the transformer is based on the following.
### Self-Attention
Self-Attention is a way for a model to focus on different parts of the input when producing an output.

For each vector in the sequence, we project the vector into a key, value and query using learned matrices $W_K$, $W_V$, and $W_Q$ respectively, producing $K = XW_K$, $V = XW_V$, and $Q = XW_Q$. Then, the formula is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V$$

The query represents what the vector is looking for, the key represents what the vector has to offer, and the value holds the content. We are computing the cosine similarity between the keys and queries, then use that similarity to scale the values.

The $\sqrt{d_k}$ prevents $QK^T$ from becoming too large, preventing the softmax from becoming spiky and stabilising the gradient.

### Multi-headed Self-Attention
The above formula is for a single head. In the transformer, many heads are used in the same layer. The outputs from each head are concatenated and projected through a linear layer.

### Positional Encoding
Since the transformer has no recurrence or convolution, it needs some way to incorporate the order of the sequence. This is done using positional encodings, which are added to the input embeddings. One popular method uses sine and cosine functions of different frequencies:

$$\text{PE}(\text{pos}, 2i)   = \text{sin}(\frac{\text{pos}}{10000^{(2i / d_\text{model})}})$$
$$\text{PE}(\text{pos}, 2i+1) = \text{cos}(\frac{\text{pos}}{10000^{(2i / d_\text{model})}})$$

So the positional encoding is a vector, where $i$ is the value of each dimension, and $\text{pos}$ is the index of the vector in the sequence.

This encoding is added element-wise at the beginning of the model, and allow the model to learn relationships between positions.

### Encoder-Decoder Architecture
The original transformer (as introduced in "Attention Is All You Need") is composed of two parts:
* Encoder: A stack of layers each containing Multi-Headed Self-Attention followed by a feedforward network.
* Decoder: Similar layers, but with an additional attention mechanism that attends over the encoder outputs.

For this repository, we will implement the encoder for simplicity.
