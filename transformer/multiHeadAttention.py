
import numpy as np

class SDPA:
    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.shape[-1] # Get the dimension of Q
        print("d_k:", d_k)
        # Prevent gradient vanishing caused by excessive dot product values
        # K.transpose(0, 2, 1) only should be transposed into the last two dimensions not the whole matrix
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k) # ValueError: shapes (2,5,4) and (4,5,2) not aligned: 4 (dim 2) != 5 (dim 1), dot fix matmul
        print("scores:", scores)
        if mask is not None:
            scores += (mask * -1e9)

        attention_weights = self.softmax(scores)
        output = np.matmul(attention_weights, V) # Weighted sum value V with attention weights
        return output, attention_weights

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.WQ = np.random.rand(d_model, d_model)
        self.WK = np.random.rand(d_model, d_model)
        self.WV = np.random.rand(d_model, d_model)
        self.WO = np.random.rand(d_model, d_model)

    # Divide an input tensor(typically a query, key, or value) into multiple heads for parallel computation in a multi-head attention mechanism
    # (batch_size, seq_len, depth) is the shape of x
    # batch_size: 
    # seq_len : Sequence length, "-2" means that Numpy automatically calculates the size if this dimension, alse seq_len value invariability
    # d_model : feature dimension (hidden dimension)
    def split_heads(self, x, batch_size): # x is input tensor
        # Separate the last dimension to (num_heads, depth)
        # Divide x into num_heads, and the feature dimension of each head becomes depth = d_model // num_heads
        x = x.reshape(batch_size, -1, self.num_heads, self.depth) # transform to (..., seq_len, num_heads, ...)
        # the transpose operation rearranges the dimensional order of the tensor
        return np.transpose(x, (0, 2, 1, 3)) # (batch_size, num_heads, seq_len, depth):  (0, 1, 2, 3) to (0, 2, 1, 3)

    def forward(self, Q, K, V, mask=None):
        batch_size  = Q.shape[0]

        Q = np.dot(Q, self.WQ)
        K = np.dot(K, self.WK)
        V = np.dot(V, self.WV)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        Q = Q.reshape(batch_size * self.num_heads, -1, self.depth)
        K = K.reshape(batch_size * self.num_heads, -1, self.depth)
        V = V.reshape(batch_size * self.num_heads, -1, self.depth)

        # attention_outputs = []
        # attention_weights = []

        # for i in range(self.num_heads):
        #     output, weights = SDPA().scaled_dot_product_attention(Q[:,  i], K[:, i], V[:, i], mask)
        #     attention_outputs.append(output)
        #     attention_weights.append(weights)

        attention_output, attention_weights = SDPA().scaled_dot_product_attention(Q, K, V, mask)

        # Reshape attention output back to (batch_size, seq_len,  d_model)
        attention_output = attention_output.reshape(batch_size, self.num_heads, -1, self.depth)
        attention_output = np.transpose(attention_output, (0, 2, 1, 3))
        concat_attention = attention_output.reshape(batch_size, -1, self.d_model)

        # Spliced multiple output
        # concat_attention = np.concatenate(attention_outputs, axis=-1)

        output = np.dot(concat_attention, self.WO)

        return output, attention_weights
    
batch_size = 2
seq_len = 5
d_model = 8
num_heads = 2

np.random.seed(42)
Q = np.random.rand(batch_size, seq_len, d_model)
K = np.random.rand(batch_size, seq_len, d_model)
V = np.random.rand(batch_size, seq_len, d_model)

mha = MultiHeadAttention(d_model, num_heads)
output, attention_weights = mha.forward(Q, K, V)

print("Muti-Head Attention Output :", output)
print("Attention Weights : ", attention_weights)