import numpy as np

class SDPA:
    # def __init__(self, Q, K, V):
    #     self.Q = Q
    #     self.K = K
    #     self.V = V

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.shape[-1] # Get the dimension of Q
        print("d_k:", d_k)
        # Prevent gradient vanishing caused by excessive dot product values
        scores = np.dot(Q, K.T) / np.sqrt(d_k) # similarity score matrix divide by the square root of the dimension of the key vector
        print("scores:", scores)
        if mask is not None:
            scores += (mask * -1e9)

        attention_weights = self.softmax(scores)
        output = np.dot(attention_weights, V) # Weighted sum value V with attention weights
        return output, attention_weights
    
if __name__ == "__main__":

    np.random.seed(42)
    Q = np.random.rand(2, 4)
    K = np.random.rand(2, 4)
    V = np.random.rand(2, 4)

    scaled_dot_product = SDPA()
    output, attention_weights = scaled_dot_product.scaled_dot_product_attention(Q, K, V)

    print("Attention output:", output)
    print("Attention weight:", attention_weights)