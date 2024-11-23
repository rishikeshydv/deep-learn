import torch
import torch.nn as nn
import torch.optim as optim

#datasets 
src_vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "How": 3, "are": 4, "you": 5, "?": 6}
tgt_vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "Comment": 3, "allez-vous": 4, "?": 5}


src_sentence = ["How", "are", "you", "?"]
tgt_sentence = ["Comment", "allez-vous", "?"]

src_indices = [src_vocab[word] for word in src_sentence]
tgt_indices = [tgt_vocab[word] for word in tgt_sentence]

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
embedding_dim = 64

src_embedding = nn.Embedding(src_vocab_size, embedding_dim)
tgt_embedding = nn.Embedding(tgt_vocab_size, embedding_dim)


# Convert indices to tensors
src_indices_tensor = torch.tensor([src_indices], dtype=torch.long)  # Shape: (batch_size, seq_length)
tgt_indices_tensor = torch.tensor([tgt_indices], dtype=torch.long)  # Shape: (batch_size, seq_length)

# Generate masks (1 for valid tokens, 0 for padding)
src_mask = (src_indices_tensor != src_vocab["<pad>"])
tgt_mask = (tgt_indices_tensor != tgt_vocab["<pad>"])

# Get embeddings
src_embeddings = src_embedding(src_indices_tensor)  # Shape: (batch_size, seq_length, embedding_dim)
tgt_embeddings = tgt_embedding(tgt_indices_tensor)  # Shape: (batch_size, seq_length, embedding_dim)

Q_ = src_embeddings  # Source sentence embeddings
K_ = tgt_embeddings  # Target sentence embeddings (keys)
V_ = tgt_embeddings  # Target sentence embeddings (values)

if K_.shape[1] < Q_.shape[1]:
    pad_length = Q_.shape[1] - K_.shape[1]
    K_ = torch.nn.functional.pad(K_, (0, 0, 0, pad_length))  # Pad along the sequence dimension
    V_ = torch.nn.functional.pad(V_, (0, 0, 0, pad_length))  # Pad along the sequence dimension


model_dimension = 64
n_heads = 8
dimension = model_dimension // n_heads
seq_length = K_.shape[1]

        # Linear layers for Q, K, V, and output
q = nn.Linear(model_dimension, model_dimension)
k = nn.Linear(model_dimension, model_dimension)
v = nn.Linear(model_dimension, model_dimension)
o = nn.Linear(model_dimension, model_dimension)

        # Define the policy network (for selecting keys based on queries)
policy_network = nn.Sequential(
            nn.Linear(dimension, 64),  # Input is the Query dimension
            nn.ReLU(),
            nn.Linear(64,seq_length),  # Output is the action (selection of relevant keys)
            nn.Softmax(dim=-1)  # Probability distribution over keys
        )


def forward( Q, K, V, mask=None):
    # Split the input tensors into multi-head dimensions
    print("Test Point 1 Passed")
    Q = split_heads(q(Q))
    K = split_heads(k(K))
    V = split_heads(v(V))
    print("Test Point 2 Passed")
    # Calculate attention weights using RL policy
    attention_weights = select_attention_weights(Q, K, mask)
    print("Test Point 3 Passed")
    print(attention_weights.shape)
    print(V.shape)

    output = torch.matmul(attention_weights, V)
    print("Test Point 4 Passed")
    combined = combine_heads(output)
    print("Test Point 5 Passed")
    return o(combined)
        
def split_heads(x):
    batch_size, seq_length, model_dim = x.size()
    x = x.view(batch_size, seq_length, n_heads, dimension)
    return x.permute(0, 2, 1, 3)  # (batch_size, n_heads, seq_length, dimension)

def combine_heads(x):
    batch_size, n_heads, seq_length, dimension = x.size()
    x = x.permute(0, 2, 1, 3).contiguous()
    return x.view(batch_size, seq_length, n_heads * dimension)

def select_attention_weights(Q, K, mask=None):
    # Assume Q and K are (batch_size, n_heads, seq_length, dimension)
    
    batch_size, n_heads, seq_length_query, dimension = Q.size()
    _, _, seq_length_key, _ = K.size()

    # Flatten the queries for policy input
   
    Q_flat = Q.reshape(batch_size * n_heads * seq_length_query, dimension)
    
    # Use the policy network to decide the attention weights for each query
    
    action_probs = policy_network(Q_flat)
    
    # Reshape action_probs back to attention shape
    
    action_probs = action_probs.view(batch_size, n_heads, seq_length_query, seq_length_key)
    
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(2)  # Adjust mask shape for multi-head attention
        action_probs = action_probs.masked_fill(mask == 0, float('-inf'))
    
    attention_weights = torch.softmax(action_probs, dim=-1)

    return attention_weights


output = forward(Q_, K_, V_, src_mask)


