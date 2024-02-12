import torch
import torch.nn as nn
import torch.nn.functional as F


class Embed(nn.Module):
    def __init__(self, input_dim, output_dim, normalize_input=False, event_level=False, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.fc3 = nn.Linear(output_dim, output_dim)
        self.event_level = event_level
        

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            if not self.event_level: 
                x = x.permute(2, 0, 1).contiguous()

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x
    

class AttBlock(nn.Module):
    def __init__(self, embed_dims, linear_dims1, linear_dims2, num_heads=8, activation='relu'):
        super(AttBlock, self).__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dims)
        self.multihead_attention = nn.MultiheadAttention(embed_dims, num_heads)
        self.layer_norm2 = nn.LayerNorm(embed_dims)
        self.linear1 = nn.Linear(embed_dims, linear_dims1)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.layer_norm3 = nn.LayerNorm(linear_dims1)
        self.linear2 = nn.Linear(linear_dims1, linear_dims2)

    def forward(self, x, padding_mask=None):
        # Layer normalization 1
        x = self.layer_norm1(x)

        if padding_mask is not None:
        # Assuming mask is 0 for non-padded and 1 for padded elements,
        # convert it to a boolean tensor with `True` for padded locations.
            padding_mask = padding_mask.bool()

        
        x_att, attention = self.multihead_attention(x, x, x, key_padding_mask=padding_mask, need_weights=True, average_attn_weights=True)
        
        # Skip connection
        x = x + x_att # Skip connection
        # Layer normalization 2
        x = self.layer_norm2(x)
        # Linear layer and activation
        x_linear1 = self.activation(self.linear1(x))
        # Skip connection for the first linear layer
        x = x + x_linear1
        # Layer normalization 3
        x = self.layer_norm3(x_linear1)
        # Linear layer with specified output dimensions
        x_linear2 = self.linear2(x)
        # Skip connection for the second linear layer
        x = x + x_linear2
        return x, attention

class ClassBlock(nn.Module):
    def __init__(self, embed_dims, linear_dims1, linear_dims2, num_heads=8, activation='relu'):
        super(ClassBlock, self).__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dims)
        self.multihead_attention = nn.MultiheadAttention(embed_dims, num_heads)
        self.layer_norm2 = nn.LayerNorm(embed_dims)
        self.linear1 = nn.Linear(embed_dims, linear_dims1)
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        self.layer_norm3 = nn.LayerNorm(linear_dims1)
        self.linear2 = nn.Linear(linear_dims1, linear_dims2)

    def forward(self, x, class_token, padding_mask=None):
        # Concatenate the class token to the input sequence along the sequence length dimension  
        x = torch.cat((class_token, x), dim=0)  # (seq_len+1, batch, embed_dim)
        # Layer normalization 1
        x = self.layer_norm1(x)

        # Multihead Attention
        if padding_mask is not None:
            # Ensure mask has the correct shape for attention
            padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            padding_mask = padding_mask.bool()
        
        
        x_att, attention = self.multihead_attention(class_token, x, x, key_padding_mask=padding_mask, need_weights=True, average_attn_weights=False)
        # Layer normalization 2
        x = self.layer_norm2(x_att)
        x = class_token + x  # Skip connection
        # Linear layer and activation
        x_linear1 = self.activation(self.linear1(x))
        # Layer normalization 3
        x_linear1 = self.layer_norm3(x_linear1)
        # Linear layer with specified output dimensions
        x_linear2 = self.linear2(x_linear1 )
        # Skip connection for the second linear layer
        x = x + x_linear2
        return x, attention

class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
class AnalysisObjectTransformer(nn.Module):
    def __init__(self, input_dim_obj, input_dim_event, embed_dims, linear_dims1, linear_dims2, mlp_hidden_1, mlp_hidden_2, num_heads=8):
        super(AnalysisObjectTransformer, self).__init__()

        self.embed_dims = embed_dims
       
        # Embedding layer (assumed to be external)
        self.embedding_layer = Embed(input_dim_obj, embed_dims)
        self.embedding_layer_event_level = Embed(input_dim_event, embed_dims, event_level=True)

        # Three blocks of self-attention
        self.block1 = AttBlock(embed_dims, linear_dims1, linear_dims1, num_heads)
        self.block2 = AttBlock(linear_dims1, linear_dims1, linear_dims1, num_heads)
        self.block3 = AttBlock(linear_dims1, linear_dims2, linear_dims2, num_heads)
        self.block5 = ClassBlock(linear_dims2, linear_dims1, linear_dims2, num_heads)
        self.block6 = ClassBlock(linear_dims2, linear_dims1, linear_dims2, num_heads)
        self.block7 = ClassBlock(linear_dims2, linear_dims1, linear_dims2, num_heads)

        # Output linear layer and sigmoid activation

        self.mlp = MLPHead(embed_dims + input_dim_event, mlp_hidden_1, mlp_hidden_2, output_dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, event_level, mask=None):

        cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims), requires_grad=True)
        cls_token = nn.init.trunc_normal_(cls_token, std=.02)
        # Embedding layer

        x = self.embedding_layer(x)
        x = x.permute(1, 0, 2)
        
        attention_weights = []

        # Three blocks of self-attention
        x, attention = self.block1(x, padding_mask=mask)

        attention_weights.append(attention)
        x, attention  = self.block2(x, padding_mask=mask)
        attention_weights.append(attention)
        x, attention = self.block3(x, padding_mask=mask)
        attention_weights.append(attention)

        cls_tokens  = cls_token.expand(1, x.size(1), -1)  # (1, N, C)
        cls_tokens, attention  = self.block5(x, cls_tokens, padding_mask=mask)
        cls_tokens, attention  = self.block6(x, cls_tokens, padding_mask=mask)
        cls_tokens, attention  = self.block7(x, cls_tokens, padding_mask=mask)

        x = torch.cat((cls_tokens.squeeze(0), event_level), dim=-1)
        x = self.mlp(x)
        output_probabilities = self.sigmoid(x)
        return output_probabilities, attention_weights

