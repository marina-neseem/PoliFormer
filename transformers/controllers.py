import torch
from .gumbel import gumbel_sigmoid

def get_random_policy(policy, ratio):
    random_p = torch.empty_like(policy).fill_(ratio).bernoulli() + policy * 0.0  # add policy * 0.0 into the loop of loss calculation to avoid the DDP issue
    return random_p

class TokensSelect(torch.nn.Module):   
    def __init__(self, num_embeddings, num_patches, random=False, random_ratio=1.):
        super(TokensSelect, self).__init__()
        self.random = random
        self.random_ratio = random_ratio

        self.num_embeddings = num_embeddings
        self.num_patches = num_patches
        self.mlp = torch.nn.Linear(num_embeddings, 100)
        self.fc = torch.nn.Linear(100*num_patches, num_patches)

    def unfreeze_controllers(self):
        # Allow training of the token select
        for layer in[self.mlp, self.fc]:
            for p in layer.parameters():
                p.requires_grad = True

    # Defining the forward pass    
    def forward(self, x):
        b_sz = x.shape[0]
        x = self.mlp(x)
        x = x.view(b_sz, -1)
        x = self.fc(x)

        tokens_mask = gumbel_sigmoid(x, hard=True, tau=5)
        if self.random:
            tokens_mask = get_random_policy(tokens_mask, self.random_ratio)
    
        return tokens_mask.bool().unsqueeze(-1), tokens_mask

class BlockSelect(torch.nn.Module):   
    def __init__(self, num_tokens, embedding_size, num_layers, random=False, random_ratio=1.):
        super(BlockSelect, self).__init__()
        print(num_tokens, embedding_size, num_layers)
        self.random = random
        self.random_ratio = random_ratio
        self.mlp = torch.nn.Linear(embedding_size, 100)
        self.fc = torch.nn.Linear(100*num_tokens, num_layers)

    def unfreeze_controllers(self):
        # Allow training of the block select
        for layer in[self.mlp, self.fc]:
            for p in layer.parameters():
                p.requires_grad = True

    # Defining the forward pass    
    # Experiments shows that during pretraining controllers hard is better false
    # while during simultaniously training policy network and model, it is better to be true
    def forward(self, x, hard_gumbel=True):
        b_sz = x.shape[0]
        temp = self.mlp(x)
        temp = temp.view(b_sz, -1)
        temp = self.fc(temp)

        block_mask = gumbel_sigmoid(temp, hard=hard_gumbel, tau=5)
        
        if self.random:
            block_mask = get_random_policy(block_mask, self.random_ratio)
    
        return block_mask
