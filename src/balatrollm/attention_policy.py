import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomAttentionExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that uses Commutative Graph Self-Attention over the cards in hand
    and processes each card's profile (Global, Relational, Self) via a shared MLP.
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        card_space = observation_space.spaces["cards"]
        self.num_cards = card_space.shape[0]
        
        self.shared_mlp_dim = 16
        # The output feature dimension must exactly match the flattened output!
        actual_features_dim = self.num_cards * self.shared_mlp_dim
        
        super().__init__(observation_space, actual_features_dim)
        
        self.card_feat_dim = card_space.shape[1] # 5
        
        self.d_model = 32
        self.card_embedding = nn.Linear(self.card_feat_dim, self.d_model)
        
        # Q and K linear layers to build the synergy matrix
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        
        self.stats_dim = observation_space.spaces["stats"].shape[0] # 2
        
        # Profile is: [Global (d_model + stats_dim), Relational (d_model), Self (d_model)]
        self.profile_dim = (self.d_model + self.stats_dim) + self.d_model + self.d_model
        
        # Shared MLP for each card
        self.shared_mlp = nn.Sequential(
            nn.Linear(self.profile_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.shared_mlp_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> torch.Tensor:
        cards = observations["cards"] # [batch_size, num_cards, 5]
        stats = observations["stats"] # [batch_size, 2]
        
        # 1. Embed cards -> E
        E = self.card_embedding(cards) # [batch_size, num_cards, d_model]
        
        # 2. Synergy Matrix S = A + A^T
        Q = self.W_q(E)
        K = self.W_k(E)
        A = torch.bmm(Q, K.transpose(1, 2)) / (self.d_model ** 0.5)
        S = A + A.transpose(1, 2) # Commutative relationship matrix
        
        # Normalize S via softmax so Relational Info isn't unbounded
        S = torch.softmax(S, dim=-1)
        
        # 3. Relational Info R_i = sum_j S_{i,j} E_j
        R = torch.bmm(S, E) # [batch_size, num_cards, d_model]
        
        # 4. Global Game Plan g = mean(E)
        g = E.mean(dim=1) # [batch_size, d_model]
        
        # 5. Build profiles
        # Append stats to game plan
        g_with_stats = torch.cat([g, stats], dim=1) # [batch_size, d_model + 2]
        
        # Expand g to match sequence length
        g_expanded = g_with_stats.unsqueeze(1).expand(-1, self.num_cards, -1)
        
        # Profile P_i = [g, R_i, E_i]
        P = torch.cat([g_expanded, R, E], dim=2) # [batch_size, num_cards, profile_dim]
        
        # 6. Apply shared MLP across cards
        card_features = self.shared_mlp(P) # [batch_size, num_cards, shared_mlp_dim]
        
        # 7. Flatten. Do NOT project via a mixing linear layer. We want to preserve card isolation!
        # Features dim will be num_cards * shared_mlp_dim
        features = torch.flatten(card_features, start_dim=1)
        
        return features

class SharedActionNet(nn.Module):
    def __init__(self, latent_dim, num_cards, num_classes=2):
        super().__init__()
        self.num_cards = num_cards
        self.card_feat_dim = latent_dim // num_cards
        self.shared_linear = nn.Linear(self.card_feat_dim, num_classes)
        # Global choice: Play vs Discard
        self.global_linear = nn.Linear(latent_dim, 2)
        
    def forward(self, x):
        # x: [batch_size, latent_dim]
        # Reshape to [batch_size, num_cards, card_feat_dim]
        x_cards = x.view(x.size(0), self.num_cards, self.card_feat_dim)
        card_logits = self.shared_linear(x_cards) # [batch_size, num_cards, 2]
        card_logits = card_logits.view(x.size(0), -1) # [batch_size, num_cards * 2]
        
        global_logits = self.global_linear(x) # [batch_size, 2]
        
        return torch.cat([global_logits, card_logits], dim=1)

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

class CustomSharedMLPPolicy(MaskableActorCriticPolicy):
    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # Override action_net with our SharedActionNet
        # Since we use net_arch=[], latent_dim_pi == features_dim
        latent_dim = self.mlp_extractor.latent_dim_pi
        
        # We know num_cards is MAX_HAND_SIZE (e.g. 20)
        # The CustomAttentionExtractor outputs num_cards * shared_mlp_dim
        # Let's dynamically infer num_cards from the environment observation space
        num_cards = self.observation_space.spaces["cards"].shape[0]
        
        self.action_net = SharedActionNet(latent_dim, num_cards, num_classes=2)

