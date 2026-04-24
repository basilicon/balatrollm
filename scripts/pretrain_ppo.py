import os
import copy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
import gymnasium as gym

from balatrollm.balatro_sim_env import BalatroSimEnv
from balatrollm.attention_policy import CustomAttentionExtractor
from balatrollm.deterministic_player import get_hand_type

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['Spades', 'Hearts', 'Clubs', 'Diamonds']

def get_chip_value(rank_str):
    if rank_str in ['T', 'J', 'Q', 'K']:
        return 10
    elif rank_str == 'A':
        return 11
    else:
        return int(rank_str)

def generate_standard_deck():
    cards = []
    for suit in SUITS:
        for rank in RANKS:
            cards.append({
                "value": {
                    "rank": rank,
                    "suit": suit,
                    "chips": get_chip_value(rank)
                },
                "state": {
                    "hidden": False,
                    "debuff": False
                },
                "modifier": {}
            })
    return cards

def default_evaluator(cards, hand_type, hands_info):
    chips = hands_info[hand_type].get('chips', 0)
    mult = hands_info[hand_type].get('mult', 1)
    for c in cards:
        if not c.get('debuff'):
            chips += c.get('chip_value', 0)
    return chips * mult

def mask_fn(env: gym.Env):
    return env.unwrapped.action_masks()

def main():
    print("Generating standard 52-card deck gamestate...")
    deck = generate_standard_deck()
    
    import random
    random.shuffle(deck)
    initial_hand = deck[:8]
    deck = deck[8:]
    
    # Empty default hands_info for standard poker hands
    hands_info = {
        'High Card': {'chips': 5, 'mult': 1},
        'Pair': {'chips': 10, 'mult': 2},
        'Two Pair': {'chips': 20, 'mult': 2},
        'Three of a Kind': {'chips': 30, 'mult': 3},
        'Straight': {'chips': 30, 'mult': 4},
        'Flush': {'chips': 35, 'mult': 4},
        'Full House': {'chips': 40, 'mult': 4},
        'Four of a Kind': {'chips': 60, 'mult': 7},
        'Straight Flush': {'chips': 100, 'mult': 8},
        'Royal Flush': {'chips': 100, 'mult': 8}
    }
    
    initial_gamestate = {
        "cards": {"cards": deck},
        "hand": {"cards": initial_hand},
        "jokers": {"cards": []},
        "round": {
            "hands_left": 4,
            "discards_left": 3,
            "chips": 0,
            "reroll_cost": 5
        },
        "blind": {
            "chips": 300
        }
    }
    
    print("Initializing BalatroSimEnv...")
    sim_env = BalatroSimEnv(initial_gamestate, default_evaluator, hands_info)
    sim_env = ActionMasker(sim_env, mask_fn)
    
    policy_kwargs = dict(
        features_extractor_class=CustomAttentionExtractor,
        features_extractor_kwargs=dict(),
        net_arch=[]
    )
    
    print("Initializing Foundation MaskablePPO...")
    from balatrollm.attention_policy import CustomSharedMLPPolicy
    model = MaskablePPO(
        CustomSharedMLPPolicy,
        sim_env,
        learning_rate=0.0005,
        n_steps=128,
        batch_size=32,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    print("Pretraining model for 10,000 timesteps...")
    model.learn(total_timesteps=10000)
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/pretrained_ppo.zip"
    model.save(model_path)
    print(f"Pretrained model successfully saved to {model_path}!")

if __name__ == "__main__":
    main()
