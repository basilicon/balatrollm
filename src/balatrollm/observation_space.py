import numpy as np
import math

MAX_HAND_SIZE = 12
MAX_JOKERS = 8

RANKS = {'2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 'T': 9, 'J': 10, 'Q': 11, 'K': 12, 'A': 13}
SUITS = {'Spades': 1, 'Hearts': 2, 'Clubs': 3, 'Diamonds': 4}

def parse_gamestate_to_obs(gamestate: dict) -> np.ndarray:
    """Converts the raw JSON gamestate into a fixed-size float32 array."""
    obs = []
    
    # 1. Parse Hand Cards (MAX_HAND_SIZE x 5)
    parsed_cards = gamestate.get("hand", {}).get("cards", [])
    for i in range(MAX_HAND_SIZE):
        if i < len(parsed_cards):
            c = parsed_cards[i]
            hidden = 1.0 if c.get("state", {}).get("hidden", False) else 0.0
            debuff = 1.0 if c.get("state", {}).get("debuff", False) else 0.0
            rank = float(RANKS.get(str(c.get("value", {}).get("rank", "")), 0))
            suit = float(SUITS.get(str(c.get("value", {}).get("suit", "")), 0))
            chips = float(c.get("value", {}).get("chips", 0)) / 10.0
            obs.extend([rank, suit, debuff, hidden, chips])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
            
    # 2. Parse Jokers (MAX_JOKERS x 2)
    jokers = gamestate.get("jokers", {}).get("cards", [])
    for i in range(MAX_JOKERS):
        if i < len(jokers):
            j = jokers[i]
            key_hash = float(hash(j.get("key", "")) % 200) / 200.0
            sell_cost = float(j.get("cost", {}).get("sell", 0)) / 10.0
            obs.extend([key_hash, sell_cost])
        else:
            obs.extend([0.0, 0.0])
            
    # 3. Parse Round Stats (5)
    r = gamestate.get("round", {})
    b = gamestate.get("blind", {})
    
    hands_left = float(r.get("hands_left", 0))
    discards_left = float(r.get("discards_left", 0))
    chips = float(r.get("chips", 0))
    target = float(b.get("chips", 1))
    
    chips_norm = min(chips / target, 1.0) if target > 0 else 0.0
    target_log = math.log10(target) if target > 0 else 0.0
    dollars = float(gamestate.get("dollars", 0)) / 100.0
    
    obs.extend([hands_left, discards_left, chips_norm, target_log, dollars])
    
    return np.array(obs, dtype=np.float32)

if __name__ == "__main__":
    dummy_gamestate = {
        "hand": {"cards": [{"value": {"rank": "A", "suit": "Spades", "chips": 11}}]},
        "round": {"hands_left": 4, "discards_left": 3, "chips": 150},
        "blind": {"chips": 300},
        "dollars": 5
    }
    obs = parse_gamestate_to_obs(dummy_gamestate)
    print(f"Observation shape: {obs.shape}")
    print(obs)
