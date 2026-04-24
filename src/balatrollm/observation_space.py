import numpy as np
import gymnasium as gym

MAX_HAND_SIZE = 20
CARD_EMBEDDING_SIZE = 5

RANKS = {'2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 'T': 9, 'J': 10, 'Q': 11, 'K': 12, 'A': 13}
SUITS = {'Spades': 1, 'Hearts': 2, 'Clubs': 3, 'Diamonds': 4}

def get_observation_space() -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "cards": gym.spaces.Box(low=0.0, high=100.0, shape=(MAX_HAND_SIZE, CARD_EMBEDDING_SIZE), dtype=np.float32),
        "stats": gym.spaces.Box(low=0.0, high=100.0, shape=(2,), dtype=np.float32)
    })

def parse_gamestate_to_obs(gamestate: dict) -> dict:
    """Converts the raw JSON gamestate into a dictionary of numpy arrays."""
    cards_matrix = np.zeros((MAX_HAND_SIZE, CARD_EMBEDDING_SIZE), dtype=np.float32)
    
    # 1. Parse Hand Cards
    parsed_cards = gamestate.get("hand", {}).get("cards", [])
    for i in range(MAX_HAND_SIZE):
        if i < len(parsed_cards):
            c = parsed_cards[i]
            if not isinstance(c, dict):
                continue
            state = c.get("state", {})
            if not isinstance(state, dict):
                state = {}
            value = c.get("value", {})
            if not isinstance(value, dict):
                value = {}
                
            hidden = 1.0 if state.get("hidden", False) else 0.0
            debuff = 1.0 if state.get("debuff", False) else 0.0
            rank = float(RANKS.get(str(value.get("rank", "")), 0))
            suit = float(SUITS.get(str(value.get("suit", "")), 0))
            chips = float(value.get("chips", 0)) / 10.0
            cards_matrix[i] = [rank, suit, debuff, hidden, chips]
            
    # 2. Parse Stats
    r = gamestate.get("round", {})
    hands_left = float(r.get("hands_left", 0))
    discards_left = float(r.get("discards_left", 0))
    
    stats_array = np.array([hands_left, discards_left], dtype=np.float32)
    
    return {
        "cards": cards_matrix,
        "stats": stats_array
    }

if __name__ == "__main__":
    dummy_gamestate = {
        "hand": {"cards": [{"value": {"rank": "A", "suit": "Spades", "chips": 11}}]},
        "round": {"hands_left": 4, "discards_left": 3}
    }
    obs = parse_gamestate_to_obs(dummy_gamestate)
    print("Cards shape:", obs["cards"].shape)
    print("Stats shape:", obs["stats"].shape)
