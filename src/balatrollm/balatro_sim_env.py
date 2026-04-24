import gymnasium as gym
import numpy as np
import math
import random
import copy

from .observation_space import get_observation_space, parse_gamestate_to_obs, MAX_HAND_SIZE
from .deterministic_player import get_hand_type, parse_card

class BalatroSimEnv(gym.Env):
    """A purely simulated local environment for training PPO."""
    
    def __init__(self, initial_gamestate: dict, evaluator, hands_info: dict):
        super().__init__()
        self.initial_gamestate = copy.deepcopy(initial_gamestate)
        self.evaluator = evaluator
        self.hands_info = hands_info
        
        # Action space: 1 for Action Type (Play/Discard), 20 for Card Selectors
        self.action_space = gym.spaces.MultiDiscrete([2] * (MAX_HAND_SIZE + 1))
        self.observation_space = get_observation_space()
        
        self.current_gamestate = None
        self.deck_cards = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            
        self.current_gamestate = copy.deepcopy(self.initial_gamestate)
        deck_cards_raw = self.current_gamestate.get("cards", {})
        if isinstance(deck_cards_raw, dict):
            self.deck_cards = deck_cards_raw.get("cards", [])
        else:
            self.deck_cards = []
        # Shuffle the deck to prevent overfitting to a single unknown deck order
        random.shuffle(self.deck_cards)
        
        return parse_gamestate_to_obs(self.current_gamestate), {}

    def action_masks(self) -> np.ndarray:
        # MultiDiscrete mask is a flat 1D boolean array of size sum(nvec) = 42
        mask = np.zeros(sum(self.action_space.nvec), dtype=np.bool_)
        
        hands_left = self.current_gamestate.get("round", {}).get("hands_left", 0)
        discards_left = self.current_gamestate.get("round", {}).get("discards_left", 0)
        
        # Play (0) vs Discard (1)
        if hands_left > 0: mask[0] = True
        if discards_left > 0: mask[1] = True
        
        hand_cards = self.current_gamestate.get("hand", {})
        if not isinstance(hand_cards, dict):
            hand_cards = {}
        hand_cards = hand_cards.get("cards", [])
        num_cards = 0
        for c in hand_cards:
            if isinstance(c, dict):
                state = c.get("state", {})
                if not isinstance(state, dict):
                    state = {}
                if not state.get("hidden", False):
                    num_cards += 1
        
        # For each card slot, index in the flat mask is 2 + 2*i and 3 + 2*i
        offset = 2
        for i in range(MAX_HAND_SIZE):
            # Index offset + 0: Keep (0). ALWAYS valid (you don't have to select a card)
            mask[offset] = True
            
            # Index offset + 1: Select (1). ONLY valid if the card actually exists
            if i < num_cards:
                mask[offset + 1] = True
            else:
                mask[offset + 1] = False
                
            offset += 2
            
        return mask

    def step(self, action):
        # action is a 1D array of length 21: [act_type, c0, c1, ..., c19]
        act_type_idx = action[0]
        act_type = "play" if act_type_idx == 0 else "discard"
        
        # Gather all selected card indices
        indices = [i for i, val in enumerate(action[1:]) if val == 1]
        
        # Fallback: limit to max 5 cards
        if len(indices) > 5:
            indices = indices[:5]
            
        hand_cards = self.current_gamestate.get("hand", {}).get("cards", [])
        
        # Fallback: if 0 cards selected, force select the first valid card to avoid softlock
        if len(indices) == 0 and len(hand_cards) > 0:
            indices = [0]
            
        selected_cards = [hand_cards[i] for i in indices if i < len(hand_cards)]
        
        reward = 0.0
        
        if act_type == "play":
            # Evaluate hand
            parsed_selected_cards = [parse_card(c) for c in selected_cards]
            htype = get_hand_type(parsed_selected_cards)
            try:
                score = self.evaluator(parsed_selected_cards, htype, self.hands_info)
            except Exception:
                score = 0.0
            
            blind_target = self.current_gamestate.get("blind", {}).get("chips", 300)
            if blind_target <= 0:
                blind_target = 300
                
            # Track accumulated chips
            current_chips = self.current_gamestate.get("round", {}).get("chips", 0)
            self.current_gamestate["round"]["chips"] = current_chips + score
            
            # Reward is proportional to how much of the blind we defeated
            reward = score / blind_target
            
            self.current_gamestate["round"]["hands_left"] -= 1
        elif act_type == "discard":
            reward = 0.0
            self.current_gamestate["round"]["discards_left"] -= 1

        # Remove cards from hand
        new_hand = [c for i, c in enumerate(hand_cards) if i not in indices]
        
        # Draw new cards
        hand_size_limit = 8 # Typical hand size
        cards_needed = max(0, hand_size_limit - len(new_hand))
        drawn_cards = self.deck_cards[:cards_needed]
        self.deck_cards = self.deck_cards[cards_needed:]
        new_hand.extend(drawn_cards)
        
        self.current_gamestate["hand"]["cards"] = new_hand
        
        obs = parse_gamestate_to_obs(self.current_gamestate)
        
        hands_left = self.current_gamestate.get("round", {}).get("hands_left", 0)
        total_chips = self.current_gamestate.get("round", {}).get("chips", 0)
        blind_target = self.current_gamestate.get("blind", {}).get("chips", 300)
        
        # Episode ends if we run out of hands OR if we beat the blind
        done = hands_left <= 0 or total_chips >= blind_target
        
        return obs, reward, done, False, {}
