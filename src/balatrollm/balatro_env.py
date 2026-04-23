import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import json
import time
import logging

from .action_space import ACTION_SPACE
from .observation_space import parse_gamestate_to_obs

logger = logging.getLogger(__name__)

class BalatroEnv(gym.Env):
    """Custom Environment that follows gym interface for Balatro."""
    metadata = {'render.modes': ['human']}

    def __init__(self, host="127.0.0.1", port=12346, seed="AAAAAAA", deck="RED", stake="WHITE"):
        super(BalatroEnv, self).__init__()
        self.host = host
        self.port = port
        self.game_seed = seed
        self.deck = deck
        self.stake = stake
        self.request_id = 0
        
        self.action_space = spaces.Discrete(len(ACTION_SPACE))
        
        # 81 features from observation_space
        self.observation_space = spaces.Box(low=0.0, high=100.0, shape=(81,), dtype=np.float32)
        
        self.current_gamestate = None
        self.previous_chips = 0.0

    def _call(self, method: str, params: dict = None):
        self.request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": self.request_id,
        }
        url = f"http://{self.host}:{self.port}/"
        response = requests.post(url, json=payload, timeout=30.0)
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"Balatro Error: {data['error']}")
        return data["result"]

    def action_masks(self) -> np.ndarray:
        """Returns a boolean array where True means valid action, False means invalid."""
        mask = np.zeros(len(ACTION_SPACE), dtype=np.bool_)
        if not self.current_gamestate:
            return mask
            
        current_state = self.current_gamestate.get("state", "")
        if current_state != "SELECTING_HAND":
            return mask
            
        hand_cards = self.current_gamestate.get("hand", {}).get("cards", [])
        num_cards = len([c for c in hand_cards if not c.get("state", {}).get("hidden", False)])
        
        hands_left = self.current_gamestate.get("round", {}).get("hands_left", 0)
        discards_left = self.current_gamestate.get("round", {}).get("discards_left", 0)
        
        for i, (act_type, indices) in enumerate(ACTION_SPACE):
            # Check if all indices in the action are within the current hand size
            valid_indices = all(idx < num_cards for idx in indices)
            if not valid_indices:
                continue
                
            if act_type == "play" and hands_left > 0:
                mask[i] = True
            elif act_type == "discard" and discards_left > 0:
                mask[i] = True
                
        # If no actions are valid, fallback to preventing crash
        if not np.any(mask):
            mask[0] = True
            
        return mask

    def step(self, action):
        act_type, indices = ACTION_SPACE[action]
        
        # We need to execute the action
        # The game API expects 1-indexed or 0-indexed cards? 
        # bot.py uses 0-indexed indices.
        
        try:
            self.current_gamestate = self._call(act_type, {"cards": indices, "reasoning": "RL Step"})
        except Exception as e:
            logger.error(f"Error executing step: {e}")
            return parse_gamestate_to_obs(self.current_gamestate), -10.0, True, False, {"error": str(e)}

        # Wait for game to process
        time.sleep(0.5)
        self.current_gamestate = self._call("gamestate")

        # Check if game is won/lost or round ended
        done = False
        reward = 0.0
        
        won = self.current_gamestate.get("won", False)
        state = self.current_gamestate.get("state", "")
        
        current_chips = float(self.current_gamestate.get("round", {}).get("chips", 0))
        target_chips = float(self.current_gamestate.get("blind", {}).get("chips", 1))
        
        # Dense reward
        chips_gained = current_chips - self.previous_chips
        reward += chips_gained / target_chips
        self.previous_chips = current_chips

        if won:
            done = True
            reward += 10.0
        elif state == "GAME_OVER":
            done = True
            reward -= 10.0
        elif state not in ["SELECTING_HAND", "GAME_PLANNING"]:
            # If we reached the shop or blindly advanced, let's treat it as a round completion for this episode
            # Or we can automatically navigate the shop. 
            # For simplicity, if we leave SELECTING_HAND, we won the round.
            done = True
            reward += 1.0

        obs = parse_gamestate_to_obs(self.current_gamestate)
        info = {}
        
        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start a new run
        try:
            self._call("menu")
            time.sleep(1.0)
            self.current_gamestate = self._call("start", {
                "deck": self.deck,
                "stake": self.stake,
                "seed": self.game_seed
            })
            time.sleep(1.0)
            self.current_gamestate = self._call("gamestate")
        except Exception as e:
            logger.error(f"Error resetting env: {e}")
            raise
            
        self.previous_chips = 0.0
        obs = parse_gamestate_to_obs(self.current_gamestate)
        return obs, {}

    def render(self, mode='human'):
        pass
