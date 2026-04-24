"""Deterministic player for in-game Balatro actions."""

import itertools
import random
import time
import math
from collections import Counter
from typing import Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

RANKS = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10,
         'J': 11, 'Q': 12, 'K': 13, 'A': 14}

def parse_card(card: dict) -> dict:
    """Parse a card from gamestate into a rich dictionary for LLM evaluation."""
    modifier = card.get('modifier', {})
    if not isinstance(modifier, dict):
        modifier = {}
        
    state_val = card.get('state', {})
    if not isinstance(state_val, dict):
        state_val = {}
    
    enhancement = modifier.get('enhancement')
    edition = modifier.get('edition')
    seal = modifier.get('seal')
    debuff = state_val.get('debuff', False)
    
    # Some cards might be stone cards without rank/suit
    if enhancement == 'STONE':
        return {
            'rank': 'None', 'suit': 'None', 'rank_val': 0, 'chip_value': 50,
            'enhancement': enhancement, 'edition': edition, 'seal': seal, 'debuff': debuff
        }
        
    value = card.get('value', {})
    if isinstance(value, dict):
        rank_str = value.get('rank', '2')
        suit = value.get('suit', 'Spades')
    else:
        rank_str = '2'
        suit = 'Spades'
        
    chip_value = 0
    if rank_str in ['T', 'J', 'Q', 'K']:
        chip_value = 10
    elif rank_str == 'A':
        chip_value = 11
    elif rank_str.isdigit():
        chip_value = int(rank_str)
        
    return {
        'rank': rank_str, 'suit': suit, 'rank_val': RANKS.get(rank_str, 0), 'chip_value': chip_value,
        'enhancement': enhancement, 'edition': edition, 'seal': seal, 'debuff': debuff
    }

def get_scoring_cards(cards: List[dict], hand_type: str) -> List[dict]:
    if hand_type in ["Flush", "Straight", "Straight Flush", "Full House", "Five of a Kind", "Flush House", "Flush Five"]:
        return cards
        
    ranks = [c['rank_val'] for c in cards if c['rank_val'] > 0]
    rank_counts = Counter(ranks)
    
    scoring_ranks = set()
    if hand_type == "Four of a Kind":
        scoring_ranks = {r for r, count in rank_counts.items() if count >= 4}
    elif hand_type == "Three of a Kind":
        scoring_ranks = {r for r, count in rank_counts.items() if count >= 3}
    elif hand_type == "Two Pair" or hand_type == "Pair":
        scoring_ranks = {r for r, count in rank_counts.items() if count >= 2}
    elif hand_type == "High Card":
        if ranks:
            scoring_ranks = {max(ranks)}
            
    scoring = []
    # High Card: only ONE card of the highest rank scores (even if they play two, it would be a pair, but just in case)
    high_card_scored = False
    
    for c in cards:
        if c.get('enhancement') == 'STONE':
            scoring.append(c)
        elif c['rank_val'] in scoring_ranks:
            if hand_type == "High Card":
                if not high_card_scored:
                    scoring.append(c)
                    high_card_scored = True
            else:
                scoring.append(c)
                
    return scoring

def default_evaluator(cards, hand_type, hands_info):
    if hand_type not in hands_info:
        return 0
    chips = hands_info[hand_type]['chips']
    mult = hands_info[hand_type]['mult']
    scoring_cards = get_scoring_cards(cards, hand_type)
    for c in scoring_cards:
        if not c.get('debuff', False):
            chips += c.get('chip_value', 0)
    return chips * mult

def get_hand_type(cards: List[dict]) -> str:
    """Return the highest poker hand type for a list of up to 5 cards."""
    if not cards:
        return "High Card"
        
    ranks = [c['rank_val'] for c in cards if c['rank_val'] > 0]
    suits = [c['suit'] for c in cards if c['suit'] != 'None']
    
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)
    
    is_flush = len(suits) == 5 and len(set(suits)) == 1
    
    is_straight = False
    if len(ranks) == 5 and len(set(ranks)) == 5:
        sorted_ranks = sorted(ranks)
        if sorted_ranks[-1] - sorted_ranks[0] == 4:
            is_straight = True
        # A-2-3-4-5 straight
        elif sorted_ranks == [2, 3, 4, 5, 14]:
            is_straight = True

    if is_flush and counts == [5]:
        return "Flush Five"
    if counts == [5]:
        return "Five of a Kind"
    if is_flush and is_straight:
        return "Straight Flush"
    if is_flush and counts == [3, 2]:
        return "Flush House"
    if counts == [4, 1] or counts == [4]:
        return "Four of a Kind"
    if counts == [3, 2]:
        return "Full House"
    if is_flush:
        return "Flush"
    if is_straight:
        return "Straight"
    if counts and counts[0] == 3:
        return "Three of a Kind"
    if counts and counts[0] == 2 and len(counts) > 1 and counts[1] == 2:
        return "Two Pair"
    if counts and counts[0] == 2:
        return "Pair"
        
    return "High Card"


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        # state: (hand, deck, score, hands_left, discards_left)
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.untried_actions = None
        self.wins = 0.0
        self.visits = 0

    def get_untried_actions(self, evaluator, hands_info):
        if self.untried_actions is not None:
            return self.untried_actions
            
        hand, deck, score, hands_left, discards_left = self.state
        actions = []
        
        if hands_left > 0:
            play_combos = []
            for r in range(1, min(6, len(hand) + 1)):
                for indices in itertools.combinations(range(len(hand)), r):
                    combo = [hand[i] for i in indices]
                    htype = get_hand_type(combo)
                    s = evaluator(combo, htype, hands_info)
                    logger.info(f"MCTS Get untried actions score: {s}, Combo: {combo}")
                    play_combos.append((s, list(indices)))
            
            # Keep top 15 highest scoring plays to prune search space
            play_combos.sort(key=lambda x: x[0], reverse=True)
            for _, indices in play_combos[:15]:
                actions.append(("play", indices))
                
        if discards_left > 0:
            discard_actions = []
            for r in range(1, min(6, len(hand) + 1)):
                for indices in itertools.combinations(range(len(hand)), r):
                    discard_actions.append(("discard", list(indices)))
            # Randomly keep 15 discard options to prune
            random.shuffle(discard_actions)
            actions.extend(discard_actions[:15])
            
        self.untried_actions = actions
        return self.untried_actions

    def uct_select_child(self):
        C = 1.414
        best_child = None
        best_score = -float('inf')
        for child in self.children:
            if child.visits == 0:
                continue
            score = (child.wins / child.visits) + C * math.sqrt(math.log(self.visits) / child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

def apply_action(state, action, evaluator, hands_info):
    hand, deck, score, hands_left, discards_left = state
    action_type, indices = action
    
    new_hand = [hand[i] for i in range(len(hand)) if i not in indices]
    new_deck = list(deck)
    random.shuffle(new_deck)
    
    cards_needed = min(8 - len(new_hand), len(new_deck))
    drawn = new_deck[:cards_needed]
    new_deck = new_deck[cards_needed:]
    new_hand.extend(drawn)
    
    new_score = score
    new_hands_left = hands_left
    new_discards_left = discards_left
    
    if action_type == "play":
        combo = [hand[i] for i in indices]
        htype = get_hand_type(combo)
        new_score += evaluator(combo, htype, hands_info)
        logger.info(f"New Score: {new_score}, Combo: {combo}")
        new_hands_left -= 1
    elif action_type == "discard":
        new_discards_left -= 1
        
    return (new_hand, new_deck, new_score, new_hands_left, new_discards_left)

def rollout(state, evaluator, hands_info, blind_target):
    hand, deck, score, hands_left, discards_left = state
    current_deck = list(deck)
    random.shuffle(current_deck)
    
    while hands_left > 0 and score < blind_target and hand:
        best_score = -1.0
        best_indices = []
        
        for r in range(1, min(6, len(hand) + 1)):
            for indices in itertools.combinations(range(len(hand)), r):
                combo = [hand[i] for i in indices]
                htype = get_hand_type(combo)
                s = evaluator(combo, htype, hands_info)
                logger.info(f"Rollout Score: {s}, Combo: {combo}")
                if s > best_score:
                    best_score = s
                    best_indices = list(indices)
                    
        # Rollout heuristic: discard if best play is poor
        needed_avg = (blind_target - score) / hands_left
        if best_score < needed_avg and discards_left > 0:
            discards_left -= 1
            sorted_indices = sorted(range(len(hand)), key=lambda i: hand[i].get('chip_value', 0))
            discard_indices = sorted_indices[:5]
            
            hand = [hand[i] for i in range(len(hand)) if i not in discard_indices]
            cards_needed = min(8 - len(hand), len(current_deck))
            hand.extend(current_deck[:cards_needed])
            current_deck = current_deck[cards_needed:]
            continue
            
        hands_left -= 1
        score += best_score
        hand = [hand[i] for i in range(len(hand)) if i not in best_indices]
        cards_needed = min(8 - len(hand), len(current_deck))
        hand.extend(current_deck[:cards_needed])
        current_deck = current_deck[cards_needed:]
        
    return 1.0 if score >= blind_target else 0.0

def mcts_search(root_state, evaluator, hands_info, blind_target, time_limit=1.5):
    logger.info(f"MCTS search started for {time_limit} seconds.")
    logger.info(f"root_state: {root_state}")
    logger.info(f"hands_info: {hands_info}")

    root_node = MCTSNode(root_state)
    start_time = time.time()
    
    simulations = 0
    while time.time() - start_time < time_limit:
        simulations += 1
        # 1. Select
        node = root_node
        while not node.get_untried_actions(evaluator, hands_info) and node.children:
            selected_node = node.uct_select_child()
            if not selected_node:
                break
            node = selected_node
                
        if not node:
            break
            
        # 2. Expand
        untried = node.get_untried_actions(evaluator, hands_info)
        if untried and (node.state[3] > 0 or node.state[4] > 0): # hands_left or discards_left > 0
            # Randomly select an untried action
            action_idx = random.randrange(len(untried))
            action = untried.pop(action_idx)
            
            # Transition
            next_state = apply_action(node.state, action, evaluator, hands_info)
            child_node = MCTSNode(next_state, parent=node, action=action)
            node.children.append(child_node)
            node = child_node
            
        # 3. Simulate (Rollout)
        win = rollout(node.state, evaluator, hands_info, blind_target)
        
        # 4. Backpropagate
        while node is not None:
            node.visits += 1
            node.wins += win
            node = node.parent
            
    logger.info(f"MCTS completed {simulations} simulations in {time_limit} seconds.")
    
    # Select best action from root (most visited)
    if not root_node.children:
        # Fallback if no search completed
        untried = root_node.get_untried_actions(evaluator, hands_info)
        if untried:
            return untried[0]
        return ("play", [0]) 
        
    best_child = max(root_node.children, key=lambda c: c.visits)
    logger.info(f"MCTS selected {best_child.action[0]} with Win Rate: {best_child.wins / best_child.visits:.2%} ({int(best_child.wins)}/{best_child.visits} visits)")
    return best_child.action


class DeterministicPlayer:
    """Uses an MCTS engine to evaluate and play hands."""
    
    def __init__(self):
        self.seen_cards = []
        
    def reset_round(self):
        """Reset state at the beginning of a round."""
        self.seen_cards = []
        
    def decide(self, gamestate: dict, evaluator: Any) -> dict:
        """Decide the next action based on MCTS and LLM evaluator."""
        
        # Auto-use any Planet cards immediately
        consumables = gamestate.get("consumables", {}).get("cards", [])
        for i, card in enumerate(consumables):
            if isinstance(card, dict) and card.get("set") == "PLANET":
                return {
                    "method": "use",
                    "params": {
                        "consumable": i,
                        "reasoning": f"Automatically using Planet card '{card.get('label', 'Unknown')}' to permanently upgrade hand."
                    }
                }
                
        hand_cards = gamestate.get("hand", {}).get("cards", [])
        valid_cards = []
        for c in hand_cards:
            if isinstance(c, dict):
                valid_cards.append(c)
        parsed_cards = [parse_card(c) for c in valid_cards]
        
        hands_info = gamestate.get("hands", {})
        
        # Build state
        hands_left = gamestate.get("round", {}).get("hands_left", 1)
        discards_left = gamestate.get("round", {}).get("discards_left", 0)
        current_score = gamestate.get("round", {}).get("chips", 0)
        blind_target = float('inf')
        for blind in gamestate.get("blinds", {}).values():
            if blind.get("status") == "CURRENT":
                blind_target = blind.get("score", float('inf'))
                break
        
        deck_cards_raw = gamestate.get("cards", {}).get("cards", [])
        valid_deck_cards = [c for c in deck_cards_raw if isinstance(c, dict)]
        parsed_deck = [parse_card(c) for c in valid_deck_cards]
        if not parsed_deck:
            parsed_deck = [{'rank': 'A', 'suit': random.choice(['Spades', 'Hearts', 'Clubs', 'Diamonds']), 'rank_val': 14, 'chip_value': 11, 'enhancement': None, 'edition': None, 'seal': None, 'debuff': False} for _ in range(52)]
            
        root_state = (parsed_cards, parsed_deck, current_score, hands_left, discards_left)
        
        action_type, indices = mcts_search(root_state, evaluator, hands_info, blind_target, time_limit=1.5)
        
        # In Balatro, we can only play/discard max 5 cards
        if len(indices) > 5:
            indices = indices[:5]
            
        # Balatro API expects 1-based indices
        api_indices = [i + 1 for i in indices]
            
        return {
            "method": action_type,
            "params": {
                "cards": api_indices,
                "reasoning": f"MCTS search decided to {action_type} {len(indices)} cards."
            }
        }
