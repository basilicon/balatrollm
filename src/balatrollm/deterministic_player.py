"""Deterministic player for in-game Balatro actions."""

import itertools
import random
from collections import Counter
from typing import Any, List, Tuple

RANKS = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10,
         'J': 11, 'Q': 12, 'K': 13, 'A': 14}

def parse_card(card: dict) -> Tuple[int, str, int]:
    """Parse a card from gamestate into a rank value, suit, and chip value."""
    # Some cards might be stone cards without rank/suit
    modifier = card.get('modifier', {})
    if isinstance(modifier, dict) and modifier.get('enhancement') == 'STONE':
        return (0, 'None', 50) # Stone cards give 50 chips
        
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
        
    return (RANKS.get(rank_str, 0), suit, chip_value)

def get_hand_type(cards: List[Tuple[int, str, int]]) -> str:
    """Return the highest poker hand type for a list of up to 5 cards."""
    if not cards:
        return "High Card"
        
    ranks = [r for r, s, c in cards if r > 0]
    suits = [s for r, s, c in cards if s != 'None']
    
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

    if is_flush and is_straight:
        return "Straight Flush"
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

def fast_evaluate_8cards(cards: List[Tuple[int, str, int]], target_hands: dict) -> float:
    """Fast evaluation of an up to 8-card hand to find the highest weighted hand type."""
    ranks = [r for r, s, c in cards if r > 0]
    suits = [s for r, s, c in cards if s != 'None']
    
    best_score = target_hands.get("High Card", 0.0)
    if not ranks: return best_score
    
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    
    def has_straight(ranks_set):
        if len(ranks_set) < 5: return False
        sorted_ranks = sorted(list(ranks_set))
        consecutive = 1
        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] == sorted_ranks[i-1] + 1:
                consecutive += 1
                if consecutive >= 5: return True
            else:
                consecutive = 1
        if 14 in ranks_set and 2 in ranks_set and 3 in ranks_set and 4 in ranks_set and 5 in ranks_set:
            return True
        return False

    has_flush = any(c >= 5 for c in suit_counts.values())
    has_straight_flag = has_straight(set(ranks))
    
    counts = sorted(rank_counts.values(), reverse=True)
    
    possible_hands = ["High Card"]
    if counts:
        if counts[0] >= 5: possible_hands.append("Five of a Kind")
        if counts[0] >= 4: possible_hands.append("Four of a Kind")
        if counts[0] >= 3 and len(counts) > 1 and counts[1] >= 2: possible_hands.append("Full House")
        if counts[0] >= 3: possible_hands.append("Three of a Kind")
        if counts[0] >= 2 and len(counts) > 1 and counts[1] >= 2: possible_hands.append("Two Pair")
        if counts[0] >= 2: possible_hands.append("Pair")
        
    if has_flush: possible_hands.append("Flush")
    if has_straight_flag: possible_hands.append("Straight")
    
    if has_flush and has_straight_flag:
        for s, count in suit_counts.items():
            if count >= 5:
                flush_ranks = set([r for r, suit, c in cards if suit == s])
                if has_straight(flush_ranks):
                    possible_hands.append("Straight Flush")
                    break

    for h in possible_hands:
        score = target_hands.get(h, 0.0)
        if score > best_score:
            best_score = score
            
    return best_score

class DeterministicPlayer:
    """Uses a predefined game plan to evaluate and play hands."""
    
    def __init__(self):
        self.seen_cards = []
        
    def reset_round(self):
        """Reset state at the beginning of a round."""
        self.seen_cards = []
        
    def decide(self, gamestate: dict[str, Any], game_plan: dict[str, Any]) -> dict[str, Any]:
        """Decide the next action based on gamestate and game plan."""
        
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
        
        # Valid cards only (allow reading hidden cards to prevent crashes)
        valid_cards = []
        for c in hand_cards:
            if isinstance(c, dict):
                valid_cards.append(c)
        parsed_cards = [parse_card(c) for c in valid_cards]
        
        target_hands = game_plan.get("target_hands", {})
        discard_agg = game_plan.get("discard_aggressiveness", 0.5)
        
        discards_left = gamestate.get("round", {}).get("discards_left", 0)
        
        # Find the best 5-card combination we can play right now
        best_play_score = -1.0
        best_play_indices = []
        best_play_type = "High Card"
        
        # Check all combinations of 1 to 5 cards and calculate their raw Balatro score
        combo_scores = []
        max_raw_score = 0.0
        hands_info = gamestate.get("hands", {})
        
        for r in range(1, 6):
            for indices in itertools.combinations(range(len(parsed_cards)), r):
                combo = [parsed_cards[i] for i in indices]
                hand_type = get_hand_type(combo)
                
                hand_data = hands_info.get(hand_type, {})
                base_chips = hand_data.get("chips", 0)
                base_mult = hand_data.get("mult", 1)
                
                # In Balatro, only scoring cards add chips, but as a heuristic we sum all played cards
                card_chips = sum(c for _, _, c in combo)
                raw_score = (base_chips + card_chips) * base_mult
                
                if raw_score > max_raw_score:
                    max_raw_score = raw_score
                    
                combo_scores.append((list(indices), hand_type, raw_score))
                
        # Now find the best play using the estimated score from LLM
        for indices, hand_type, raw_score in combo_scores:
            score = target_hands.get(hand_type, 0.0)
            
            # Tiebreaker: favor higher raw score, then play more cards
            adjusted_score = score + (raw_score * 0.00001) + (len(indices) * 0.000001)
            
            if adjusted_score > best_play_score:
                best_play_score = adjusted_score
                best_play_indices = indices
                best_play_type = hand_type

        hands_left = gamestate.get("round", {}).get("hands_left", 1)
        
        # Decide whether to discard or sacrifice a hand play
        # If discard_agg is 1.0, threshold is max target weight (we discard unless we have the absolute best hand).
        # If discard_agg is 0.0, threshold is 0.0 (we never discard).
        threshold = discard_agg * max(target_hands.values(), default=1.0)
        
        if best_play_score < threshold and (discards_left > 0 or hands_left > 1):
            # Extract remaining deck for Monte Carlo
            deck_cards_raw = gamestate.get("cards", {}).get("cards", [])
            # In Balatro, draw pile is hidden. So just check if it's a dict.
            valid_deck_cards = [c for c in deck_cards_raw if isinstance(c, dict)]
            parsed_deck = [parse_card(c) for c in valid_deck_cards]
            hand_limit = gamestate.get("hand", {}).get("limit", 8)
            N_SIMS = 50

            # We want to discard or sacrifice. Find the best subset of cards to KEEP.
            best_keep_score = -1.0
            best_keep_indices = []
            
            for r in range(1, len(parsed_cards) + 1):
                for indices in itertools.combinations(range(len(parsed_cards)), r):
                    combo = [parsed_cards[i] for i in indices]
                    
                    if parsed_deck:
                        cards_to_draw = hand_limit - len(combo)
                        total_sim_score = 0.0
                        for _ in range(N_SIMS):
                            drawn = random.sample(parsed_deck, min(cards_to_draw, len(parsed_deck)))
                            total_sim_score += fast_evaluate_8cards(combo + drawn, target_hands)
                        score = total_sim_score / N_SIMS
                    else:
                        base_type = get_hand_type(combo)
                        score = target_hands.get(base_type, 0.0)
                        
                    adjusted_score = score + (len(indices) * 0.001)
                    if adjusted_score > best_keep_score:
                        best_keep_score = adjusted_score
                        best_keep_indices = list(indices)
            
            discard_indices = [i for i in range(len(parsed_cards)) if i not in best_keep_indices]
            
            # Can discard/play max 5 cards
            if len(discard_indices) > 5:
                discard_indices = discard_indices[:5]
                
            if len(discard_indices) > 0:
                if discards_left > 0:
                    return {
                        "method": "discard",
                        "params": {
                            "cards": discard_indices,
                            "reasoning": f"Discarding to dig for better hands (aggressiveness {discard_agg}). Kept best potential."
                        }
                    }
                else:
                    return {
                        "method": "play",
                        "params": {
                            "cards": discard_indices,
                            "reasoning": f"Sacrificing a hand play to dig for better cards (aggressiveness {discard_agg}). Kept best potential."
                        }
                    }
                
        # If we didn't discard, play
        return {
            "method": "play",
            "params": {
                "cards": best_play_indices,
                "reasoning": f"Playing {best_play_type} (score: {best_play_score:.3f})."
            }
        }

    def _simulate_round(self, deck: List[Tuple[int, str, int]], payouts: dict, weights: dict, discard_agg: float, hands_left: int, discards_left: int) -> float:
        """Run a fast simulation of a single round with the given game plan logic."""
        score = 0.0
        sim_deck = list(deck)
        random.shuffle(sim_deck)
        
        hand = sim_deck[:8]
        sim_deck = sim_deck[8:]
        
        while hands_left > 0 and hand:
            # Determine best play
            best_play_score = -1.0
            best_play_indices = []
            best_play_type = "High Card"
            
            for r in range(1, min(6, len(hand) + 1)):
                for indices in itertools.combinations(range(len(hand)), r):
                    combo = [hand[i] for i in indices]
                    htype = get_hand_type(combo)
                    
                    s = weights.get(htype, 0.0)
                    # Micro tiebreaker
                    s += sum(c for _,_,c in combo) * 0.00001 + len(indices) * 0.000001
                    
                    if s > best_play_score:
                        best_play_score = s
                        best_play_indices = list(indices)
                        best_play_type = htype
                        
            threshold = discard_agg * max(weights.values(), default=1.0)
            
            if best_play_score < threshold and discards_left > 0:
                # Discard
                best_keep_score = -1.0
                best_keep_indices = []
                
                for r in range(1, len(hand) + 1):
                    for indices in itertools.combinations(range(len(hand)), r):
                        combo = [hand[i] for i in indices]
                        cards_to_draw = 8 - len(combo)
                        total_sim = 0.0
                        for _ in range(3): # Only 3 sims for ultra-fast optimization
                            drawn = random.sample(sim_deck, min(cards_to_draw, len(sim_deck)))
                            total_sim += fast_evaluate_8cards(combo + drawn, weights)
                        avg_sim = total_sim / 3
                        
                        if avg_sim > best_keep_score:
                            best_keep_score = avg_sim
                            best_keep_indices = list(indices)
                            
                discard_indices = [i for i in range(len(hand)) if i not in best_keep_indices][:5]
                if discard_indices:
                    discards_left -= 1
                    hand = [hand[i] for i in range(len(hand)) if i not in discard_indices]
                    draw_count = min(len(discard_indices), len(sim_deck))
                    hand.extend(sim_deck[:draw_count])
                    sim_deck = sim_deck[draw_count:]
                    continue
                    
            # Play
            hands_left -= 1
            score += payouts.get(best_play_type, 0)
            hand = [hand[i] for i in range(len(hand)) if i not in best_play_indices]
            draw_count = min(len(best_play_indices), len(sim_deck))
            hand.extend(sim_deck[:draw_count])
            sim_deck = sim_deck[draw_count:]
            
        return score

    def optimize_strategy(self, gamestate: dict[str, Any], payouts: dict[str, int]) -> dict[str, Any]:
        """Use a Genetic Algorithm to find the optimal target weights and discard aggressiveness."""
        import logging
        logger = logging.getLogger(__name__)
        
        hands_left = gamestate.get("round", {}).get("hands_left", 4)
        discards_left = gamestate.get("round", {}).get("discards_left", 3)
        deck_cards_raw = gamestate.get("cards", {}).get("cards", [])
        valid_deck_cards = [c for c in deck_cards_raw if isinstance(c, dict)]
        parsed_deck = [parse_card(c) for c in valid_deck_cards]
        
        if not parsed_deck:
            # Fallback if deck is empty
            parsed_deck = [(random.randint(2, 14), random.choice(['Spades', 'Hearts', 'Clubs', 'Diamonds']), 10) for _ in range(52)]
            
        # Genetic Algorithm Parameters
        POPULATION_SIZE = 15
        GENERATIONS = 3
        SIMS_PER_EVAL = 4
        
        # Initialize random population
        population = []
        for _ in range(POPULATION_SIZE):
            plan = {
                "weights": {k: random.random() for k in payouts.keys()},
                "agg": random.uniform(0.1, 0.9)
            }
            population.append(plan)
            
        best_plan = None
        best_avg = -1.0
        
        for gen in range(GENERATIONS):
            scored_population = []
            for plan in population:
                total_score = 0.0
                for _ in range(SIMS_PER_EVAL):
                    total_score += self._simulate_round(parsed_deck, payouts, plan["weights"], plan["agg"], hands_left, discards_left)
                avg = total_score / SIMS_PER_EVAL
                scored_population.append((avg, plan))
                
                if avg > best_avg:
                    best_avg = avg
                    best_plan = plan
                    
            # Sort by score descending
            scored_population.sort(key=lambda x: x[0], reverse=True)
            
            if gen == GENERATIONS - 1:
                break
                
            # Selection and Crossover (Keep top 5, mutate/crossover for the rest)
            top_plans = [p for _, p in scored_population[:5]]
            new_population = list(top_plans)
            
            while len(new_population) < POPULATION_SIZE:
                parent1, parent2 = random.sample(top_plans, 2)
                child = {"weights": {}, "agg": 0.0}
                
                # Crossover
                for k in payouts.keys():
                    child["weights"][k] = parent1["weights"][k] if random.random() < 0.5 else parent2["weights"][k]
                    # Mutation
                    if random.random() < 0.2:
                        child["weights"][k] = random.random()
                        
                child["agg"] = parent1["agg"] if random.random() < 0.5 else parent2["agg"]
                if random.random() < 0.2:
                    child["agg"] = random.uniform(0.1, 0.9)
                    
                new_population.append(child)
                
            population = new_population
            
        logger.info(f"Genetic Algorithm optimization complete. Best Expected Value: {best_avg}")
        return {
            "target_hands": best_plan["weights"],
            "discard_aggressiveness": best_plan["agg"]
        }
