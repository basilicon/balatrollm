import itertools
from typing import List, Tuple

MAX_HAND_SIZE = 20

def build_action_space() -> List[Tuple[str, List[int]]]:
    """Builds the mapping from action index to (action_type, card_indices)."""
    actions = []
    
    # Generate all combinations of length 1 to 5 for a hand of size MAX_HAND_SIZE
    combos = []
    for r in range(1, 6):
        combos.extend(list(itertools.combinations(range(MAX_HAND_SIZE), r)))
        
    # Add play actions
    for combo in combos:
        actions.append(("play", list(combo)))
        
    # Add discard actions
    for combo in combos:
        actions.append(("discard", list(combo)))
        
    return actions

ACTION_SPACE = build_action_space()

if __name__ == "__main__":
    print(f"Total actions: {len(ACTION_SPACE)}")
    print("Example actions:")
    for i in range(5):
        print(f"  {i}: {ACTION_SPACE[i]}")
    for i in range(len(ACTION_SPACE)-5, len(ACTION_SPACE)):
        print(f"  {i}: {ACTION_SPACE[i]}")
