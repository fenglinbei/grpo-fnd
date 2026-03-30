from typing import List, Dict, Any

def basic_collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch