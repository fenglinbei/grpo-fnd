
from enum import Enum
from typing import List
from pydantic import BaseModel

ID2LABEL = {
    0: "PANTS_FIRE",
    1: "FALSE",
    2: "BARELY_TRUE",
    3: "HALF_TRUE",
    4: "MOSTLY_TRUE",
    5: "TRUE",
}

LABEL2ID = {v: k for k, v in ID2LABEL.items()}

class LabelEnum(int, Enum):
    PANTS_FIRE = 0
    FALSE = 1
    BARELY_TRUE = 2
    HALF_TRUE = 3
    MOSTLY_TRUE = 4
    TRUE = 5

class Sample(BaseModel):
    id: int
    claim: str
    label: LabelEnum
    explanation: str
    evidence: List[str]