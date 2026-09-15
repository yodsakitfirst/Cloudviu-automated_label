from dataclasses import dataclass


@dataclass(frozen=True)
class Tile:
    index: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Candidate:
    xyxy: tuple[float, float, float, float]
    localization_confidence: float
    prompt_name: str
    tile_index: int
    mask_used: bool


@dataclass(frozen=True)
class RankedSku:
    class_id: int
    barcode: str
    sku_name: str
    score: float
    visual_similarity: float
    text_similarity: float


@dataclass(frozen=True)
class ClassifiedCandidate:
    candidate: Candidate
    assigned_class_id: int
    assigned_name: str
    accepted: bool
    review_reason: str
    rankings: tuple[RankedSku, ...]
