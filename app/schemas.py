from pydantic import BaseModel
from typing import Dict, Any
from datetime import datetime

class AnalyzeSummary(BaseModel):
    file: str
    total_rows: int
    attack_count: int
    normal_count: int
    timestamp: datetime

class PacketOut(BaseModel):
    id: str
    timestamp: datetime
    src: str
    dst: str
    status: str
    meta: Dict[str, Any]
