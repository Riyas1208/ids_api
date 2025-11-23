import random
from datetime import datetime
import uuid
import pandas as pd
import io
from typing import Dict, Any

def simulate_packet() -> Dict[str, Any]:
    rnd = random.Random()
    pkt_id = str(uuid.uuid4())[:8]
    ts = datetime.utcnow()
    src = f"192.168.{rnd.randint(0,255)}.{rnd.randint(1,254)}"
    dst = f"10.0.{rnd.randint(0,255)}.{rnd.randint(1,254)}"
    status = "attack" if rnd.random() < 0.14 else "normal"
    meta = {
        "proto": rnd.choice(["TCP", "UDP", "ICMP"]),
        "size": rnd.randint(40,1500),
        "score": round(rnd.random(), 3)
    }
    return {
        "id": pkt_id,
        "timestamp": ts,
        "src": src,
        "dst": dst,
        "status": status,
        "meta": meta
    }

def parse_csv_bytes(content: bytes, max_rows: int = None):
    """
    Safely parse CSV bytes into pandas.DataFrame.
    Returns a DataFrame. Optionally limit rows for memory.
    """
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        s = content.decode('utf-8', errors='ignore')
        df = pd.read_csv(io.StringIO(s))
    if max_rows is not None and len(df) > max_rows:
        return df.head(max_rows)
    return df
