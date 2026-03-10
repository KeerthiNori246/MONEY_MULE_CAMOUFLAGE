import os
import pickle
import glob
from datetime import datetime
import networkx as nx

SNAPSHOT_DIR = "../../graphs/snapshots"

def load_snapshot(date_str):
    """Load single snapshot"""
    path = f"{SNAPSHOT_DIR}/{date_str}.gpickle"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No snapshot for {date_str}")
    
    with open(path, "rb") as f:
        return pickle.load(f)

def load_all_snapshots():
    """Load all snapshots as dict"""
    snapshots = {}
    pattern = f"{SNAPSHOT_DIR}/*.gpickle"
    
    for path in sorted(glob.glob(pattern)):
        date_str = os.path.basename(path).replace('.gpickle', '')
        snapshots[date_str] = load_snapshot(date_str)
    
    return snapshots

def get_date_range():
    """Get available snapshot dates"""
    pattern = f"{SNAPSHOT_DIR}/*.gpickle"
    dates = [os.path.basename(p).replace('.gpickle', '') 
             for p in glob.glob(pattern)]
    return sorted(dates)
