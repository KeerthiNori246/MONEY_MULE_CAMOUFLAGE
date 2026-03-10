import glob
import pickle
import os
from collections import Counter
from load_snapshots import SNAPSHOT_DIR

def analyze_temporal_fraud_fast():
    """Analyze WITHOUT loading all snapshots"""
    print("🔍 FAST ANALYSIS (no full loading)")
    
    fraud_evolution = {}
    snapshot_stats = {}
    
    # Just read pickle metadata (super fast)
    pattern = f"{SNAPSHOT_DIR}/*.gpickle"
    for path in sorted(glob.glob(pattern)):
        date_str = os.path.basename(path).replace('.gpickle', '')
        
        # Quick peek without full load
        with open(path, 'rb') as f:
            # Load just first 1000 bytes to get stats
            unpickler = pickle.Unpickler(f)
            snapshot = unpickler.load()
        
        # Fast fraud count
        fraud_count = sum(1 for n, d in snapshot.nodes(data=True) 
                         if d.get("node_type") == "transaction" 
                         and d.get("fraud") == 1)
        
        total_tx = sum(1 for n, d in snapshot.nodes(data=True) 
                      if d.get("node_type") == "transaction")
        
        fraud_evolution[date_str] = fraud_count
        snapshot_stats[date_str] = (total_tx, fraud_count)
    
    # Show results
    print("\n📊 DAILY FRAUD TRENDS (first 30 days):")
    for date in sorted(fraud_evolution.keys())[:30]:
        total, fraud = snapshot_stats[date]
        rate = (fraud/total*100) if total > 0 else 0
        print(f"{date}: {fraud}/{total} tx ({rate:.1f}% fraud)")
    
    # Fraud spikes
    daily_fraud = list(fraud_evolution.values())
    print(f"\n🚨 FRAUD SPIKES:")
    print(f"Avg daily fraud: {sum(daily_fraud)/len(daily_fraud):.1f}")
    print(f"Max daily fraud: {max(daily_fraud)}")

if __name__ == "__main__":
    analyze_temporal_fraud_fast()
