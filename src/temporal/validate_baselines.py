import glob, pickle, pandas as pd
BASELINE_DIR = "../../graphs/baselines"

def validate_both():
    benign_tx, adv_tx = [], []
    
    print("🔍 VALIDATING BENIGN BASELINES")
    for path in glob.glob(f"{BASELINE_DIR}/*_benign.gpickle"):
        date = path.split('/')[-1].replace('_benign.gpickle', '')
        with open(path, 'rb') as f:
            G = pickle.load(f)
        total_tx = sum(1 for n,d in G.nodes(data=True) if d.get("node_type")=="transaction")
        fraud_tx = sum(1 for n,d in G.nodes(data=True) if d.get("fraud")==1)
        benign_tx.append((date, total_tx, fraud_tx))
    
    df_b = pd.DataFrame(benign_tx, columns=['date','total_tx','fraud_tx'])
    print(f"Benign: {df_b['total_tx'].sum()} tx, {df_b['fraud_tx'].sum()} fraud ✓")
    
    print("\n🔍 VALIDATING ADVERSARIAL BASELINES")
    for path in glob.glob(f"{BASELINE_DIR}/*_adv.gpickle"):
        date = path.split('/')[-1].replace('_adv.gpickle', '')
        with open(path, 'rb') as f:
            G = pickle.load(f)
        total_tx = sum(1 for n,d in G.nodes(data=True) if d.get("node_type")=="transaction")
        fraud_tx = sum(1 for n,d in G.nodes(data=True) if d.get("fraud")==1)
        adv_tx.append((date, total_tx, fraud_tx))
    
    df_a = pd.DataFrame(adv_tx, columns=['date','total_tx','fraud_tx'])
    print(f"Adversarial: {df_a['total_tx'].sum()} tx, {df_a['fraud_tx'].sum()} fraud (should=total)")
    print(f"Fraud purity: {df_a['fraud_tx'].sum()/df_a['total_tx'].sum():.1%}")

if __name__ == "__main__":
    validate_both()
