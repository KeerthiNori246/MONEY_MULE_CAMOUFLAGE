"""
run_motif_pipeline.py
----------------------
Single entry point. Run this file to execute the full
Temporal Motif Evolution pipeline:

    Step 1: Project each daily baseline snapshot → tx-tx graph
    Step 2: Count triangles, wedges, FLF chains per day
    Step 3: Compute ΔM_t time-series
    Step 4: Detect spikes, compare benign vs adversarial
    Step 5: Save CSVs + plots

Usage:
    cd src/temporal/sci/
    python run_motif_pipeline.py
"""

from temporal_motif_evolution import compute_both
from spike_detection import run_spike_analysis


def main():
    print("\n" + "="*60)
    print("  TEMPORAL MOTIF EVOLUTION PIPELINE")
    print("="*60)

    # --- Step 1–3: Compute motif time-series for both baselines ---
    print("\n[PHASE 1] Computing motif time-series...")
    df_benign, df_adv = compute_both()

    # --- Step 4–5: Spike detection + comparison plots ---
    print("\n[PHASE 2] Running spike detection & comparison...")
    summary_benign, summary_adv = run_spike_analysis(k=2.0)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("  Results saved in: graphs/sci_results/")
    print("  Plots saved in  : graphs/sci_results/plots/")
    print("="*60)


if __name__ == "__main__":
    main()