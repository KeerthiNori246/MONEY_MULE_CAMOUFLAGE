"""
run_motif_pipeline.py
----------------------
Single entry point for the full SCI — Temporal Motif Evolution pipeline.

Phases:
    Phase 1 — Compute motif time-series (triangles, wedges) for both baselines
    Phase 2 — Spike detection + comparison plots for motif signals
    Phase 3 — Compute hub redistribution time-series (ΔD_t, ΔC_t, HR_t)
    Phase 4 — Spike detection + comparison plots for hub signals

Usage:
    cd src/temporal/sci/
    python run_motif_pipeline.py
"""

from temporal_motif_evolution import compute_both
from spike_detection import run_spike_analysis
from hub_redistribution import compute_both_hub
from hub_spike_detection import run_hub_spike_analysis


def main():
    print("\n" + "="*60)
    print("  SCI PIPELINE — MOTIF + HUB REDISTRIBUTION")
    print("="*60)

    # ------------------------------------------------------------------
    # Phase 1 — Motif time-series
    # ------------------------------------------------------------------
    print("\n[PHASE 1] Computing motif time-series (triangles, wedges)...")
    df_benign_motif, df_adv_motif = compute_both()

    # ------------------------------------------------------------------
    # Phase 2 — Motif spike detection
    # ------------------------------------------------------------------
    print("\n[PHASE 2] Running motif spike detection & comparison...")
    summary_motif_benign, summary_motif_adv = run_spike_analysis(k=2.0)

    # ------------------------------------------------------------------
    # Phase 3 — Hub redistribution time-series
    # ------------------------------------------------------------------
    print("\n[PHASE 3] Computing hub redistribution time-series (ΔD_t, ΔC_t, HR_t)...")
    df_benign_hub, df_adv_hub = compute_both_hub(alpha=0.5, beta=0.5)

    # ------------------------------------------------------------------
    # Phase 4 — Hub spike detection
    # ------------------------------------------------------------------
    print("\n[PHASE 4] Running hub spike detection & comparison...")
    summary_hub_benign, summary_hub_adv = run_hub_spike_analysis(k=2.0)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("  Results saved in : graphs/sci_results/")
    print("  Plots saved in   : graphs/sci_results/plots/")
    print("\n  Motif CSVs:")
    print("    motif_timeseries_benign.csv")
    print("    motif_timeseries_adv.csv")
    print("    spikes_benign.csv")
    print("    spikes_adv.csv")
    print("\n  Hub CSVs:")
    print("    hub_timeseries_benign.csv")
    print("    hub_timeseries_adv.csv")
    print("    hub_spikes_benign.csv")
    print("    hub_spikes_adv.csv")
    print("="*60)


if __name__ == "__main__":
    main()