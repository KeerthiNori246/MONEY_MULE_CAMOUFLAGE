"""
run_motif_pipeline.py
----------------------
Single entry point for the full SCI pipeline.

Phases:
    Phase 1 — Motif time-series (triangles, wedges) for both baselines
    Phase 2 — Motif spike detection + plots
    Phase 3 — Hub redistribution time-series (ΔD_t, ΔC_t, HR_t)
    Phase 4 — Hub spike detection + plots
    Phase 5 — Combine motif + hub → SCI_t for both baselines
    Phase 6 — SCI spike detection + final separability plots

Usage:
    cd src/temporal/sci/
    python run_motif_pipeline.py
"""

from temporal_motif_evolution import compute_both
from spike_detection           import run_spike_analysis
from hub_redistribution        import compute_both_hub
from hub_spike_detection       import run_hub_spike_analysis
from compute_sci               import compute_both_sci
from sci_spike_detection       import run_sci_spike_analysis


def main():
    print("\n" + "="*60)
    print("  SCI PIPELINE — FULL RUN")
    print("="*60)

    # ------------------------------------------------------------------
    # Phase 1 — Motif time-series
    # ------------------------------------------------------------------
    print("\n[PHASE 1] Computing motif time-series (triangles, wedges)...")
    compute_both()

    # ------------------------------------------------------------------
    # Phase 2 — Motif spike detection
    # ------------------------------------------------------------------
    print("\n[PHASE 2] Running motif spike detection & comparison...")
    run_spike_analysis(k=2.0)

    # ------------------------------------------------------------------
    # Phase 3 — Hub redistribution time-series
    # ------------------------------------------------------------------
    print("\n[PHASE 3] Computing hub redistribution time-series (ΔD_t, ΔC_t, HR_t)...")
    compute_both_hub(alpha=0.5, beta=0.5)

    # ------------------------------------------------------------------
    # Phase 4 — Hub spike detection
    # ------------------------------------------------------------------
    print("\n[PHASE 4] Running hub spike detection & comparison...")
    run_hub_spike_analysis(k=2.0)

    # ------------------------------------------------------------------
    # Phase 5 — Combine into SCI_t
    # ------------------------------------------------------------------
    print("\n[PHASE 5] Computing SCI_t (motif + hub → unified signal)...")
    compute_both_sci(
        motif_w_triangles=0.5,
        motif_w_wedges=0.5,
        sci_w_motif=0.5,
        sci_w_hub=0.5
    )

    # ------------------------------------------------------------------
    # Phase 6 — SCI spike detection + final separability plots
    # ------------------------------------------------------------------
    print("\n[PHASE 6] Running SCI spike detection & final comparison...")
    run_sci_spike_analysis(k=2.0)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print("  Results saved in : graphs/sci_results/")
    print("  Plots saved in   : graphs/sci_results/plots/")
    print()
    print("  Motif CSVs:")
    print("    motif_timeseries_benign.csv / _adv.csv")
    print("    spikes_benign.csv / spikes_adv.csv")
    print()
    print("  Hub CSVs:")
    print("    hub_timeseries_benign.csv / _adv.csv")
    print("    hub_spikes_benign.csv / hub_spikes_adv.csv")
    print()
    print("  SCI CSVs:")
    print("    sci_timeseries_benign.csv / _adv.csv")
    print("    sci_spikes_benign.csv / sci_spikes_adv.csv")
    print()
    print("  Key plots:")
    print("    sci_SCI_t_comparison.png    ← benign vs adv SCI_t")
    print("    sci_overlay.png             ← main result overlay")
    print("    sci_components_benign.png   ← ΔM_t / HR_t / SCI_t breakdown")
    print("    sci_components_adv.png")
    print("="*60)


if __name__ == "__main__":
    main()