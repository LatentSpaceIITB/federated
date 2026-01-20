#!/bin/bash
# MedPromptFolio Experiment Scripts
# ==================================

# Exit on error
set -e

# Configuration
OUTPUT_DIR="./output"
MEDCLIP_CHECKPOINT="/home/pinak/MedClip/proj1/pretrained/medclip-vit"
NUM_ROUNDS=50
BATCH_SIZE=32
SEED=42

# Create output directory
mkdir -p $OUTPUT_DIR

echo "=================================================="
echo "MedPromptFolio Experiments"
echo "=================================================="

# ============================================
# 1. Quick Test with Synthetic Data
# ============================================
run_synthetic_test() {
    echo ""
    echo "Running synthetic data test..."
    python scripts/train.py \
        --method medpromptfolio \
        --use_synthetic \
        --num_rounds 10 \
        --batch_size 16 \
        --output_dir "$OUTPUT_DIR/synthetic_test" \
        --seed $SEED
}

# ============================================
# 2. Full Baseline Comparison (Synthetic)
# ============================================
run_baseline_comparison() {
    echo ""
    echo "Running baseline comparison..."

    for method in medpromptfolio fedavg fedprox local; do
        echo "  Running $method..."
        python scripts/train.py \
            --method $method \
            --use_synthetic \
            --num_rounds $NUM_ROUNDS \
            --batch_size $BATCH_SIZE \
            --output_dir "$OUTPUT_DIR/baselines/$method" \
            --seed $SEED
    done
}

# ============================================
# 3. Theta Ablation Study
# ============================================
run_theta_ablation() {
    echo ""
    echo "Running theta ablation study..."

    for theta in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
        echo "  Running theta=$theta..."
        python scripts/train.py \
            --method medpromptfolio \
            --theta $theta \
            --use_synthetic \
            --num_rounds $NUM_ROUNDS \
            --batch_size $BATCH_SIZE \
            --output_dir "$OUTPUT_DIR/theta_ablation/theta_$theta" \
            --seed $SEED
    done
}

# ============================================
# 4. Context Length Ablation
# ============================================
run_nctx_ablation() {
    echo ""
    echo "Running context length ablation..."

    for n_ctx in 2 4 8 16 32; do
        echo "  Running n_ctx=$n_ctx..."
        python scripts/train.py \
            --method medpromptfolio \
            --n_ctx $n_ctx \
            --use_synthetic \
            --num_rounds $NUM_ROUNDS \
            --batch_size $BATCH_SIZE \
            --output_dir "$OUTPUT_DIR/nctx_ablation/nctx_$n_ctx" \
            --seed $SEED
    done
}

# ============================================
# 5. Full Experiment with Real Data
# ============================================
run_real_data_experiment() {
    echo ""
    echo "Running experiment with real data..."
    echo "Note: Requires actual dataset paths"

    # Update these paths for your setup
    CHEXPERT_ROOT="/data/chexpert"
    MIMIC_ROOT="/data/mimic-cxr"
    NIH_ROOT="/data/nih-chestxray14"
    VINDR_ROOT="/data/vindr-cxr"

    python scripts/train.py \
        --method medpromptfolio \
        --chexpert_root $CHEXPERT_ROOT \
        --mimic_root $MIMIC_ROOT \
        --nih_root $NIH_ROOT \
        --vindr_root $VINDR_ROOT \
        --medclip_checkpoint $MEDCLIP_CHECKPOINT \
        --num_rounds $NUM_ROUNDS \
        --batch_size $BATCH_SIZE \
        --output_dir "$OUTPUT_DIR/real_data" \
        --seed $SEED
}

# ============================================
# 6. Multi-Seed Experiments
# ============================================
run_multiseed() {
    echo ""
    echo "Running multi-seed experiments..."

    for seed in 42 123 456 789 1000; do
        echo "  Running seed=$seed..."
        python scripts/train.py \
            --method medpromptfolio \
            --use_synthetic \
            --num_rounds $NUM_ROUNDS \
            --batch_size $BATCH_SIZE \
            --output_dir "$OUTPUT_DIR/multiseed/seed_$seed" \
            --seed $seed
    done
}

# ============================================
# Main Script
# ============================================

# Parse arguments
case "$1" in
    "test")
        run_synthetic_test
        ;;
    "baselines")
        run_baseline_comparison
        ;;
    "theta")
        run_theta_ablation
        ;;
    "nctx")
        run_nctx_ablation
        ;;
    "real")
        run_real_data_experiment
        ;;
    "multiseed")
        run_multiseed
        ;;
    "all")
        run_synthetic_test
        run_baseline_comparison
        run_theta_ablation
        ;;
    *)
        echo "Usage: $0 {test|baselines|theta|nctx|real|multiseed|all}"
        echo ""
        echo "Options:"
        echo "  test      - Quick test with synthetic data"
        echo "  baselines - Compare all baseline methods"
        echo "  theta     - Theta ablation study"
        echo "  nctx      - Context length ablation"
        echo "  real      - Run with real chest X-ray data"
        echo "  multiseed - Multi-seed experiments"
        echo "  all       - Run test, baselines, and theta ablation"
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "Experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=================================================="
