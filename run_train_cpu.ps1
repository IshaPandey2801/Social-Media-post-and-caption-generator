# run_train_cpu.ps1 - runs fine_tune_rewriter on CPU
.\venv\Scripts\Activate.ps1

# merge if needed
if (Test-Path output\train_augmented.jsonl) {
    Get-Content output\train.jsonl, output\train_augmented.jsonl | Set-Content output\train_merged.jsonl
    $trainfile = "output\train_merged.jsonl"
} else {
    $trainfile = "output\train.jsonl"
}

python fine_tune_rewriter.py --train $trainfile --valid output\valid.jsonl --output_dir outputs\t5_rewriter --batch_size 4 --num_epochs 3
