# Data Directory

This directory stores the Amazon Reviews 2023 dataset files.

The data will be automatically downloaded when running experiments:

```bash
python src/run_experiments.py --categories All_Beauty
```

## Manual Download

You can also manually download from:
https://amazon-reviews-2023.github.io/

Place the downloaded files in this directory:
- `All_Beauty.jsonl.gz` (reviews)
- `meta_All_Beauty.jsonl.gz` (metadata)

## Supported Categories

- All_Beauty
- Electronics
- Clothing_Shoes_and_Jewelry
- Home_and_Kitchen
- Sports_and_Outdoors
- And more...
