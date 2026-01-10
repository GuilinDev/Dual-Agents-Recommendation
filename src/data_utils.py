"""
Shared data utilities for Amazon dataset loading and preprocessing.
"""

import os
import json
import gzip
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
import logging
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class AmazonDataLoader:
    """Load and preprocess Amazon Review dataset."""

    # Amazon Reviews 2023 dataset URLs (updated Oct 2024)
    BASE_URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw'

    DATASET_URLS = {
        'Electronics': {
            'reviews': f'{BASE_URL}/review_categories/Electronics.jsonl.gz',
            'meta': f'{BASE_URL}/meta_categories/meta_Electronics.jsonl.gz'
        },
        'Clothing_Shoes_and_Jewelry': {
            'reviews': f'{BASE_URL}/review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz',
            'meta': f'{BASE_URL}/meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz'
        },
        'All_Beauty': {
            'reviews': f'{BASE_URL}/review_categories/All_Beauty.jsonl.gz',
            'meta': f'{BASE_URL}/meta_categories/meta_All_Beauty.jsonl.gz'
        },
        'Home_and_Kitchen': {
            'reviews': f'{BASE_URL}/review_categories/Home_and_Kitchen.jsonl.gz',
            'meta': f'{BASE_URL}/meta_categories/meta_Home_and_Kitchen.jsonl.gz'
        }
    }

    def __init__(self, data_dir: str, category: str = 'Electronics'):
        """
        Initialize data loader.

        Args:
            data_dir: Directory to store/load data
            category: Dataset category ('Electronics' or 'Clothing_Shoes_and_Jewelry')
        """
        self.data_dir = data_dir
        self.category = category
        os.makedirs(data_dir, exist_ok=True)

        self.reviews_file = os.path.join(data_dir, f'{category}_reviews.jsonl.gz')
        self.meta_file = os.path.join(data_dir, f'meta_{category}.jsonl.gz')

    def download_dataset(self, force: bool = False):
        """Download dataset if not exists."""
        urls = self.DATASET_URLS.get(self.category)
        if not urls:
            raise ValueError(f"Unknown category: {self.category}")

        for name, url in [('reviews', urls['reviews']), ('meta', urls['meta'])]:
            filepath = self.reviews_file if name == 'reviews' else self.meta_file

            if os.path.exists(filepath) and not force:
                logger.info(f"{name} file already exists: {filepath}")
                continue

            logger.info(f"Downloading {name} from {url}...")
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))
                with open(filepath, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=name) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))

                logger.info(f"Downloaded to {filepath}")
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
                raise

    def load_reviews(self, max_reviews: Optional[int] = None) -> List[Dict]:
        """Load review data."""
        reviews = []
        logger.info(f"Loading reviews from {self.reviews_file}...")

        with gzip.open(self.reviews_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, desc="Loading reviews")):
                if max_reviews and i >= max_reviews:
                    break
                try:
                    review = json.loads(line.strip())
                    reviews.append(review)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(reviews)} reviews")
        return reviews

    def load_metadata(self) -> Dict[str, Dict]:
        """Load item metadata."""
        metadata = {}
        logger.info(f"Loading metadata from {self.meta_file}...")

        with gzip.open(self.meta_file, 'rt', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading metadata"):
                try:
                    item = json.loads(line.strip())
                    asin = item.get('parent_asin') or item.get('asin')
                    if asin:
                        metadata[asin] = item
                except json.JSONDecodeError:
                    continue

        logger.info(f"Loaded {len(metadata)} items")
        return metadata

    def preprocess(
        self,
        min_user_interactions: int = 5,
        min_item_interactions: int = 5,
        max_reviews: Optional[int] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
        """
        Preprocess dataset with filtering.

        Args:
            min_user_interactions: Minimum interactions per user
            min_item_interactions: Minimum interactions per item
            max_reviews: Maximum reviews to load (for testing)

        Returns:
            Tuple of (interactions DataFrame, item metadata dict)
        """
        # Load raw data
        reviews = self.load_reviews(max_reviews)
        metadata = self.load_metadata()

        # Convert to DataFrame
        df = pd.DataFrame(reviews)

        # Standardize column names - prefer parent_asin over asin for item_id
        if 'parent_asin' in df.columns:
            df['item_id'] = df['parent_asin']
        elif 'asin' in df.columns:
            df['item_id'] = df['asin']

        # Keep necessary columns
        required_cols = ['user_id', 'item_id', 'rating', 'timestamp']
        available_cols = [c for c in required_cols if c in df.columns]
        df = df[available_cols].copy()

        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

        logger.info(f"Initial interactions: {len(df)}")

        # Filter by user activity
        user_counts = df['user_id'].value_counts()
        active_users = user_counts[user_counts >= min_user_interactions].index
        df = df[df['user_id'].isin(active_users)]
        logger.info(f"After user filtering (>={min_user_interactions}): {len(df)}")

        # Filter by item activity
        item_counts = df['item_id'].value_counts()
        active_items = item_counts[item_counts >= min_item_interactions].index
        df = df[df['item_id'].isin(active_items)]
        logger.info(f"After item filtering (>={min_item_interactions}): {len(df)}")

        # Filter metadata to active items
        filtered_metadata = {k: v for k, v in metadata.items() if k in active_items}

        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')

        logger.info(f"Final: {len(df)} interactions, {df['user_id'].nunique()} users, {df['item_id'].nunique()} items")

        return df, filtered_metadata

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.1,
        val_ratio: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create temporal train/val/test split.

        Args:
            df: Interactions DataFrame
            test_ratio: Ratio for test set
            val_ratio: Ratio for validation set

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        df = df.sort_values('timestamp')

        n = len(df)
        test_start = int(n * (1 - test_ratio))
        val_start = int(n * (1 - test_ratio - val_ratio))

        train_df = df.iloc[:val_start]
        val_df = df.iloc[val_start:test_start]
        test_df = df.iloc[test_start:]

        logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        return train_df, val_df, test_df


class UserBehaviorProcessor:
    """Process user behavior for recommendation and explanation generation."""

    def __init__(self, interactions: pd.DataFrame, metadata: Dict[str, Dict]):
        """
        Initialize processor.

        Args:
            interactions: User-item interactions DataFrame
            metadata: Item metadata dictionary
        """
        self.interactions = interactions
        self.metadata = metadata

        # Build user histories
        self.user_histories = self._build_user_histories()

    def _build_user_histories(self) -> Dict[str, List[Dict]]:
        """Build user interaction histories."""
        histories = defaultdict(list)

        for _, row in self.interactions.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']

            item_info = self.metadata.get(item_id, {})
            # Handle description safely
            desc = item_info.get('description', '')
            if isinstance(desc, list):
                desc = desc[0] if desc else ''
            interaction = {
                'item_id': item_id,
                'title': item_info.get('title', 'Unknown'),
                'category': item_info.get('main_category', 'Unknown'),
                'rating': row.get('rating', 0),
                'timestamp': row.get('timestamp', 0),
                'description': desc,
                'price': item_info.get('price', 0),
            }
            histories[user_id].append(interaction)

        # Sort by timestamp
        for user_id in histories:
            histories[user_id] = sorted(histories[user_id], key=lambda x: x.get('timestamp', 0))

        return dict(histories)

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile statistics."""
        history = self.user_histories.get(user_id, [])

        if not history:
            return {}

        categories = [item.get('category', 'Unknown') for item in history]
        ratings = [item.get('rating', 0) for item in history if item.get('rating', 0) > 0]

        return {
            'user_id': user_id,
            'interaction_count': len(history),
            'category_diversity': len(set(categories)) / len(categories) if categories else 0,
            'avg_rating': np.mean(ratings) if ratings else 0,
            'unique_categories': list(set(categories)),
            'category_distribution': dict(pd.Series(categories).value_counts()),
        }

    def get_user_history(self, user_id: str, max_items: int = 50) -> List[Dict]:
        """Get user's interaction history."""
        history = self.user_histories.get(user_id, [])
        return history[-max_items:] if max_items else history

    def sample_users(self, n_users: int, min_interactions: int = 10) -> List[str]:
        """Sample users with sufficient interactions."""
        eligible_users = [
            uid for uid, history in self.user_histories.items()
            if len(history) >= min_interactions
        ]

        if len(eligible_users) < n_users:
            logger.warning(f"Only {len(eligible_users)} eligible users (requested {n_users})")
            return eligible_users

        return random.sample(eligible_users, n_users)


def create_sample_dataset(data_dir: str, n_samples: int = 1000) -> Tuple[pd.DataFrame, Dict]:
    """
    Create a small sample dataset for testing.

    Args:
        data_dir: Directory to save sample data
        n_samples: Number of sample interactions

    Returns:
        Tuple of (interactions DataFrame, metadata dict)
    """
    os.makedirs(data_dir, exist_ok=True)

    # Generate synthetic data
    users = [f"user_{i}" for i in range(100)]
    items = [f"item_{i}" for i in range(200)]
    categories = ['Electronics', 'Computers', 'Phone', 'Camera', 'Audio', 'Gaming']

    interactions = []
    metadata = {}

    # Generate item metadata
    sellers = [f"seller_{i}" for i in range(30)]
    for item_id in items:
        category = random.choice(categories)
        metadata[item_id] = {
            'asin': item_id,
            'title': f"{category} Product {item_id[-3:]}",
            'main_category': category,
            'category': category,  # Add both keys for compatibility
            'description': f"A high-quality {category.lower()} product with excellent features.",
            'price': round(random.uniform(10, 500), 2),
            'seller_id': random.choice(sellers),
            'is_new': random.random() < 0.2,
            'popularity': random.random(),
        }

    # Generate interactions
    base_time = int(datetime(2023, 1, 1).timestamp())
    for i in range(n_samples):
        interactions.append({
            'user_id': random.choice(users),
            'item_id': random.choice(items),
            'rating': random.randint(1, 5),
            'timestamp': base_time + i * 3600,  # 1 hour apart
        })

    df = pd.DataFrame(interactions)
    logger.info(f"Created sample dataset: {len(df)} interactions, {len(users)} users, {len(items)} items")

    return df, metadata


if __name__ == "__main__":
    # Test with sample data
    print("Testing data utilities with sample data...")

    data_dir = "/tmp/sample_amazon_data"
    df, metadata = create_sample_dataset(data_dir, n_samples=500)

    processor = UserBehaviorProcessor(df, metadata)

    # Test user profile
    users = list(processor.user_histories.keys())[:5]
    for user_id in users:
        profile = processor.get_user_profile(user_id)
        print(f"\nUser {user_id}:")
        print(f"  Interactions: {profile.get('interaction_count', 0)}")
        print(f"  Category diversity: {profile.get('category_diversity', 0):.2f}")

    print("\nData utilities test complete!")
