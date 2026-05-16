# Creator Content Similarity Search Engine

A Python-based creator discovery engine using **TF-IDF vectorization** and **cosine similarity**.

## Features

- Sample dataset of **100 creator bios** across gaming, streaming, and content creation
- Search function to find similar creators from a query bio
- Precision@10 evaluation metric

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage Example

```python
from creator_similarity import (
    SAMPLE_CREATOR_BIOS,
    CreatorSimilarityEngine,
    evaluate_precision_at_10,
)

engine = CreatorSimilarityEngine(SAMPLE_CREATOR_BIOS)

query = "Competitive FPS streamer with ranked grind sessions and Twitch livestreams"
results = engine.search_similar_creators(query, top_k=5)

for creator in results:
    print(f"{creator['name']} -> score={creator['score']:.3f}")

# Precision@10 evaluation
example_eval_set = [
    {
        "query_bio": "Cozy simulation creator with weekly challenge episodes",
        "relevant_ids": {"creator_056", "creator_060", "creator_086"},
    }
]
print("Precision@10:", evaluate_precision_at_10(engine, example_eval_set))
```

## Run Tests

```bash
python -m unittest -q
```
