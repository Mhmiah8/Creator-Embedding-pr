from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TypedDict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_sample_creator_dataset() -> list[dict[str, str]]:
    """Build a sample dataset containing 100 creator bios."""
    genres = [
        "FPS",
        "RPG",
        "MOBA",
        "battle royale",
        "speedrun",
        "simulation",
        "sandbox",
        "strategy",
        "fighting",
        "sports",
    ]
    styles = [
        "high-energy",
        "educational",
        "comedic",
        "analytical",
        "story-driven",
        "challenge-based",
        "community-focused",
        "competitive",
        "cozy",
        "tech-focused",
    ]
    formats = [
        "Twitch livestreams",
        "YouTube highlights",
        "short-form clips",
        "tutorial videos",
        "reaction streams",
        "ranked grind sessions",
        "collaboration streams",
        "gear breakdowns",
        "behind-the-scenes vlogs",
        "weekly challenge episodes",
    ]

    creators: list[dict[str, str]] = []
    idx = 1
    for genre in genres:
        for style, content_format in zip(styles, formats):
            creators.append(
                {
                    "id": f"creator_{idx:03d}",
                    "name": f"Creator {idx:03d}",
                    "bio": (
                        f"{style.title()} gaming creator focused on {genre} content. "
                        f"Produces {content_format} with strong audience interaction "
                        f"and regular streaming schedules."
                    ),
                }
            )
            idx += 1

    return creators


SAMPLE_CREATOR_BIOS = build_sample_creator_dataset()


@dataclass
class CreatorSimilarityEngine:
    creators: list[dict[str, str]]

    def __post_init__(self) -> None:
        if not self.creators:
            raise ValueError("creators dataset must not be empty")

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.creator_bios = [creator["bio"] for creator in self.creators]
        self.creator_vectors = self.vectorizer.fit_transform(self.creator_bios)

    def search_similar_creators(
        self, query_bio: str, top_k: int = 10
    ) -> list[dict[str, str | float]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_vector = self.vectorizer.transform([query_bio])
        similarities = cosine_similarity(query_vector, self.creator_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]

        results: list[dict[str, str | float]] = []
        for idx in top_indices:
            creator = self.creators[idx]
            results.append(
                {
                    "id": creator["id"],
                    "name": creator["name"],
                    "bio": creator["bio"],
                    "score": float(similarities[idx]),
                }
            )

        return results


def precision_at_k(retrieved_ids: Iterable[str], relevant_ids: set[str], k: int = 10) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    top_k = list(retrieved_ids)[:k]
    if not top_k:
        return 0.0

    hits = sum(1 for creator_id in top_k if creator_id in relevant_ids)
    return hits / len(top_k)


class EvaluationItem(TypedDict):
    query_bio: str
    relevant_ids: set[str]


def evaluate_precision_at_10(
    engine: CreatorSimilarityEngine, evaluation_set: list[EvaluationItem]
) -> float:
    if not evaluation_set:
        return 0.0

    scores: list[float] = []
    for item in evaluation_set:
        query_bio = item["query_bio"]
        relevant_ids = item["relevant_ids"]
        results = engine.search_similar_creators(query_bio, top_k=10)
        retrieved_ids = [str(result["id"]) for result in results]
        scores.append(precision_at_k(retrieved_ids, relevant_ids, k=10))

    return sum(scores) / len(scores)


if __name__ == "__main__":
    engine = CreatorSimilarityEngine(SAMPLE_CREATOR_BIOS)
    query = "Competitive FPS streamer with ranked grind sessions and Twitch livestreams"
    for result in engine.search_similar_creators(query, top_k=5):
        print(f"{result['name']} ({result['score']:.3f})")
