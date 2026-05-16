import unittest

from creator_similarity import (
    SAMPLE_CREATOR_BIOS,
    CreatorSimilarityEngine,
    evaluate_precision_at_10,
    precision_at_k,
)


class CreatorSimilarityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = CreatorSimilarityEngine(SAMPLE_CREATOR_BIOS)

    def test_sample_dataset_has_100_bios(self) -> None:
        self.assertEqual(100, len(SAMPLE_CREATOR_BIOS))

    def test_search_returns_ranked_results(self) -> None:
        results = self.engine.search_similar_creators(
            "Competitive MOBA creator with tutorial videos and active community",
            top_k=10,
        )

        self.assertEqual(10, len(results))
        self.assertGreaterEqual(results[0]["score"], results[-1]["score"])

    def test_precision_at_k(self) -> None:
        retrieved = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
        relevant = {"a", "d", "x"}
        self.assertAlmostEqual(0.2, precision_at_k(retrieved, relevant, k=10))

    def test_evaluate_precision_at_10(self) -> None:
        evaluation_set = [
            {
                "query_bio": "FPS creator making ranked grind sessions and high-energy livestreams",
                "relevant_ids": {"creator_001", "creator_008", "creator_071"},
            },
            {
                "query_bio": "Cozy simulation streamer posting weekly challenge episodes",
                "relevant_ids": {"creator_056", "creator_060", "creator_086"},
            },
        ]

        score = evaluate_precision_at_10(self.engine, evaluation_set)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
