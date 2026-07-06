"""
SurGE benchmark evaluator for MARS.

Evaluates MARS-generated surveys against ground-truth reference data from
the SurGE benchmark (Tsinghua University, 2025).

Two ways to load data:

1. **MARS-converted format** (ready-to-use)::

       evaluator = SurGEEvaluator(data_dir="data/surge")

   Expects::

       data_dir/
       ├── topics.json              # {"topic_id": "topic string", ...}
       ├── ground_truth_refs.json   # {"topic_id": [{"title":"...", ...}, ...]}
       └── gold_surveys/            # (optional)

2. **Direct SurGE repo loading** (when you have SurGE cloned)::

       evaluator = SurGEEvaluator.from_surge(
           surge_dir="D:/.../SurGE/data"
       )

   This uses :class:`mars.evaluation.surge_adapter.SurGEAdapter` to read the
   official ``surveys.json`` (and optionally ``corpus.json``) directly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from mars.evaluation.metrics import (
    compute_citation_metrics,
    extract_cited_paper_titles,
    rouge_l,
)

logger = logging.getLogger(__name__)

# Default benchmark: 12 CS topics curated as a minimal evaluation set.
_DEFAULT_TOPICS: Dict[str, str] = {
    "federated_learning_privacy": "Federated Learning with Differential Privacy",
    "graph_neural_networks": "Graph Neural Networks for Recommendation",
    "llm_reasoning": "Large Language Model Reasoning and Chain-of-Thought",
    "knowledge_graph_embedding": "Knowledge Graph Embedding Methods",
    "multi_agent_systems": "LLM-based Multi-Agent Systems",
    "contrastive_learning": "Contrastive Learning for Self-Supervised Representation",
    "transformer_efficiency": "Efficient Transformer Architectures",
    "text_to_image_generation": "Text-to-Image Diffusion Models",
    "code_generation_llm": "Code Generation with Large Language Models",
    "reinforcement_learning_human_feedback": "RLHF and Alignment Methods for LLMs",
    "neural_architecture_search": "Neural Architecture Search and AutoML",
    "explainable_ai": "Explainable AI Methods and Interpretability",
}

_DEFAULT_GT_REFS: Dict[str, List[Dict[str, str]]] = {
    "federated_learning_privacy": [
        {"title": "Communication-Efficient Learning of Deep Networks from Decentralized Data", "doi": "10.48550/arXiv.1602.05629", "year": "2016"},
        {"title": "Deep Learning with Differential Privacy", "doi": "10.1145/2976749.2978318", "year": "2016"},
        {"title": "Federated Learning: Challenges, Methods, and Future Directions", "doi": "10.1109/MSP.2020.2975749", "year": "2020"},
        {"title": "Advances and Open Problems in Federated Learning", "doi": "10.48550/arXiv.1912.04977", "year": "2019"},
        {"title": "Differentially Private Federated Learning: A Client Level Perspective", "doi": "10.48550/arXiv.1712.07557", "year": "2017"},
        {"title": "Federated Optimization: Distributed Machine Learning for On-Device Intelligence", "doi": "10.48550/arXiv.1610.02527", "year": "2016"},
        {"title": "Learning Differentially Private Recurrent Language Models", "doi": "10.48550/arXiv.1710.06963", "year": "2018"},
        {"title": "A Survey on Federated Learning Systems: Vision, Hype and Reality for Data Privacy and Protection", "doi": "10.1109/TKDE.2021.3124599", "year": "2021"},
    ],
    "graph_neural_networks": [
        {"title": "Semi-Supervised Classification with Graph Convolutional Networks", "doi": "10.48550/arXiv.1609.02907", "year": "2017"},
        {"title": "Graph Attention Networks", "doi": "10.48550/arXiv.1710.10903", "year": "2018"},
        {"title": "Inductive Representation Learning on Large Graphs", "doi": "10.48550/arXiv.1706.02216", "year": "2017"},
        {"title": "Neural Message Passing for Quantum Chemistry", "doi": "10.48550/arXiv.1704.01212", "year": "2017"},
        {"title": "How Powerful are Graph Neural Networks?", "doi": "10.48550/arXiv.1810.00826", "year": "2019"},
        {"title": "A Comprehensive Survey on Graph Neural Networks", "doi": "10.1109/TNNLS.2020.2978386", "year": "2021"},
    ],
    "llm_reasoning": [
        {"title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "doi": "10.48550/arXiv.2201.11903", "year": "2022"},
        {"title": "Large Language Models are Zero-Shot Reasoners", "doi": "10.48550/arXiv.2205.11916", "year": "2022"},
        {"title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", "doi": "10.48550/arXiv.2305.10601", "year": "2023"},
        {"title": "Self-Consistency Improves Chain of Thought Reasoning in Language Models", "doi": "10.48550/arXiv.2203.11171", "year": "2022"},
        {"title": "Graph of Thoughts: Solving Elaborate Problems with Large Language Models", "doi": "10.48550/arXiv.2308.09687", "year": "2023"},
    ],
    "knowledge_graph_embedding": [
        {"title": "Translating Embeddings for Modeling Multi-relational Data", "doi": "", "year": "2013"},
        {"title": "Knowledge Graph Embedding: A Survey of Approaches and Applications", "doi": "10.1109/TKDE.2017.2754499", "year": "2017"},
        {"title": "Complex Embeddings for Simple Link Prediction", "doi": "10.48550/arXiv.1606.06357", "year": "2016"},
        {"title": "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space", "doi": "10.48550/arXiv.1902.10197", "year": "2019"},
        {"title": "A Review of Relational Machine Learning for Knowledge Graphs", "doi": "10.1109/JPROC.2015.2483592", "year": "2016"},
    ],
    "multi_agent_systems": [
        {"title": "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation", "doi": "10.48550/arXiv.2308.08155", "year": "2023"},
        {"title": "CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society", "doi": "10.48550/arXiv.2303.17760", "year": "2023"},
        {"title": "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework", "doi": "10.48550/arXiv.2308.00352", "year": "2023"},
        {"title": "A Survey on LLM-based Multi-Agent Systems: Workflow, Infrastructure, and Challenges", "doi": "10.1007/s11263-024-02149-w", "year": "2024"},
    ],
    "contrastive_learning": [
        {"title": "A Simple Framework for Contrastive Learning of Visual Representations", "doi": "10.48550/arXiv.2002.05709", "year": "2020"},
        {"title": "Momentum Contrast for Unsupervised Visual Representation Learning", "doi": "10.48550/arXiv.1911.05722", "year": "2020"},
        {"title": "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning", "doi": "10.48550/arXiv.2006.07733", "year": "2020"},
        {"title": "SimCLR v2: Big Self-Supervised Models are Strong Semi-Supervised Learners", "doi": "10.48550/arXiv.2006.10029", "year": "2020"},
        {"title": "Supervised Contrastive Learning", "doi": "10.48550/arXiv.2004.11362", "year": "2021"},
    ],
    "transformer_efficiency": [
        {"title": "Attention Is All You Need", "doi": "10.48550/arXiv.1706.03762", "year": "2017"},
        {"title": "Linformer: Self-Attention with Linear Complexity", "doi": "10.48550/arXiv.2006.04768", "year": "2020"},
        {"title": "Reformer: The Efficient Transformer", "doi": "10.48550/arXiv.2001.04451", "year": "2020"},
        {"title": "Efficient Transformers: A Survey", "doi": "10.1145/3530811", "year": "2022"},
        {"title": "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "doi": "10.48550/arXiv.2205.14135", "year": "2022"},
    ],
    "text_to_image_generation": [
        {"title": "Denoising Diffusion Probabilistic Models", "doi": "10.48550/arXiv.2006.11239", "year": "2020"},
        {"title": "High-Resolution Image Synthesis with Latent Diffusion Models", "doi": "10.48550/arXiv.2112.10752", "year": "2022"},
        {"title": "Hierarchical Text-Conditional Image Generation with CLIP Latents", "doi": "10.48550/arXiv.2204.06125", "year": "2022"},
        {"title": "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding", "doi": "10.48550/arXiv.2205.11487", "year": "2022"},
        {"title": "Scaling Autoregressive Models for Content-Rich Text-to-Image Generation", "doi": "10.48550/arXiv.2206.10789", "year": "2022"},
    ],
    "code_generation_llm": [
        {"title": "Evaluating Large Language Models Trained on Code", "doi": "10.48550/arXiv.2107.03374", "year": "2021"},
        {"title": "CodeBERT: A Pre-Trained Model for Programming and Natural Languages", "doi": "10.48550/arXiv.2002.08155", "year": "2020"},
        {"title": "CodeLlama: Open Foundation Models for Code", "doi": "10.48550/arXiv.2308.12950", "year": "2023"},
        {"title": "StarCoder: May the Source Be with You!", "doi": "10.48550/arXiv.2305.06161", "year": "2023"},
        {"title": "A Systematic Evaluation of Large Language Models of Code", "doi": "10.48550/arXiv.2202.13169", "year": "2022"},
    ],
    "reinforcement_learning_human_feedback": [
        {"title": "Training Language Models to Follow Instructions with Human Feedback", "doi": "10.48550/arXiv.2203.02155", "year": "2022"},
        {"title": "Deep Reinforcement Learning from Human Preferences", "doi": "10.48550/arXiv.1706.03741", "year": "2017"},
        {"title": "Proximal Policy Optimization Algorithms", "doi": "10.48550/arXiv.1707.06347", "year": "2017"},
        {"title": "Scaling Laws for Reward Model Overoptimization", "doi": "10.48550/arXiv.2210.10760", "year": "2022"},
        {"title": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "doi": "10.48550/arXiv.2305.18290", "year": "2023"},
        {"title": "Constitutional AI: Harmlessness from AI Feedback", "doi": "10.48550/arXiv.2212.08073", "year": "2022"},
    ],
    "neural_architecture_search": [
        {"title": "Neural Architecture Search with Reinforcement Learning", "doi": "10.48550/arXiv.1611.01578", "year": "2017"},
        {"title": "DARTS: Differentiable Architecture Search", "doi": "10.48550/arXiv.1806.09055", "year": "2019"},
        {"title": "Efficient Neural Architecture Search via Parameter Sharing", "doi": "10.48550/arXiv.1802.03268", "year": "2018"},
        {"title": "Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation", "doi": "10.48550/arXiv.1901.02985", "year": "2019"},
        {"title": "NAS-Bench-101: Towards Reproducible Neural Architecture Search", "doi": "10.48550/arXiv.2002.03465", "year": "2020"},
    ],
    "explainable_ai": [
        {"title": "'Why Should I Trust You?': Explaining the Predictions of Any Classifier", "doi": "10.48550/arXiv.1602.04938", "year": "2016"},
        {"title": "A Unified Approach to Interpreting Model Predictions", "doi": "10.48550/arXiv.1705.07874", "year": "2017"},
        {"title": "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization", "doi": "10.48550/arXiv.1610.02391", "year": "2017"},
        {"title": "Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges", "doi": "10.1016/j.inffus.2019.12.012", "year": "2020"},
        {"title": "Attention is Not Explanation", "doi": "10.48550/arXiv.1902.10186", "year": "2019"},
    ],
}


@dataclass
class SurGEResult:
    """Result of evaluating MARS on a single SurGE topic."""

    topic_id: str
    topic_name: str
    recall: float
    precision: float
    f1: float
    generated_ref_count: int
    ground_truth_ref_count: int
    matched_ref_count: int
    rouge_l_score: Optional[float] = None
    gold_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "topic_name": self.topic_name,
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "generated_ref_count": self.generated_ref_count,
            "ground_truth_ref_count": self.ground_truth_ref_count,
            "matched_ref_count": self.matched_ref_count,
            "rouge_l": self.rouge_l_score,
            "gold_available": self.gold_available,
        }


class SurGEEvaluator:
    """Evaluate a MARS-generated survey against SurGE ground-truth data.

    Parameters:
        data_dir: Path to MARS-format SurGE data files.  When the directory is
            empty or does not exist, the evaluator falls back to built-in
            minimal benchmark data (12 CS topics).

    Use :meth:`from_surge` to load directly from an official SurGE repo clone.
    """

    def __init__(self, data_dir: str | Path = "data/surge"):
        self._data_dir = Path(data_dir)
        self._topics: Dict[str, str] = {}
        self._gt_refs: Dict[str, List[Dict[str, str]]] = {}
        self._gold_surveys: Dict[str, str] = {}
        self._load_data()

    # ------------------------------------------------------------------
    # Factory: load directly from SurGE official dataset
    # ------------------------------------------------------------------

    @classmethod
    def from_surge(
        cls,
        surge_dir: str | Path,
        load_corpus: bool = False,
    ) -> "SurGEEvaluator":
        """Create evaluator directly from an official SurGE dataset clone.

        Args:
            surge_dir: Path to the SurGE ``data/`` directory containing
                ``surveys.json``.
            load_corpus: When True and ``corpus.json`` exists, enables
                paper-title-level citation metrics.  Defaults to False
                (doc_id-level only); enable when you have the full corpus.

        Returns:
            A :class:`SurGEEvaluator` pre-populated with 205 SurGE topics.
        """
        from mars.evaluation.surge_adapter import SurGEAdapter

        adapter = SurGEAdapter(surge_dir, load_corpus=load_corpus)

        evaluator = cls.__new__(cls)
        evaluator._data_dir = Path(surge_dir)
        evaluator._topics = adapter.get_topics()
        evaluator._gt_refs = {}
        for survey_id in evaluator._topics:
            evaluator._gt_refs[survey_id] = adapter.get_ground_truth_refs(survey_id)
        evaluator._gold_surveys = {}
        evaluator._adapter = adapter  # keep ref for search/get_structure
        return evaluator

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        """Load benchmark data from disk or fall back to built-in defaults."""
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Topics
        topics_path = self._data_dir / "topics.json"
        if topics_path.is_file():
            self._topics = json.loads(topics_path.read_text(encoding="utf-8"))
        else:
            logger.info("No topics.json found — using built-in 12-topic set.")
            self._topics = dict(_DEFAULT_TOPICS)

        # Ground-truth references
        gt_path = self._data_dir / "ground_truth_refs.json"
        if gt_path.is_file():
            self._gt_refs = json.loads(gt_path.read_text(encoding="utf-8"))
        else:
            logger.info("No ground_truth_refs.json found — using built-in data.")
            self._gt_refs = dict(_DEFAULT_GT_REFS)

        # Gold surveys (optional)
        gold_path = self._data_dir / "gold_surveys" / "gold_surveys.json"
        if gold_path.is_file():
            self._gold_surveys = json.loads(gold_path.read_text(encoding="utf-8"))
        elif (self._data_dir / "gold_surveys.json").is_file():
            self._gold_surveys = json.loads(
                (self._data_dir / "gold_surveys.json").read_text(encoding="utf-8")
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def topics(self) -> Dict[str, str]:
        """Return the topic_id → topic_name mapping."""
        return dict(self._topics)

    @property
    def topic_ids(self) -> List[str]:
        """Return all available topic IDs."""
        return list(self._topics.keys())

    def get_ground_truth_refs(self, topic_id: str) -> List[Dict[str, str]]:
        """Return ground-truth references for a topic (title, doi, year)."""
        return list(self._gt_refs.get(topic_id, []))

    def get_gold_survey(self, topic_id: str) -> Optional[str]:
        """Return the human-written gold survey text, if available."""
        return self._gold_surveys.get(topic_id)

    def evaluate(
        self,
        topic_id: str,
        generated_survey: str,
        *,
        generated_refs: Optional[List[str]] = None,
    ) -> SurGEResult:
        """Evaluate a generated survey against SurGE ground-truth.

        Args:
            topic_id: One of the topic IDs from ``self.topic_ids``.
            generated_survey: Full Markdown text of the generated survey.
            generated_refs: Pre-extracted paper titles from the generated
                survey.  When None, titles are extracted automatically from
                the reference section of *generated_survey*.

        Returns:
            :class:`SurGEResult` with recall, precision, F1, and counts.
        """
        if topic_id not in self._topics:
            raise KeyError(
                f"Unknown topic '{topic_id}'. Available: {list(self._topics.keys())}"
            )

        gt_refs = self._gt_refs.get(topic_id, [])
        gt_titles = [r["title"] for r in gt_refs]

        if generated_refs is None:
            generated_refs = extract_cited_paper_titles(generated_survey)

        metrics = compute_citation_metrics(generated_refs, gt_titles, fuzzy=True)

        # ROUGE-L if gold survey available
        rouge = None
        gold_available = False
        gold = self._gold_surveys.get(topic_id)
        if gold:
            gold_available = True
            rouge = round(rouge_l(generated_survey, gold), 4)

        return SurGEResult(
            topic_id=topic_id,
            topic_name=self._topics[topic_id],
            recall=metrics["recall"],
            precision=metrics["precision"],
            f1=metrics["f1"],
            generated_ref_count=metrics["generated_count"],
            ground_truth_ref_count=metrics["ground_truth_count"],
            matched_ref_count=metrics["matched_count"],
            rouge_l_score=rouge,
            gold_available=gold_available,
        )

    def evaluate_all(
        self,
        survey_map: Dict[str, str],
    ) -> List[SurGEResult]:
        """Evaluate all topics that have a generated survey.

        Args:
            survey_map: ``{topic_id: generated_survey_text}`` mapping.

        Returns:
            List of :class:`SurGEResult`, one per evaluated topic.
        """
        results: List[SurGEResult] = []
        for topic_id, survey_text in survey_map.items():
            if topic_id not in self._topics:
                logger.warning("Skipping unknown topic '%s'.", topic_id)
                continue
            results.append(self.evaluate(topic_id, survey_text))
        return results

    def aggregate_results(self, results: List[SurGEResult]) -> Dict[str, float]:
        """Compute macro-average metrics across all evaluated topics."""
        if not results:
            return {}
        n = len(results)
        return {
            "num_topics": n,
            "avg_recall": round(sum(r.recall for r in results) / n, 4),
            "avg_precision": round(sum(r.precision for r in results) / n, 4),
            "avg_f1": round(sum(r.f1 for r in results) / n, 4),
            "avg_rouge_l": (
                round(sum(r.rouge_l_score for r in results if r.rouge_l_score is not None)
                      / max(1, sum(1 for r in results if r.rouge_l_score is not None)), 4)
                if any(r.rouge_l_score is not None for r in results)
                else None
            ),
        }

    def save_results(self, results: List[SurGEResult], path: Path | str) -> None:
        """Save evaluation results as JSON."""
        output = {
            "aggregate": self.aggregate_results(results),
            "per_topic": [r.to_dict() for r in results],
        }
        Path(path).write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("SurGE evaluation results saved to %s", path)
