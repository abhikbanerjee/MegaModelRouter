"""Registry: one entry per paper (+ the most informative variants)."""
from .automix import AutoMixRouter
from .benchmark_routing import BenchmarkRoutingRouter
from .cache_distil import CacheDistilRouter
from .collab import EnsembleVoteRouter
from .ecoassistant import EcoAssistantRouter
from .embedllm import EmbedLLMRouter
from .flyswat import FlySwatRouter
from .frugalgpt import FrugalGPTRouter
from .hybrid_llm import HybridLLMRouter
from .llm_blender import LLMBlenderRouter
from .metallm import MetaLLMRouter
from .multiple_minds import MultipleMindsRouter
from .optllm import OptLLMRouter
from .predictive_pref import PredictivePreferenceRouter
from .routellm import RouteLLMRouter
from .routoo import RoutooRouter
from .tryage import TryageRouter
from .two_tier import TwoTierRouter
from .zooter import ZooterRouter

# name -> factory. Order follows the paper list (newest first).
REGISTRY = {
    "embedllm":            lambda: EmbedLLMRouter(),
    "routellm-mf":         lambda: RouteLLMRouter("mf"),
    "routellm-sw_ranking": lambda: RouteLLMRouter("sw_ranking"),
    "routellm-clf":        lambda: RouteLLMRouter("clf"),
    "metallm-offline":     lambda: MetaLLMRouter("offline"),
    "metallm-online":      lambda: MetaLLMRouter("online"),
    "collab-vote":         lambda: EnsembleVoteRouter(),
    "automix-pomdp":       lambda: AutoMixRouter("pomdp"),
    "automix-threshold":   lambda: AutoMixRouter("threshold"),
    "optllm":              lambda: OptLLMRouter(),
    "two-tier-marginmin":  lambda: TwoTierRouter("margin_min"),
    "minds-classifier":    lambda: MultipleMindsRouter("classifier"),
    "minds-cluster":       lambda: MultipleMindsRouter("cluster"),
    "hybrid-det":          lambda: HybridLLMRouter("det"),
    "hybrid-prob":         lambda: HybridLLMRouter("prob"),
    "hybrid-trans":        lambda: HybridLLMRouter("trans"),
    "predictive-pref":     lambda: PredictivePreferenceRouter(),
    "routoo":              lambda: RoutooRouter(),
    "flyswat-ilp":         lambda: FlySwatRouter("ilp"),
    "flyswat-greedy":      lambda: FlySwatRouter("greedy"),
    "zooter":              lambda: ZooterRouter(),
    "cache-distil-margin": lambda: CacheDistilRouter("margin"),
    "cache-distil-qbc":    lambda: CacheDistilRouter("qbc"),
    "cache-distil-front":  lambda: CacheDistilRouter("front"),
    "ecoassistant-hierarchy": lambda: EcoAssistantRouter(),
    "benchmark-knn":       lambda: BenchmarkRoutingRouter(),
    "tryage":              lambda: TryageRouter(),
    "llm-blender":         lambda: LLMBlenderRouter(),
    "frugalgpt":           lambda: FrugalGPTRouter(),
    "frugalgpt-cache":     lambda: FrugalGPTRouter(variant="cache"),
}


def make(name: str):
    return REGISTRY[name]()
