from typing import List, Optional

from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel

from fms_extras.models.speculator import MLPSpeculator


class MLPSpeculatorConfig(PretrainedConfig):
    model_type = "mlp_speculator"

    attribute_map = {
        "hidden_size": "emb_dim",
    }

    def __init__(
        self,
        vocab_size: int = 32000,
        emb_dim: int = 4096,
        inner_dim: int = 0,
        n_predict: int = 3,
        top_k_tokens_per_head: List[int] = [5, 4, 3],
        n_candidates: int = 5,
        **kwargs
    ):
        assert len(top_k_tokens_per_head) == n_predict
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.inner_dim = inner_dim
        self.n_predict = n_predict
        self.top_k_tokens_per_head = top_k_tokens_per_head
        self.n_candidates = n_candidates
        super().__init__(**kwargs)


class MLPSpeculatorPreTrainedModel(MLPSpeculator, PreTrainedModel):
    config_class = MLPSpeculatorConfig

    def __init__(self, config: MLPSpeculatorConfig):
        super().__init__(
            config=config,
            emb_dim=config.emb_dim,
            inner_dim=config.inner_dim,
            vocab_size=config.vocab_size,
            n_predict=config.n_predict,
        )
