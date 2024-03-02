from opennmt.models import transformer
from opennmt.models.sequence_to_sequence import EmbeddingsSharingLevel

class BaselineTransformer(transformer.Transformer):
    def __init__(self):
        super().__init__(num_layers=6,
                         num_units=1024,
                         num_heads=16,
                         ffn_inner_dim=1024,
                         dropout=0.1,
                         attention_dropout=0.1,
                         ffn_dropout=0.1,
                         share_embeddings=EmbeddingsSharingLevel.AUTO,)

model = BaselineTransformer
