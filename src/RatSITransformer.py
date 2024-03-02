import logging
import itertools
import tensorflow as tf
from opennmt import config as config_util
from opennmt.utils import misc
from opennmt.inputters.inputter import ParallelInputter
from opennmt.inputters.text_inputter import WordEmbedder
from opennmt.layers import common
from opennmt.layers.position import SinusoidalPositionEncoder
from opennmt.layers.reducer import ConcatReducer, JoinReducer
from opennmt.layers.transformer import MultiHeadAttentionReduction
from opennmt.encoders.encoder import ParallelEncoder
from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder
from opennmt.decoders.self_attention_decoder import SelfAttentionDecoder
from opennmt.models.sequence_to_sequence import (
    EmbeddingsSharingLevel,
    SequenceToSequence
)

tf.get_logger().setLevel(logging.INFO)

class SelectiveInteractionsParallelEncoder(ParallelEncoder):
    def call(self, inputs, sequence_length=None, training=None):
        all_outputs = []
        all_states = []
        all_sequence_lengths = []
        parallel_inputs = isinstance(inputs, (list, tuple))
        parallel_encoders = isinstance(self.encoders, (list, tuple))

        if parallel_encoders and parallel_inputs and len(inputs) != len(self.encoders):
            raise ValueError(
                "ParallelEncoder expects as many inputs as parallel encoders"
            )
        if parallel_encoders:
            encoders = self.encoders
        else:
            encoders = itertools.repeat(
                self.encoders, len(inputs) if parallel_inputs else 1
            )

        for i, encoder in enumerate(encoders):
            if parallel_inputs:
                encoder_inputs = inputs[i]
                length = sequence_length[i]
            else:
                encoder_inputs = inputs
                length = sequence_length

            if i > 0:
                # selective Interactions: encoders except the first one has <SRC> as pretext
                assert(parallel_inputs)
                outputs, state, length = encoder(
                    encoder_inputs, sequence_length=length, training=training,
                    src_length = sequence_length[0]
                )
            else:
                outputs, state, length = encoder(
                    encoder_inputs, sequence_length=length, training=training
                )

            if self.outputs_layer_fn is not None:
                if isinstance(self.outputs_layer_fn, list):
                    outputs = self.outputs_layer_fn[i](outputs)
                else:
                    outputs = self.outputs_layer_fn(outputs)

            all_outputs.append(outputs)
            all_states.append(state)
            all_sequence_lengths.append(length)

        outputs, sequence_length = self.outputs_reducer(
            all_outputs, sequence_length=all_sequence_lengths
        )

        if self.combined_output_layer_fn is not None:
            outputs = self.combined_output_layer_fn(outputs)

        return (outputs, self.states_reducer(all_states), sequence_length)


class WeightSharingSelectiveInteractionsSelfAttentionEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        encModule,
    ):
        super().__init__()
        self._encModule = encModule
        self._sensepId = 3

    def build_mask(self, inputs, sequence_length=None, dtype=tf.bool):
        """Builds a boolean mask for :obj:`inputs`."""
        if sequence_length is None:
            return None
        return tf.sequence_mask(
            sequence_length, maxlen=tf.shape(inputs)[1], dtype=dtype
        )

    def call(self, inputs, sequence_length=None, training=None, src_length=None):
        inputs *= self._encModule.num_units**0.5
        if self._encModule.position_encoder is not None:
            inputs = self._encModule.position_encoder(inputs)
        inputs = common.dropout(inputs, self._encModule.dropout, training=training)
        mask = self.build_mask(inputs, sequence_length=sequence_length)

        for _layer in self._encModule.layers:
            def _layer_call(layer, x, mask=None, training=None):
                y, _ = layer.self_attention(x, mask=mask, training=training)
                y = layer.ffn(y, training=training)
                return y
            inputs = _layer_call(_layer, inputs, mask=mask, training=training)

        outputs = self._encModule.layer_norm(inputs) if self._encModule.layer_norm is not None else inputs

        # apply selective interaction
        # <CXT> ｟SENSEP｠ <SRC> => <CXT>
        assert(src_length != None)
        assert(sequence_length != None)
        cxt_length = sequence_length - src_length - 1

        return outputs, None, cxt_length


class RatSITransformer(SequenceToSequence):
    def __init__(
        self,
        #
        # source_inputter=None,
        # target_inputter=None,
        num_target_context = 5,
        #
        num_layers=6,
        num_units=1024,
        num_heads=16,
        ffn_inner_dim=1024,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ffn_activation=tf.nn.relu,
        mha_bias=True,
        position_encoder_class=SinusoidalPositionEncoder,
        share_embeddings=EmbeddingsSharingLevel.AUTO,
        maximum_relative_position=None,
        attention_reduction=MultiHeadAttentionReduction.FIRST_HEAD_LAST_LAYER,
        pre_norm=True,
        output_layer_bias=True,
    ):
        self.num_target_context = num_target_context
        source_inputter=ParallelInputter(
                [
                    WordEmbedder(embedding_size=num_units)
                    for _ in range(num_target_context + 1)
                ]
            )
        target_inputter=WordEmbedder(embedding_size=num_units)


        if isinstance(num_layers, (list, tuple)):
            num_encoder_layers, num_decoder_layers = num_layers
        else:
            num_encoder_layers, num_decoder_layers = num_layers, num_layers


        encoder = SelfAttentionEncoder(
                num_encoder_layers,
                num_units=num_units,
                num_heads=num_heads,
                ffn_inner_dim=ffn_inner_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                ffn_activation=ffn_activation,
                mha_bias=mha_bias,
                position_encoder_class=position_encoder_class,
                maximum_relative_position=maximum_relative_position,
                pre_norm=pre_norm,
            )

        target_context_encoders = [
            WeightSharingSelectiveInteractionsSelfAttentionEncoder(encoder)
            for _ in range(self.num_target_context)
        ]

        encoder = SelectiveInteractionsParallelEncoder(
            [encoder, *target_context_encoders],
            outputs_reducer=ConcatReducer(axis=1), # concat along time dimension
            states_reducer=None,
        )

        decoder = SelfAttentionDecoder(
            num_decoder_layers,
            num_units=num_units,
            num_heads=num_heads,
            ffn_inner_dim=ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation,
            mha_bias=mha_bias,
            position_encoder_class=position_encoder_class,
            num_sources=1,
            maximum_relative_position=maximum_relative_position,
            attention_reduction=attention_reduction,
            pre_norm=pre_norm,
            output_layer_bias=output_layer_bias,
        )

        self._num_units = num_units
        super().__init__(
            source_inputter,
            target_inputter,
            encoder,
            decoder,
            share_embeddings=share_embeddings,
        )


    def auto_config(self, num_replicas=1):
        config = super().auto_config(num_replicas=num_replicas)
        config = config_util.merge_config(
            config,
            {
                "params": {
                    "average_loss_in_time": True,
                    "label_smoothing": 0.1,
                    "optimizer": "LazyAdam",
                    "optimizer_params": {"beta_1": 0.9, "beta_2": 0.998},
                    "learning_rate": 2.0,
                    "decay_type": "NoamDecay",
                    "decay_params": {
                        "model_dim": self._num_units,
                        "warmup_steps": 8000,
                    },
                },
                "train": {
                    "effective_batch_size": 25000,
                    "batch_size": 3072,
                    "batch_type": "tokens",
                    "maximum_features_length": (
                        100
                        if self.features_inputter.num_outputs == 1
                        else [100] * self.features_inputter.num_outputs
                    ),
                    "maximum_labels_length": 100,
                    "keep_checkpoint_max": 8,
                    "average_last_checkpoints": 8,
                },
            },
        )
        max_length = config["train"]["maximum_features_length"]
        return misc.merge_dict(
            config, {"train": {"maximum_features_length": [max_length for _ in range(self.num_target_context + 1)]}}
        )


model = RatSITransformer
