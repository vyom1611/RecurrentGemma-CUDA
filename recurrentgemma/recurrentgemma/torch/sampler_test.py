# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Minimal test for sampler."""

from collections.abc import Iterable

from absl.testing import absltest
from absl.testing import parameterized
from recurrentgemma import common
import recurrentgemma.torch as griffin_lib
import torch


class MockVocab:

  def __init__(self):
    self._start_id = 3
    self._mapping_text_to_id = {
        '<pad>': 0,
        '<s>': 1,
        '</s>': 2,
        'input': 3,
        'string': 4,
        'hello': 5,
        'world': 6,
        'Hello': 7,
        '!': 8,
        'How': 9,
        'are': 10,
        'you?': 11,
    }
    self._vocab_size = len(self._mapping_text_to_id)
    self._separator = ' '

  def pad_id(self) -> int:
    return 0

  def bos_id(self) -> int:
    return 1

  def eos_id(self) -> int:
    return 2

  def GetPieceSize(self) -> int:  # pylint: disable=invalid-name
    return self._vocab_size

  def DecodeIds(self, ids: Iterable[int]) -> str:  # pylint: disable=invalid-name
    reverse_mapping = {v: k for k, v in self._mapping_text_to_id.items()}
    return self._separator.join(reverse_mapping[token] for token in ids)

  def EncodeAsIds(self, text: str) -> list[int]:  # pylint: disable=invalid-name
    words = text.split(self._separator)
    return [self._mapping_text_to_id[word] for word in words]


class SamplerTest(parameterized.TestCase):

  def test_samples(self):
    vocab = MockVocab()
    model_config = common.GriffinConfig(
        block_types=[
            common.TemporalBlockType.RECURRENT,
            common.TemporalBlockType.ATTENTION,
            common.TemporalBlockType.RECURRENT,
        ],
        vocab_size=vocab.GetPieceSize(),
        lru_width=128,
        width=128,
        mlp_expanded_width=512,
        num_heads=4,
        embeddings_scale_by_sqrt_dim=True,
        attention_window_size=2048,
        logits_soft_cap=30.0,
    )
    model = griffin_lib.Griffin(model_config)

    sampler = griffin_lib.Sampler(
        model=model,
        vocab=vocab,
    )

    result = sampler(['input string', 'hello world'], total_generation_steps=10)
    self.assertIsNotNone(result)

  @parameterized.product(echo=[True, False], return_logits=[True, False])
  def test_output_shapes(self, echo: bool, return_logits: bool):
    vocab = MockVocab()
    model_config = common.GriffinConfig(
        block_types=[
            common.TemporalBlockType.RECURRENT,
            common.TemporalBlockType.ATTENTION,
            common.TemporalBlockType.RECURRENT,
        ],
        vocab_size=vocab.GetPieceSize(),
        lru_width=128,
        width=128,
        mlp_expanded_width=512,
        num_heads=4,
        embeddings_scale_by_sqrt_dim=True,
        attention_window_size=2048,
        logits_soft_cap=30.0,
    )
    model = griffin_lib.Griffin(model_config)

    raw_input = 'Hello ! How are you?'
    token_input = torch.tensor(
        [vocab.bos_id()] + vocab.EncodeAsIds(raw_input)
    ).reshape((1, -1))

    batch_size, n_input_tokens = token_input.shape
    sampler = griffin_lib.Sampler(
        model=model,
        vocab=vocab,
    )

    total_generation_steps = 10
    output_sampler = sampler(
        [raw_input],
        total_generation_steps=total_generation_steps,
        echo=echo,
        return_logits=return_logits
    )
    total_tokens = total_generation_steps
    if echo:
      total_tokens += n_input_tokens

    if not return_logits:
      self.assertEmpty(output_sampler.logits)
    else:
      self.assertLen(output_sampler.logits, batch_size)
      for logits in output_sampler.logits:
        self.assertEqual(logits.shape, (total_tokens, vocab.GetPieceSize()))

    self.assertLen(output_sampler.tokens, batch_size)
    for tokens in output_sampler.tokens:
      self.assertEqual(tokens.shape, (total_tokens,))

  @parameterized.parameters([torch.bfloat16, torch.float32])
  def test_forward_equivalence(self, dtype: torch.dtype):
    vocab = MockVocab()
    model_config = common.GriffinConfig(
        block_types=[
            common.TemporalBlockType.RECURRENT,
            common.TemporalBlockType.ATTENTION,
            common.TemporalBlockType.RECURRENT,
        ],
        vocab_size=vocab.GetPieceSize(),
        lru_width=128,
        width=128,
        mlp_expanded_width=512,
        num_heads=4,
        embeddings_scale_by_sqrt_dim=True,
        attention_window_size=2048,
        logits_soft_cap=30.0,
    )

    model = griffin_lib.Griffin(model_config)

    state_dict = model.state_dict()
    for name, tensor in state_dict.items():
      # Cast to dtype and avoid zero-initialization.
      tensor.normal_(0, 1)
      state_dict[name] = tensor.to(dtype)

    raw_input = 'Hello ! How are you?'
    token_input = torch.tensor(
        [vocab.bos_id()] + vocab.EncodeAsIds(raw_input)
    ).reshape((1, -1))

    batch_size, n_input_tokens = token_input.shape

    segment_pos = torch.repeat_interleave(
        torch.arange(n_input_tokens)[None],
        batch_size,
        axis=0,
    )
    output_forward, _ = model.forward(
        tokens=token_input,
        segment_pos=segment_pos,
    )
    sampler = griffin_lib.Sampler(
        model=model,
        vocab=vocab,
    )

    total_generation_steps = 10
    output_sampler = sampler(
        [raw_input],
        total_generation_steps=total_generation_steps,
        echo=True,
        return_logits=True,
    )
    total_sampled_tokens = total_generation_steps + token_input.shape[-1]

    self.assertLen(output_sampler.logits, batch_size)
    for logits in output_sampler.logits:
      self.assertEqual(
          logits.shape, (total_sampled_tokens, model_config.vocab_size)
      )

    self.assertLen(output_sampler.tokens, batch_size)
    for tokens in output_sampler.tokens:
      self.assertEqual(tokens.shape, (total_sampled_tokens,))

    if dtype == torch.bfloat16:
      rtol = 1e-4
      atol = 1e-2
    else:
      rtol = 1e-6
      atol = 1e-3

    for i, out_logits in enumerate(output_sampler.logits):
      torch.testing.assert_close(
          output_forward[i, :n_input_tokens],
          out_logits[:n_input_tokens].type(output_forward.dtype),
          rtol=rtol,
          atol=atol,
      )


if __name__ == '__main__':
  absltest.main()
