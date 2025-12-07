import sys
import os
import torch
import pytest
from unittest.mock import MagicMock, patch

# Add parent directory to path to import tutorial_9
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tutorial_9

def test_ngram_model():
    text = "あいうえおあいうえお"
    n = 2
    model = tutorial_9.NgramModel(text, n=n)

    # Check if counts are correct
    assert model.ngram_counts['あ']['い'] == 2
    assert model.ngram_counts['い']['う'] == 2

    # Check generation
    generated = model.generate('あ', length=5)
    assert len(generated) == 6 # start_char + 5 generated
    assert generated.startswith('あ')

    # Test n=3
    n = 3
    model = tutorial_9.NgramModel(text, n=n)
    # context length 2
    # "あいうえおあいうえお"
    # "あい" -> "う"
    assert model.ngram_counts['あい']['う'] == 2
    generated = model.generate('あい', length=5)
    assert len(generated) == 7 # start_context (2) + 5 generated
    assert generated.startswith('あい')

def test_simple_transformer_shape():
    vocab_size = 100
    n_embd = 32
    block_size = 16
    n_head = 4
    n_layer = 2

    model = tutorial_9.SimpleTransformer(vocab_size, n_embd, block_size, n_head, n_layer)

    batch_size = 4
    # Create dummy input
    idx = torch.randint(0, vocab_size, (batch_size, block_size))

    logits, loss = model(idx)

    # Check output shape: (B, T, vocab_size)
    assert logits.shape == (batch_size, block_size, vocab_size)
    assert loss is None

def test_simple_transformer_loss():
    vocab_size = 100
    n_embd = 32
    block_size = 16
    n_head = 4
    n_layer = 2

    model = tutorial_9.SimpleTransformer(vocab_size, n_embd, block_size, n_head, n_layer)

    batch_size = 4
    idx = torch.randint(0, vocab_size, (batch_size, block_size))
    targets = torch.randint(0, vocab_size, (batch_size, block_size))

    logits, loss = model(idx, targets)

    assert loss is not None
    assert isinstance(loss.item(), float)

def test_load_data_mock():
    mock_content = '{"text": "sample text 1"}\n{"text": "sample text 2"}'

    with patch('gzip.open', return_value=MagicMock()) as mock_gzip:
        # Configure the mock to behave like a file object
        mock_gzip.return_value.__enter__.return_value.__iter__.return_value = mock_content.splitlines()

        text = tutorial_9.load_data("dummy.gz", n_lines=2)
        assert "sample text 1" in text
        assert "sample text 2" in text
