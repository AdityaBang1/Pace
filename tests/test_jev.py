"""The Jev-shaped model is only useful if its interface and masking are exact."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pace.jev_model import (DEFAULT_QUESTION, JevChoiceModel, JevConfig, JevEngine,
                            mean_pool, save_jev)
from pace.jev_train import ChoiceDataset, _choice_collate, train_tokenizer

LABELS = ["course_title", "credits", "office_hours"]
SENTENCES = [
    "Course Title: General Chemistry",
    "Credits: 4 units",
    "Office Hours: Mondays 2-4pm in Room 210",
    "Homework 20% and exams 80%",
] * 40


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    root = tmp_path_factory.mktemp("jev")
    corpus = root / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps({"text": text}) for text in SENTENCES),
                      encoding="utf-8")
    return train_tokenizer([corpus], root, vocab_size=200, limit=len(SENTENCES))


def _config(tokenizer) -> JevConfig:
    return JevConfig(vocab_size=tokenizer.get_vocab_size(), embed_dim=32, hidden=64,
                     layers=2, heads=2, ffn=128, max_state_tokens=24, max_option_tokens=8)


def test_mean_pool_ignores_padding():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1, 0]])
    pooled = mean_pool(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))


def test_choice_model_scores_every_option(tokenizer):
    model = JevChoiceModel(_config(tokenizer), LABELS).eval()
    rows = [{"text": "Credits: 4 units", "label": "credits"}]
    dataset = ChoiceDataset(rows, tokenizer, LABELS)
    input_ids, attention, state_mask, labels = next(iter(
        torch.utils.data.DataLoader(dataset, batch_size=1,
                                    collate_fn=lambda batch: _choice_collate(batch,
                                                                             dataset.pad))))
    logits = model(input_ids, attention, state_mask, dataset.option_ids, dataset.option_mask)
    assert logits.shape == (1, len(LABELS))
    assert labels.item() == 1


def test_the_state_mask_covers_the_state_but_not_the_question(tokenizer):
    dataset = ChoiceDataset([{"text": "Credits: 4 units", "label": "credits"}],
                            tokenizer, LABELS)
    ids, mask, _ = dataset[0]
    question_length = len(dataset.question_ids)
    assert mask.sum().item() > 0
    assert mask.sum().item() <= len(ids) - question_length
    assert mask[-question_length:].sum().item() == 0


def test_a_saved_model_answers_the_same_way_after_reload(tokenizer, tmp_path):
    model = JevChoiceModel(_config(tokenizer), LABELS).eval()
    save_jev(model, tokenizer, tmp_path)
    engine = JevEngine(tmp_path, device="cpu")
    result = engine.classify("Credits: 4 units")
    assert result["label"] in LABELS
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-4
    assert result["top"][0]["label"] == result["label"]


def test_the_engine_packs_the_same_question_it_was_trained_with(tokenizer, tmp_path):
    model = JevChoiceModel(_config(tokenizer), LABELS).eval()
    save_jev(model, tokenizer, tmp_path)
    engine = JevEngine(tmp_path, device="cpu")
    ids, attention, state_mask = engine._encode(["Credits: 4 units"])
    assert ids.shape == attention.shape == state_mask.shape
    question_ids = engine.tokenizer.encode(engine.question).ids
    assert ids[0, -len(question_ids) - 1:-1].tolist() == question_ids
    # The question tokens never contribute to the pooled state.
    assert state_mask[0, -len(engine.tokenizer.encode(engine.question).ids):].sum() == 0


def test_option_descriptions_change_the_score(tokenizer):
    model = JevChoiceModel(_config(tokenizer), LABELS).eval()
    dataset = ChoiceDataset([{"text": "Credits: 4 units", "label": "credits"}],
                            tokenizer, LABELS)
    ids, mask, _ = dataset[0]
    input_ids = ids.unsqueeze(0)
    attention = torch.ones_like(input_ids)
    state_mask = mask.unsqueeze(0)
    first = model(input_ids, attention, state_mask, dataset.option_ids, dataset.option_mask)
    shuffled = dataset.option_ids.flip(0)
    shuffled_mask = dataset.option_mask.flip(0)
    second = model(input_ids, attention, state_mask, shuffled, shuffled_mask)
    # Scores travel with the option text: flipping the menu flips the scores.
    assert torch.allclose(first, second.flip(-1), atol=1e-4)


def test_the_menu_loss_matches_cross_entropy_when_nothing_is_hidden():
    from pace.jev_train import _menu_loss

    logits = torch.randn(5, 7)
    labels = torch.tensor([0, 3, 6, 2, 1])
    reference = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.05)
    assert torch.allclose(_menu_loss(logits, labels, None, 0.05), reference, atol=1e-6)


def test_dropped_options_carry_no_probability():
    from pace.jev_train import _menu_loss

    logits = torch.randn(4, 6)
    labels = torch.tensor([1, 2, 3, 4])
    allowed = torch.ones(4, 6, dtype=torch.bool)
    allowed[0, 5] = False
    allowed[1, 0] = False
    loss = _menu_loss(logits, labels, allowed, 0.05)
    probs = torch.log_softmax(logits.masked_fill(~allowed, -1e4), dim=-1).exp()
    assert float(probs[0, 5]) == 0.0 and float(probs[1, 0]) == 0.0
    assert torch.isfinite(loss) and float(loss) < 4.0


def test_pretraining_masks_whole_words_and_remasks_each_epoch(tokenizer):
    from pace.jev_train import MaskedDataset

    text = "Office Hours: Mondays 2-4pm in Room 210"
    dataset = MaskedDataset([text] * 8, tokenizer, max_length=32, mask_prob=0.4, span_prob=0.0)
    mask_id = tokenizer.token_to_id("[MASK]")
    first = {tuple(dataset[index][0].tolist()) for index in range(4)}
    dataset.set_epoch(2)
    second = {tuple(dataset[index][0].tolist()) for index in range(4)}
    assert any(mask_id in row for row in first), "nothing was masked"
    assert first != second, "the same holes were reused in a second epoch"

    checked = 0
    for index in range(len(dataset)):
        ids, labels = dataset[index][0].tolist(), dataset[index][1].tolist()
        pieces = [tokenizer.id_to_token(token) or "" for token in ids]
        masked = [position for position, label in enumerate(labels) if label != -100]
        # Masking works in whole words: a masked continuing piece always follows
        # another masked piece of the same word.
        for position in masked:
            assert position > 0
            if pieces[position].startswith("##"):
                assert labels[position - 1] != -100
            checked += 1
    assert checked > 0, "no position was ever masked"


def test_dynamic_int8_keeps_the_same_scores(tokenizer):
    """Quantized parity: the phone build must not silently change decisions."""
    import copy

    # torch's MHA sparsity fast-path probe reads layer.linear1.weight, which is a
    # function once the Linear is a dynamic quantized module.
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        model = JevChoiceModel(_config(tokenizer), LABELS).eval()
        dataset = ChoiceDataset([{"text": "Credits: 4 units", "label": "credits"}],
                                tokenizer, LABELS)
        ids, mask, _ = dataset[0]
        input_ids = ids.unsqueeze(0)
        attention = torch.ones_like(input_ids)
        state_mask = mask.unsqueeze(0)
        exact = model(input_ids, attention, state_mask,
                      dataset.option_ids, dataset.option_mask)
        quantized = torch.ao.quantization.quantize_dynamic(
            copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8)
        near = quantized(input_ids, attention, state_mask,
                         dataset.option_ids, dataset.option_mask)
        assert near.shape == exact.shape
        assert float((near - exact).abs().max()) < 0.1
        assert int(near.argmax(-1)) == int(exact.argmax(-1))
    finally:
        torch.backends.mha.set_fastpath_enabled(True)


def test_scoring_with_the_checkpoints_own_menu_matches_classification(tokenizer, tmp_path):
    from pace.jev_model import JevEngine

    model = JevChoiceModel(_config(tokenizer), LABELS).eval()
    save_jev(model, tokenizer, tmp_path)
    engine = JevEngine(tmp_path, device="cpu")
    text = "Credits: 4 units"
    written = dict(zip(engine.labels, engine.model.descriptions))
    rewritten = engine.score_against([text], written)[0]
    stored = engine.probabilities([text])[0]
    assert set(rewritten) == set(stored)
    for label, probability in rewritten.items():
        assert abs(probability - stored[label]) < 1e-4
