"""Comprehensive test suite for Local Travel Guide Generator core logic across all 5 milestones.
All tests run with fixed random seeds for deterministic reproducibility.
"""
import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

from app import core


# Milestone 1: Prepare a small dataset of travel interests and corresponding local recommendations for a fictional city.
def test_milestone_1():
    res = core.milestone_1()
    assert isinstance(res, dict)
    assert res["city"] == "Aethelgard"
    assert res["total_categories"] >= 5
    assert len(res["samples"]) >= res["total_categories"]

    first_sample = res["samples"][0]
    assert "prompt" in first_sample
    assert "completion" in first_sample
    assert "full_text" in first_sample
    assert "Destination: Aethelgard" in first_sample["prompt"]
    assert "Interest:" in first_sample["prompt"]
    assert len(first_sample["completion"]) > 20


# Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
def test_milestone_2():
    # 1. Test LoRALinear directly
    base_linear = nn.Linear(32, 64)
    lora_mod = core.LoRALinear(base_linear, r=4, alpha=8.0)

    # Invariant checks
    assert not any(p.requires_grad for p in lora_mod.base.parameters())
    assert lora_mod.A.requires_grad and lora_mod.B.requires_grad
    assert torch.all(lora_mod.B == 0)

    x = torch.randn(2, 32)
    # At step 0, LoRA output must be identical to base output
    assert torch.allclose(lora_mod(x), base_linear(x))

    # 2. Test model injection
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model, trainable_params, total_params = core.inject_lora(model, r=8, alpha=16.0)

    # Trainable parameters must be a tiny fraction (~0.18%)
    fraction = trainable_params / total_params
    assert fraction < 0.005, f"Expected < 0.5% trainable parameters, got {fraction * 100:.2f}%"
    assert trainable_params == 147456  # 12 tensors: 6 * (8*768 + 2304*8)

    # All c_attn modules should be LoRALinear instances
    assert all(isinstance(block.attn.c_attn, core.LoRALinear) for block in model.transformer.h)

    # 3. Test milestone_2 return dict
    res = core.milestone_2()
    assert res["trainable_params"] == 147456
    assert res["adapter_tensor_count"] == 12
    assert len(res["loss_history"]) >= 1


# Milestone 3: Develop a Streamlit UI to input interests and destination, generating a mini-guide.
def test_milestone_3():
    res = core.milestone_3()
    assert res["status"] == "ready"
    assert res["has_streamlit"] is True

    # Validate main.py syntax
    import py_compile
    py_compile.compile(res["ui_file"], doraise=True)


# Milestone 4: Evaluate the relevance and appeal of the generated recommendations.
def test_milestone_4():
    res = core.milestone_4()
    assert "relevance_score" in res
    assert "distinct_2_diversity" in res
    assert "average_token_length" in res
    assert isinstance(res["sample_evaluations"], list)
    assert len(res["sample_evaluations"]) > 0


# Milestone 5: Implement adapter merging for inference.
def test_milestone_5():
    res = core.milestone_5()
    assert res["merged"] is True
    assert res["ready_for_deployment"] is True
    assert res["layer_type"] == "Conv1D"

    # Deep verification of merge math
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    core.inject_lora(model, r=4, alpha=8.0, dropout=0.0)
    model.eval()

    # Simulate trained weights in B
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_attn.B.add_(torch.randn_like(block.attn.c_attn.B) * 0.01)

    dummy_input = torch.tensor([[15496, 11, 995]])
    with torch.no_grad():
        logits_before_merge = model(dummy_input).logits

    # Merge model
    merged_model = core.merge_lora_model(model)
    merged_model.eval()
    with torch.no_grad():
        logits_after_merge = merged_model(dummy_input).logits

    # Merged logits must be identical up to floating point precision
    diff = (logits_before_merge - logits_after_merge).abs().max().item()
    assert diff < 1e-3, f"Max diff {diff} exceeds threshold"
    assert torch.allclose(logits_before_merge, logits_after_merge, atol=1e-3)
    # Check that wrapper is replaced with original Conv1D
    assert all(isinstance(block.attn.c_attn, Conv1D) for block in merged_model.transformer.h)
