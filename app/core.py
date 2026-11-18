"""Core implementation of Local Travel Guide Generator using LoRA from scratch.

This module provides:
1. Dataset preparation for travel interests and recommendations in a fictional city (Aethelgard).
2. LoRALinear module and adapter injection/training from scratch (without peft library).
3. Recommendation generation with prompt formatting and generation utilities.
4. Evaluation pipeline measuring relevance (keyword hit rate), diversity, and length metrics.
5. Adapter merging logic for zero-latency inference deployment.
"""

from collections import Counter
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

# =====================================================================
# Milestone 1: Dataset Preparation
# =====================================================================

CITY_NAME = "Aethelgard"

# High-quality dataset of travel interests mapped to rich local recommendations in Aethelgard
# Origin / Source: Custom synthetic domain dataset created for the Local Travel Guide generator drill,
# modeling 8 authentic traveler personas and landmarks across coastal and historic districts.
AETHELGARD_DATASET = [
    {
        "interest": "Culinary & Street Food",
        "keywords": ["food", "culinary", "market", "tasting", "street food", "dining", "flavor"],
        "recommendation": (
            "Head to the Sunken Spice Bazaars in the Low Quarter. Sample sizzling saffron flatbreads "
            "from street vendors and visit The Copper Kettle for slow-roasted honeyed quail paired with "
            "fermented cloudberry cider. Don't miss the midnight night-market along Lantern Pier."
        ),
    },
    {
        "interest": "Ancient History & Ruins",
        "keywords": ["history", "ancient", "ruins", "monument", "temple", "relic", "archaeology"],
        "recommendation": (
            "Climb the Whispering Steps to the Citadel of Sunken Kings. Explore the vaulted subterranean "
            "crypts where ancient basalt codices are preserved, and take a guided twilight tour of the "
            "Marble Acropolis overlooking the Moonlit Bay."
        ),
    },
    {
        "interest": "Nature, Hiking & Parks",
        "keywords": ["nature", "hiking", "park", "garden", "trail", "outdoors", "forest"],
        "recommendation": (
            "Hike the Mist-Shrouded Ridge Trail above the Emerald Falls. Wander through the Hanging "
            "Botanical Arboretums where bioluminescent moss lights the canopy at dusk, and rent a wooden skiff "
            "to explore the quiet inlets of Heron Lagoon."
        ),
    },
    {
        "interest": "Nightlife & Music",
        "keywords": ["nightlife", "music", "bars", "jazz", "dance", "cocktails", "entertainment"],
        "recommendation": (
            "Begin the evening at Velvet Cellar, an underground acoustic jazz cavern carved into coastal bedrock. "
            "Later, move to the Starlight Rooftop Lounge overlooking the Harbor District for artisanal smoked cocktails "
            "and live eclectic synth ensembles that play until sunrise."
        ),
    },
    {
        "interest": "Art & Bohemian Culture",
        "keywords": ["art", "galleries", "bohemian", "sculpture", "studios", "design", "crafts"],
        "recommendation": (
            "Explore the Artisan Guild Alleys in Old North Quarter. Browse independent glassblowing studios, "
            "visit the Prism Contemporary Art Pavilion, and unwind in poet-friendly courtyard cafes serving lavender teas."
        ),
    },
    {
        "interest": "Family & Relaxed Leisure",
        "keywords": ["family", "kids", "relaxed", "leisure", "park", "beach", "easy"],
        "recommendation": (
            "Spend a sunny morning at Seagull Promenade with the clockwork carousel and puppet theater. "
            "Pack a picnic for Meadow Park's wildflower gardens and take an afternoon tram ride across the scenic "
            "Aqueduct Bridge for panoramic views of the entire coastal bay."
        ),
    },
    {
        "interest": "Adventure & Water Sports",
        "keywords": ["adventure", "kayak", "surfing", "diving", "climbing", "water sports", "thrill"],
        "recommendation": (
            "Rent sea kayaks at Corsair Point to navigate the tide caves and sea arches of Falcon Bay. "
            "For cliffside thrills, hire an accredited guide for the Iron Crest Via Ferrata route overlooking the stormy breakers."
        ),
    },
    {
        "interest": "Luxury & Wellness Spas",
        "keywords": ["luxury", "spa", "wellness", "relaxation", "resort", "baths", "massage"],
        "recommendation": (
            "Indulge at the Royal Thermal Baths, fed by geothermal mountain mineral springs. "
            "Follow a steam-stone ritual with an essential pine-oil massage and a private rooftop sunset tasting menu at Celestine."
        ),
    },
]


def format_prompt(interest: str, destination: str = CITY_NAME) -> str:
    """Format input interest and destination into training/inference prompt."""
    return f"Destination: {destination}\nInterest: {interest}\nGuide:"


def get_training_samples(multiplier: int = 4) -> List[Dict[str, str]]:
    """Return formatted prompt-completion pairs suitable for fine-tuning."""
    samples = []
    for item in AETHELGARD_DATASET:
        prompt = format_prompt(item["interest"], CITY_NAME)
        completion = f" {item['recommendation']}"
        full_text = f"{prompt}{completion}"
        samples.append({
            "interest": item["interest"],
            "prompt": prompt,
            "completion": completion,
            "full_text": full_text,
            "keywords": item["keywords"]
        })
    # Multiply to simulate dataset size if requested
    return samples * multiplier


# =====================================================================
# Milestone 2 & 5: LoRA Architecture from Scratch
# =====================================================================

class LoRALinear(nn.Module):
    """
    A frozen base Linear or Conv1D layer + a trainable low-rank detour (B @ A), scaled by alpha/r.
    Follows exact LoRA invariants from Hu et al. (2021):
    - Base parameters frozen (requires_grad = False)
    - A initialized with small Gaussian noise
    - B initialized strictly to zeros (B@A = 0 at step 0)
    - Forward pass: base(x) + (dropout(x) @ A.T) @ B.T * (alpha / r)
    """
    def __init__(self, base: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        # Freeze base parameters
        for p in self.base.parameters():
            p.requires_grad = False

        # Support both PyTorch nn.Linear and Hugging Face GPT-2 Conv1D
        if isinstance(base, Conv1D):
            # Conv1D weights are stored transposed: [d_in, d_out]
            self.d_in, self.d_out = base.weight.shape
        elif isinstance(base, nn.Linear):
            self.d_in = base.in_features
            self.d_out = base.out_features
        else:
            raise TypeError(f"Unsupported base layer type: {type(base)}")

        # Trainable low-rank adapter matrices
        # A: down-projection [r, d_in]
        # B: up-projection [d_out, r]
        self.A = nn.Parameter(torch.randn(r, self.d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(self.d_out, r))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Detour: (dropout(x) @ A.T) @ B.T
        detour = (self.dropout(x) @ self.A.T) @ self.B.T
        return self.base(x) + detour * self.scale

    def merged(self) -> nn.Module:
        """
        Fold the low-rank delta (B @ A * scale) permanently back into the base layer weight,
        restoring the original layer with zero inference overhead.
        """
        with torch.no_grad():
            delta = (self.B @ self.A) * self.scale  # shape: [d_out, d_in]
            if isinstance(self.base, Conv1D):
                # Conv1D stores W as [d_in, d_out], so the update goes in as delta.T
                self.base.weight.add_(delta.T)
            else:
                self.base.weight.add_(delta)
        return self.base


def inject_lora(model: nn.Module, r: int = 8, alpha: float = 16.0, dropout: float = 0.05) -> Tuple[nn.Module, int, int]:
    """
    Freeze all model weights, then wrap the attention Q/K/V projections (c_attn)
    with LoRALinear across all transformer blocks.
    Returns (model, trainable_params, total_params).
    """
    # 1. Freeze the whole pretrained base
    for p in model.parameters():
        p.requires_grad = False

    # 2. Inject LoRALinear into every transformer block's c_attn
    for block in model.transformer.h:
        block.attn.c_attn = LoRALinear(block.attn.c_attn, r=r, alpha=alpha, dropout=dropout)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return model, trainable_params, total_params


def extract_lora_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only the trainable LoRA adapter tensors (A and B parameters)."""
    return {k: v.clone() for k, v in model.state_dict().items() if k.endswith(".A") or k.endswith(".B")}


def load_lora_state(model: nn.Module, adapter_state: Dict[str, torch.Tensor]) -> None:
    """Load adapter tensors into a model with injected LoRALinear wrappers."""
    model.load_state_dict(adapter_state, strict=False)


def merge_lora_model(model: nn.Module) -> nn.Module:
    """
    Fold all LoRA adapters across transformer blocks back into their base weights,
    removing the wrapper and returning the clean, deployable base model.
    """
    for block in model.transformer.h:
        if isinstance(block.attn.c_attn, LoRALinear):
            block.attn.c_attn = block.attn.c_attn.merged()
    return model


# =====================================================================
# Milestone 2: Training Loop
# =====================================================================

def train_lora_adapter(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    samples: List[Dict[str, str]],
    epochs: int = 3,
    lr: float = 1e-3,
    batch_size: int = 4,
    device: str = "cpu"
) -> List[float]:
    """
    Train only the injected LoRA adapter matrices on the prompt-completion samples.
    Loss is calculated on the language modeling objective (causal LM next-token prediction).
    """
    model.to(device)
    model.train()

    # Pass ONLY trainable parameters (the 12 adapter tensors) to AdamW
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # Prepare inputs
    texts = [s["full_text"] for s in samples]
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    # Mask padding tokens in the labels with -100 so they are ignored by CrossEntropyLoss
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    dataset_size = input_ids.size(0)
    loss_history = []

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        epoch_loss = 0.0
        batches = math.ceil(dataset_size / batch_size)

        for b in range(batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            b_ids = input_ids[idx]
            b_mask = attention_mask[idx]
            b_labels = labels[idx]

            optimizer.zero_grad()
            outputs = model(input_ids=b_ids, attention_mask=b_mask, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, batches)
        loss_history.append(avg_loss)

    model.eval()
    return loss_history


# =====================================================================
# Text Generation Utility
# =====================================================================

def generate_recommendation(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    interest: str,
    destination: str = CITY_NAME,
    max_new_tokens: int = 70,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    device: str = "cpu"
) -> str:
    """Generate a travel guide recommendation given an interest and destination."""
    model.eval()
    model.to(device)
    prompt = format_prompt(interest, destination)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the completion following the prompt
    if prompt in full_output:
        completion = full_output.split(prompt, 1)[-1].strip()
    else:
        completion = full_output.strip()

    return completion


# =====================================================================
# Milestone 4: Evaluation Pipeline
# =====================================================================

def evaluate_recommendations(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    test_cases: Optional[List[Dict[str, Any]]] = None,
    destination: str = CITY_NAME
) -> Dict[str, Any]:
    """
    Evaluate the quality, relevance, and personalization of generated recommendations.
    Metrics evaluated:
    1. Keyword Relevance: Checks if expected domain keywords appear in the generated text.
    2. Distinct-2 Diversity: Measures the proportion of unique bigrams across generated texts.
    3. Length & Coherence: Average token length and non-emptiness.
    """
    if test_cases is None:
        test_cases = AETHELGARD_DATASET

    results = []
    keyword_hits = 0
    all_bigrams = []
    total_tokens = 0

    for item in test_cases:
        gen = generate_recommendation(model, tokenizer, item["interest"], destination=destination)
        gen_lower = gen.lower()

        # Check keyword presence
        matched = [kw for kw in item.get("keywords", []) if kw in gen_lower]
        is_relevant = len(matched) > 0
        if is_relevant:
            keyword_hits += 1

        tokens = gen_lower.split()
        total_tokens += len(tokens)
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        all_bigrams.extend(bigrams)

        results.append({
            "interest": item["interest"],
            "generated": gen,
            "matched_keywords": matched,
            "is_relevant": is_relevant,
            "length": len(tokens)
        })

    num_cases = len(test_cases)
    relevance_rate = (keyword_hits / num_cases) * 100.0 if num_cases > 0 else 0.0
    distinct_2 = (len(set(all_bigrams)) / max(1, len(all_bigrams))) * 100.0 if all_bigrams else 0.0
    avg_length = total_tokens / max(1, num_cases)

    return {
        "relevance_score": relevance_rate,
        "distinct_2_diversity": distinct_2,
        "average_token_length": avg_length,
        "sample_evaluations": results
    }


# =====================================================================
# Milestone Stubs (Called by tests and external runner)
# =====================================================================

def milestone_1() -> Dict[str, Any]:
    """Prepare a small dataset of travel interests and corresponding local recommendations."""
    samples = get_training_samples(multiplier=2)
    return {
        "dataset_name": f"{CITY_NAME}_Travel_Interests",
        "total_categories": len(AETHELGARD_DATASET),
        "total_samples": len(samples),
        "city": CITY_NAME,
        "samples": samples
    }


def milestone_2(model_name: str = "distilgpt2", r: int = 8, alpha: float = 16.0) -> Dict[str, Any]:
    """Implement LoRA adapter injection and training on a pre-trained LLM (GPT-2)."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model, trainable_params, total_params = inject_lora(model, r=r, alpha=alpha)

    # Perform lightweight verification training
    samples = get_training_samples(multiplier=1)[:4]
    loss_history = train_lora_adapter(model, tokenizer, samples, epochs=2, batch_size=2)
    adapter_state = extract_lora_state(model)

    return {
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_percentage": (trainable_params / total_params) * 100.0,
        "loss_history": loss_history,
        "adapter_tensor_count": len(adapter_state),
        "adapter_state": adapter_state
    }


def milestone_3() -> Dict[str, Any]:
    """Develop a Streamlit UI to input interests and destination, generating a mini-guide."""
    main_py_path = Path(__file__).parent / "main.py"
    assert main_py_path.exists(), "app/main.py must exist"
    content = main_py_path.read_text(encoding="utf-8")
    assert "st.title" in content, "Streamlit app must define title"
    return {
        "ui_file": str(main_py_path),
        "status": "ready",
        "has_streamlit": "streamlit" in content
    }


def milestone_4() -> Dict[str, Any]:
    """Evaluate the relevance and appeal of the generated recommendations."""
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    eval_metrics = evaluate_recommendations(model, tokenizer, test_cases=AETHELGARD_DATASET[:3])
    return eval_metrics


def milestone_5(model_name: str = "distilgpt2") -> Dict[str, Any]:
    """Implement adapter merging for inference."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model, _, _ = inject_lora(model, r=8, alpha=16.0)

    # Verify adapter is injected
    assert any(isinstance(block.attn.c_attn, LoRALinear) for block in model.transformer.h)

    # Merge adapter back into base
    merged_model = merge_lora_model(model)

    # Verify LoRALinear wrapper is gone and replaced by base Conv1D
    is_clean = all(isinstance(block.attn.c_attn, Conv1D) for block in merged_model.transformer.h)
    assert is_clean, "All LoRALinear wrappers must be merged and converted back to Conv1D"

    return {
        "merged": True,
        "layer_type": type(merged_model.transformer.h[0].attn.c_attn).__name__,
        "ready_for_deployment": True
    }
