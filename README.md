# 🧭 Local Travel Guide Generator (LoRA from Scratch)

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/3kg47t9av2/badge.svg)](https://ask.thili.ai/b/3kg47t9av2)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-8%20passed-brightgreen.svg)]()
[![LoRA](https://img.shields.io/badge/LoRA-0.18%25%20trainable-purple.svg)]()

> **An end-to-end Local Travel Guide Generator fine-tuned with LoRA (Low-Rank Adaptation) from scratch on GPT-2.**
> Generates rich, personalized local travel itineraries and recommendations for a fictional destination (*Aethelgard*) while keeping 99.82% of the base model frozen and enabling zero-latency adapter merging.

---

## 🌟 Key Highlights & Engineering Achievements

- **LoRA Implemented Completely from Scratch**: No dependencies on external PEFT libraries (`peft`). Built directly using pure PyTorch tensors, following the foundational principles established in **Hu et al. (2021)**:
  $$\Delta W = B \cdot A, \quad A \in \mathbb{R}^{r \times d_{in}}, \quad B \in \mathbb{R}^{d_{out} \times r}, \quad r \ll d$$
- **Extreme Parameter Efficiency**:
  - Base Pretrained Model (`distilgpt2`): **81,912,576 parameters** (100% frozen).
  - Trainable Adapter ($r=8$, $\alpha=16$ across 6 attention blocks): **147,456 parameters** (**0.18%** of total).
  - Checkpoint storage footprint reduced from **~327 MB** down to **~613 KB** (>530x storage reduction).
- **Mathematical Invariants Guaranteed**:
  - **Zero-Start Invariant**: Matrix $A \sim \mathcal{N}(0, 0.01^2)$ and $B = \mathbf{0}$, ensuring $B \cdot A = \mathbf{0}$ at step 0 so fine-tuning begins without disturbing pretrained capabilities.
  - **Scaling Factor**: Output detour multiplied by $\frac{\alpha}{r}$ to decouple rank hyperparameter tuning from gradient magnitude.
  - **Zero-Overhead Inference Merging**: Supports permanently folding $\Delta W$ directly into base weights ($W' = W + \frac{\alpha}{r} (BA)$) with zero extra multiply-adds or latency overhead.
- **Production-Ready Streamlit UI**:
  - Interactive persona selector & custom prompt engineering.
  - In-browser live LoRA training & convergence visualization.
  - One-click adapter merging & live inference engine switching.
  - Quantitative quality evaluation dashboard measuring Keyword Relevance Hit-rate and Distinct-2 Diversity.

---

## 📐 Architecture & Math

### 1. Low-Rank Matrix Factorization
In standard GPT-2 attention blocks, query, key, and value projections are fused into a single linear projection `c_attn`:
$$x \in \mathbb{R}^{B \times L \times 768} \longrightarrow y \in \mathbb{R}^{B \times L \times 2304}$$

Our custom `LoRALinear` wraps `c_attn` with a parallel low-rank detour:
$$h(x) = \text{base}(x) + \left( \text{Dropout}(x) \cdot A^T \right) \cdot B^T \times \frac{\alpha}{r}$$

| Component | Dimensions | Number of Parameters (per layer) |
| :--- | :---: | :---: |
| **Base `c_attn` (Frozen)** | $768 \times 2304$ | $1,769,472$ |
| **Down-projection $A$** | $8 \times 768$ | $6,144$ |
| **Up-projection $B$** | $2304 \times 8$ | $18,432$ |
| **Total LoRA per Layer** | | **$24,576$** |
| **Total across 6 Layers** | | **$147,456$ (0.18%)** |

### 2. Zero-Latency Merging
During serving, keeping parallel detours adds $+1.39\%$ compute per token. By applying distributivity:
$$y = x \cdot W^T + \frac{\alpha}{r} (x A^T B^T) = x \left( W + \frac{\alpha}{r} B A \right)^T$$
Calling `merged()` folds $\Delta W$ directly back into the underlying weights, completely eliminating the wrapper:
```python
delta = (self.B @ self.A) * (self.alpha / self.r)
self.base.weight.add_(delta.T)  # Transposed for GPT-2 Conv1D
```

---

## 🎯 Dataset: The Fictional City of *Aethelgard*

A curated travel intelligence dataset containing comprehensive local itineraries across distinct traveler personas:
1. **Culinary & Street Food**: Sunken Spice Bazaars, saffron flatbreads, The Copper Kettle, cloudberry cider.
2. **Ancient History & Ruins**: Whispering Steps, Citadel of Sunken Kings, subterranean basalt codices, Marble Acropolis.
3. **Nature & Hiking**: Mist-Shrouded Ridge Trail, Emerald Falls, bioluminescent Hanging Arboretums, Heron Lagoon.
4. **Nightlife & Jazz**: Velvet Cellar bedrock jazz cavern, Starlight Rooftop Lounge, artisanal smoked cocktails.
5. **Art & Bohemian Culture**: Artisan Guild Alleys, Prism Contemporary Art Pavilion, glassblowing studios.
6. **Family & Leisure**: Seagull Promenade, clockwork carousel, Meadow Park wildflower gardens, Aqueduct tram.
7. **Adventure & Water Sports**: Corsair Point sea kayaking, tide caves, Iron Crest Via Ferrata cliff routes.
8. **Luxury & Wellness Spas**: Royal Thermal Baths, geothermal mineral springs, Celestine rooftop tasting menu.

---

## 🚀 Running the Project

### Prerequisites
- Python 3.11 or 3.12
- Git

### Quickstart
```bash
# 1. Clone repository
git clone https://github.com/YOUR-USERNAME/local-travel-guide-generator.git
cd local-travel-guide-generator

# 2. Create virtual environment & install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Launch interactive Streamlit Studio
streamlit run app/main.py

# 4. Run automated test suite
pytest
```

---

## 🧪 Automated Test Suite

All 5 milestones are covered by unit and integration tests in `tests/test_core.py` and `tests/test_smoke.py`:
- `test_milestone_1`: Verifies dataset schema, prompt formatting, and multi-category coverage.
- `test_milestone_2`: Verifies weight freezing, zero-start initialization invariant ($B=0$), parameter ratio (<0.5%), and training loop convergence.
- `test_milestone_3`: Verifies Streamlit UI configuration and file validity.
- `test_milestone_4`: Verifies quantitative evaluation metrics (Relevance hit rate, Distinct-2 diversity).
- `test_milestone_5`: Verifies mathematical equivalence of merged weights vs. unmerged adapter forward pass ($\text{diff} < 10^{-3}$).

Run with:
```bash
pytest -v
```

---

## 🏆 Project Milestones Completed
- [x] **Milestone 1**: Prepare structured dataset of travel interests & recommendations for fictional city (*Aethelgard*).
- [x] **Milestone 2**: Implement LoRA adapter injection and training loop on pre-trained LLM (`distilgpt2`).
- [x] **Milestone 3**: Develop an interactive Streamlit UI to input interests and destination, generating a mini-guide.
- [x] **Milestone 4**: Evaluate relevance, diversity, and appeal of generated recommendations.
- [x] **Milestone 5**: Implement adapter merging for zero-latency inference deployment.

---

## 📜 License
MIT License. Created for the Language Models from Scratch Drill.
