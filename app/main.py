"""Streamlit interactive UI for Local Travel Guide Generator with LoRA from scratch."""
import json
import os
from pathlib import Path
import time
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app import core

st.set_page_config(
    page_title="Aethelgard AI Guide — LoRA Travel Engine",
    page_icon="🧭",
    layout="wide"
)

# Custom Styling for Award-Winning Hackathon Presentation
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #60A5FA, #A78BFA, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .guide-box {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.9));
        border-left: 4px solid #818CF8;
        border-radius: 8px;
        padding: 20px;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧭 Aethelgard Local Travel Guide Generator")
st.markdown('<div class="sub-header">Powered by <b>LoRA from Scratch</b> (Low-Rank Adaptation) — 0.18% Trainable Parameters, 100% Intact Base Model</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# Caching Model & Tokenizer
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading base model & tokenizer...")
def get_base_assets():
    model_name = "distilgpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    return tok, base_model

tokenizer, base_model = get_base_assets()

# Session State for Training & Adapters
if "lora_model" not in st.session_state:
    st.session_state.lora_model = None
if "merged_model" not in st.session_state:
    st.session_state.merged_model = None
if "training_stats" not in st.session_state:
    st.session_state.training_stats = None
if "is_merged" not in st.session_state:
    st.session_state.is_merged = False

# -------------------------------------------------------------
# Sidebar: LoRA Configuration & Architecture Insights
# -------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ LoRA Architecture Controls")
    st.write("Tune low-rank hyperparameters directly:")
    
    lora_r = st.slider("LoRA Rank (r)", min_value=1, max_value=32, value=8, step=1,
                       help="Rank of the bottleneck matrices A and B.")
    lora_alpha = st.slider("LoRA Scaling (α)", min_value=1.0, max_value=64.0, value=16.0, step=1.0,
                           help="Scaling constant. Effective multiplier is α / r.")
    epochs = st.slider("Training Epochs", min_value=1, max_value=8, value=3)
    learning_rate = st.select_slider(
        "Learning Rate (AdamW)",
        options=[1e-4, 3e-4, 5e-4, 1e-3, 2e-3],
        value=1e-3,
        format_func=lambda x: f"{x:.0e}"
    )

    st.markdown("---")
    st.subheader("📊 Architectural Math")
    d_in = 768
    d_out = 2304  # fused Q, K, V
    params_per_layer = (d_in * lora_r) + (lora_r * d_out)
    total_lora_params = params_per_layer * 6
    total_base_params = sum(p.numel() for p in base_model.parameters())
    ratio = (total_lora_params / total_base_params) * 100.0

    st.metric(label="Trainable Parameters", value=f"{total_lora_params:,}", delta=f"{ratio:.2f}% of model", delta_color="inverse")
    st.metric(label="Memory Wall Reduction", value="~550x", help="Optimizer states & gradients stored for adapter only")
    st.metric(label="Zero-Start Invariant", value="B = 0 (step 0 no-op)")

# -------------------------------------------------------------
# Main Tabs: Generator, Trainer & Inspector, Benchmarks
# -------------------------------------------------------------
tab_guide, tab_train, tab_benchmark = st.tabs([
    "🗺️ Personalized Guide Generator",
    "⚡ LoRA Studio: Train & Merge",
    "📈 Evaluation & Metrics"
])

# =============================================================
# Tab 1: Personalized Guide Generator
# =============================================================
with tab_guide:
    st.subheader("Generate an Insider's Itinerary")
    st.write("Select a travel persona and destination to generate a customized local guide.")

    col1, col2 = st.columns([1, 1])
    with col1:
        destination = st.text_input("Destination City", value="Aethelgard", help="The fictional city from our curated dataset")
        interests_list = [item["interest"] for item in core.AETHELGARD_DATASET]
        selected_interest = st.selectbox("Tourist Interest / Persona", options=interests_list)
        custom_query = st.text_input("Or write custom interest", value="", placeholder="e.g. Secret underground cocktail speakeasies")

    with col2:
        st.write("🔧 **Inference Engine Mode**")
        active_mode = "Base Pretrained (Zero-shot)"
        if st.session_state.is_merged:
            active_mode = "Merged LoRA Model (Zero Overhead Inference)"
        elif st.session_state.lora_model is not None:
            active_mode = "Active LoRA Adapter (Injected Q/K/V Detour)"

        st.info(f"**Current Engine:** `{active_mode}`")
        temp = st.slider("Creativity (Temperature)", min_value=0.2, max_value=1.2, value=0.75, step=0.05)
        max_tokens = st.slider("Max New Tokens", min_value=30, max_value=120, value=75, step=5)

    interest_to_use = custom_query.strip() if custom_query.strip() else selected_interest

    if st.button("✨ Generate Personalized Guide", type="primary", use_container_width=True):
        # Choose active model
        if st.session_state.is_merged:
            model_to_run = st.session_state.merged_model
        elif st.session_state.lora_model is not None:
            model_to_run = st.session_state.lora_model
        else:
            model_to_run = base_model

        with st.spinner("Synthesizing local insider knowledge..."):
            t0 = time.time()
            recommendation = core.generate_recommendation(
                model_to_run,
                tokenizer,
                interest=interest_to_use,
                destination=destination,
                max_new_tokens=max_tokens,
                temperature=temp
            )
            elapsed = time.time() - t0

        st.markdown(f"""
        <div class="guide-box">
            <b>📍 Destination:</b> {destination} &nbsp;|&nbsp; <b>🎯 Focus:</b> {interest_to_use}<br><br>
            {recommendation}
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"⏱️ Generated in {elapsed:.2f}s | Mode: {active_mode}")

# =============================================================
# Tab 2: LoRA Studio (Train, Extract & Merge)
# =============================================================
with tab_train:
    st.subheader("LoRA Adapter Injection & In-Memory Training")
    st.write(
        "Train an adapter on the Aethelgard dataset right in your browser. "
        "Watch the loss fall as matrix B gently grows away from zero."
    )

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        if st.button("🚀 Train LoRA Adapter Now", use_container_width=True):
            with st.spinner(f"Injecting rank-{lora_r} adapter and training for {epochs} epochs..."):
                fresh_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
                model_with_lora, trainable_params, total_params = core.inject_lora(
                    fresh_model, r=lora_r, alpha=lora_alpha, dropout=0.05
                )

                samples = core.get_training_samples(multiplier=2)
                loss_hist = core.train_lora_adapter(
                    model_with_lora, tokenizer, samples, epochs=epochs, lr=learning_rate, batch_size=4
                )

                st.session_state.lora_model = model_with_lora
                st.session_state.is_merged = False
                st.session_state.training_stats = {
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "loss_history": loss_hist,
                    "r": lora_r,
                    "alpha": lora_alpha,
                    "epochs": epochs
                }
            st.success(f"Adapter trained! Final loss: {loss_hist[-1]:.4f}")

    with col_btn2:
        merge_disabled = (st.session_state.lora_model is None or st.session_state.is_merged)
        if st.button("⚡ Merge Adapter into Base (W' = W + B@A)", disabled=merge_disabled, use_container_width=True):
            with st.spinner("Folding B@A back into base weights..."):
                merged = core.merge_lora_model(st.session_state.lora_model)
                st.session_state.merged_model = merged
                st.session_state.is_merged = True
            st.success("Adapter successfully folded into W! Zero extra inference latency.")

    with col_btn3:
        if st.button("🔄 Reset to Base Model", use_container_width=True):
            st.session_state.lora_model = None
            st.session_state.merged_model = None
            st.session_state.is_merged = False
            st.session_state.training_stats = None
            st.info("Reset complete. Running base model.")

    if st.session_state.training_stats:
        stats = st.session_state.training_stats
        st.markdown("---")
        st.write("#### 📉 Training Convergence")
        st.line_chart(stats["loss_history"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trainable Weights", f"{stats['trainable_params']:,}")
        c2.metric("Frozen Weights", f"{stats['total_params'] - stats['trainable_params']:,}")
        c3.metric("Trainable Ratio", f"{(stats['trainable_params'] / stats['total_params']) * 100:.2f}%")
        c4.metric("Adapter File Size", "~613 KB")

# =============================================================
# Tab 3: Evaluation & Benchmarks
# =============================================================
with tab_benchmark:
    st.subheader("Automated Quality & Personalization Evaluation")
    st.write(
        "Evaluate recommendation relevance and domain specificity using keyword-hit "
        "relevance and Distinct-2 diversity metrics."
    )

    if st.button("🧪 Run Full Evaluation Suite", type="secondary"):
        active_eval_model = st.session_state.merged_model or st.session_state.lora_model or base_model
        with st.spinner("Benchmarking generated recommendations across all categories..."):
            eval_results = core.evaluate_recommendations(active_eval_model, tokenizer)

        m1, m2, m3 = st.columns(3)
        m1.metric("Relevance Hit Rate", f"{eval_results['relevance_score']:.1f}%", help="Proportion of recommendations containing domain keywords")
        m2.metric("Distinct-2 Diversity", f"{eval_results['distinct_2_diversity']:.1f}%", help="Proportion of unique bigrams across answers")
        m3.metric("Avg Token Length", f"{eval_results['average_token_length']:.1f} tokens")

        st.write("#### Detailed Category Audits:")
        st.dataframe(
            [
                {
                    "Interest": res["interest"],
                    "Keywords Matched": ", ".join(res["matched_keywords"]) if res["matched_keywords"] else "None",
                    "Relevant": "✅" if res["is_relevant"] else "⚠️",
                    "Generated Excerpt": res["generated"][:90] + "..."
                }
                for res in eval_results["sample_evaluations"]
            ],
            use_container_width=True
        )

# Footer
st.markdown("---")
st.caption("Built with PyTorch & Hugging Face Transformers. Implements LoRA (Hu et al., 2021) from scratch without external PEFT libraries.")
