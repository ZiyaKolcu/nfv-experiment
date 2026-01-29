# Nutrition Facts Vision (NFV) - Experimental Evaluation Framework

This repository contains the **benchmarking scripts, datasets, evaluation logic, and statistical analysis tools** developed for the research paper: *"Nutrition Facts Vision: An LLM-Powered Mobile System for Personalized Food Label Analysis and Risk Assessment."*

It serves as the scientific companion to the [NFV Mobile Application](https://github.com/ZiyaKolcu/nutrition-facts-vision-app), focusing strictly on the quantitative validation of the system's performance across different data quality conditions and Large Language Models (LLMs).

## 🧪 Project Overview

The goal of this framework is to reproduce the experimental results presented in the paper, specifically testing two core hypotheses and validating the results statistically:

* **Hypothesis 1 (H1 - Data Extraction):** Evaluating the ability of LLMs to correct noisy OCR text and extract structured nutritional data (JSON) under different input conditions.
* **Hypothesis 2 (H2 - Clinical Reasoning):** Measuring the "Cascading Quality Effect"—how upstream OCR quality impacts downstream personalized risk assessment (Safety Recall & Precision).
* **Statistical Validation:** Rigorous pairwise comparison of models (Efficient vs. High-Reasoning) using non-parametric tests to confirm significance.

## 📂 Repository Structure

The project is organized to isolate evaluation stages and statistical testing:

```bash
├── app/services/nutrition/    # Core LLM Clients (OpenAI, Gemini, Claude) & Logic
├── common/                    # Shared utilities for data loading and metric calculation
├── hypothesis_1/              # Scripts & Data for Stage 1 Evaluation (OCR Correction)
│   ├── merged_ground_truths.json  # 100% Human-verified reference dataset
│   ├── run_h1.py                  # Script to execute Stage 1 benchmarks
│   └── evaluate_h1.py             # Script to calculate Precision/Recall/F1
├── hypothesis_2/              # Scripts & Data for Stage 2 Evaluation (Risk Assessment)
│   ├── profiles.json              # Deterministic logic for Synthetic Health Profiles
│   ├── run_h2.py                  # Script to execute Stage 2 benchmarks
│   └── evaluate_h2.py             # Script to calculate Safety Recall & Conservative Precision
├── statistical_test/          # NEW: Statistical significance & Error analysis
│   ├── statistical_significance_test.py # Wilcoxon & Bootstrap analysis script
│   └── error_finder.py            # Utility to identify specific failure cases
├── ocr texts/                 # Input Data Sources (Experimental Conditions)
│   ├── raw_ocr_texts.json         # Condition 1: Baseline (Noisy On-Device OCR)
│   ├── structured_ocr_texts.json  # Condition 2: Method A (Heuristic Grouping)
│   └── cloud_vision_ocr_texts.json# Condition 3: Method B (High-Fidelity Cloud OCR)
└── requirements.txt           # Python dependencies

```

## ⚙️ Experimental Conditions

The framework evaluates performance across three distinct data quality tiers:

1. **Baseline:** Raw output from Google ML Kit (On-Device).
2. **Method A:** On-Device output processed with a custom geometric clustering heuristic.
3. **Method B:** Output from Google Cloud Vision API (Upper Bound).

## 🚀 Installation & Setup

1. **Clone the repository:**

```bash
git clone [https://github.com/ZiyaKolcu/nfv-experiment.git](https://github.com/ZiyaKolcu/nfv-experiment.git)
cd nfv-experiment

```

2. **Install dependencies:**

```bash
pip install -r requirements.txt

```

3. **Environment Configuration:**
Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant...

```

## 📊 Running the Experiments

### 1. Reproducing Stage 1 Results (OCR Correction)

To test how well models correct noisy text into structured JSON:

```bash
python hypothesis_1/run_h1.py

```

* **Output:** Generates raw JSON extractions in `hypothesis_1/results and evaluation/`.
* **Evaluation:** Run `evaluate_h1.py` afterward to compute F1 scores.

### 2. Reproducing Stage 2 Results (Risk Assessment)

To test the safety and reasoning capabilities of the models (dependent on H1 results):

```bash
python hypothesis_2/run_h2.py

```

* **Input:** Uses outputs from H1 to simulate the full pipeline.
* **Output:** Generates risk assessment logs in `hypothesis_2/results and evaluation/`.

### 3. Statistical Significance Analysis

To perform pairwise comparisons between "Efficient" (Flash/Haiku/Mini) and "High-Reasoning" (Pro/Sonnet/GPT-5.1) models:

```bash
python statistical_test/statistical_significance_test.py

```

**Methodology:**
This script executes the following for both H1 (Extraction) and H2 (Risk Assessment):

* **Wilcoxon Signed-Rank Test:** Non-parametric test for paired samples.
* **Bootstrap Confidence Intervals:** 10,000 resamples (CI=0.95) to estimate the stability of performance differences.
* **Output:** A console summary table and a CSV report (`statistical_significance_results.csv`) detailing p-values and significance levels (*** p<0.001).

**Model Pairs Tested:**

* **Anthropic:** Claude Haiku 4.5 vs. Claude Sonnet 4.5
* **Google:** Gemini 3 Flash vs. Gemini 3 Pro
* **OpenAI:** GPT-5-mini vs. GPT-5.1

## 🧠 Supported Models

The framework supports benchmarking the following model families:

* **OpenAI:** GPT-5-mini, GPT-5.1
* **Google:** Gemini 3 Flash, Gemini 3 Pro
* **Anthropic:** Claude 4.5 Haiku, Claude 4.5 Sonnet

## 📝 Datasets & Ground Truth

* **`merged_ground_truths.json`**: Contains verbatim transcriptions and normalized nutritional values for 50 Turkish food products.
* **`profiles.json`**: Logic rules for synthetic health profiles (e.g., Type 2 Diabetic, Nut Allergy) used to validate risk assessment.

## 📄 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@software{nfv_ocr_experiment,
  title={Nutrition Facts Vision: An LLM-Powered Mobile System for Personalized Food Label Analysis},
  author={Kolcu, Ziya},
  year={2025},
  url={[https://github.com/ZiyaKolcu/nfv_experiment_app](https://github.com/ZiyaKolcu/nfv_experiment_app)}
}

```

---

*This repository is intended for academic peer review and reproducibility purposes.*
