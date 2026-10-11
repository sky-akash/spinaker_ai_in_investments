# AI in Quantitative Investing
## Artificial Intelligence for Algorithmic Investment Strategies

**A One-Day Intensive Course — 4 × 60 Minutes**

**Instructor:** dr hab. Robert Ślepaczuk, prof. UW  
Head of Dept. of Quantitative Finance and Machine Learning &
Head of Quantitative Finance Research Group (QFRG)  
Faculty of Economic Sciences, University of Warsaw  
✉ rslepaczuk@wne.uw.edu.pl · [LinkedIn](https://www.linkedin.com/in/robert-%C5%9Blepaczuk-phd-2053036/) · [GitHub](https://github.com/robertslepaczuk) · [Google Scholar](https://scholar.google.pl/citations?user=9hDUxqwAAAAJ&hl=pl) · [ORCID](https://orcid.org/0000-0001-5227-2014)

---

## Course Overview

| Session | Title | Notebook |
|---------|-------|----------|
| Session 1 (60 min) | From Efficient Markets to Algorithmic Strategies | `01_wfo_backtest.ipynb` |
| Session 2 (60 min) | Machine Learning & Deep Learning for Return Forecasting | `02_lstm_transformer.ipynb` |
| Session 3 (60 min) | Deep Reinforcement Learning & Portfolio Optimisation | `03_drl_portfolio.ipynb` |
| Session 4 (60 min) | Informer for High-Frequency Bitcoin Trading | slides only (`slides/`) |

---

## Repository Structure

```
ai-investments-course-2026/
├── slides/
│   ├── main.tex              # Main Beamer document
│   ├── session1.tex          # Session 1 slides
│   ├── session2.tex          # Session 2 slides
│   ├── session3.tex          # Session 3 slides
│   ├── session4.tex          # Session 4 slides
│   └── presentation_*.pdf    # Compiled PDF (timestamped)
├── notebooks/
│   ├── 01_wfo_backtest.ipynb       # Session 1: WFO & MADL
│   ├── 02_lstm_transformer.ipynb   # Session 2: LSTM & Transformer
│   └── 03_drl_portfolio.ipynb      # Session 3: DRL portfolio
├── requirements.txt          # Python dependencies
└── pyproject.toml            # Project metadata
```

---

## Quick Start

### Step 1 — Clone the repository
```bash
git clone https://github.com/Spinaker-AI-in-Investments/ai-investments-course-2026.git
cd ai-investments-course-2026
```

### Step 2 — Create a virtual environment
```bash
python -m venv .venv
```

### Step 3 — Activate the environment
- **Windows:** `.venv\Scripts\activate`
- **Mac / Linux:** `source .venv/bin/activate`

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Open in VS Code
```bash
code .
```
Then: `Ctrl+Shift+P` → *Python: Select Interpreter* → choose `.venv`

### Step 6 — Run notebooks
Open `notebooks/01_wfo_backtest.ipynb` and select the `.venv` kernel.  
Run cells from top to bottom. Each session has its own notebook — follow the road map below.

### Compute device (notebooks 02 and 03)

Those two notebooks train small neural networks. They use CUDA when an NVIDIA
GPU is present, and CPU otherwise — including on Apple Silicon Macs.

That is deliberate. Apple's MPS backend works, but it accumulates
floating-point operations in a different order than the CPU, so the same
notebook produces slightly different numbers on a Mac than on a PC. For course
material, everyone seeing the same output matters more than a few seconds saved
on models this small.

If you do want the Mac GPU, opt in explicitly:

```bash
COURSE_USE_MPS=1 jupyter lab
```

---

## Step-by-Step Road Map

| Step | When | Action |
|------|------|--------|
| 0 | Before the course | Clone repo, create `.venv`, `pip install -r requirements.txt` |
| 1 | Session 1 | Open `notebooks/01_wfo_backtest.ipynb` · Run all cells top to bottom |
| 2 | Session 2 | Open `notebooks/02_lstm_transformer.ipynb` · Run all cells top to bottom |
| 3 | Session 3 | Open `notebooks/03_drl_portfolio.ipynb` · Run all cells top to bottom |
| 4 | Session 4 | Slides only — follow `slides/` (Informer / BTC-USDT); no notebook |

> **Note on computation time:**  
> Notebook 01: ~5–10 min (WFO grid search).  
> Notebook 02: ~20–40 min (LSTM/Transformer training, GPU recommended).  
> Notebook 03: ~30–60 min (DRL training, GPU strongly recommended).

---

## Notebook Summaries

### `01_wfo_backtest.ipynb` — Walk-Forward Optimisation & MADL

**What you will implement:**
- Dual moving-average momentum/contrarian strategy on SPY (2005–2024)
- Walk-Forward Optimisation (WFO) with IS/OOS splitting
- Performance metrics: CAGR, Sharpe, MDD, Calmar, IR*, IR**, IR***
- MADL and GMADL loss functions as directional evaluation metrics
- Transaction cost sensitivity analysis (0–100 bps)
- Statistical significance testing (SR t-test, Bonferroni correction)
- Double Out-of-Sample (Double OOS) validation protocol

**Key results:**
- WFO momentum significantly reduces MDD vs Buy & Hold
- MADL-optimised strategies show better directional accuracy
- Strategy degrades sharply above 10–20 bps transaction costs

**References:**
- Michańków, Sakowski & Ślepaczuk (2024). MADL. *Journal of Computational Science*, 81, 102375. [doi:10.1016/j.jocs.2024.102375](https://doi.org/10.1016/j.jocs.2024.102375)
- Mroziewicz & Ślepaczuk (2026). Double OOS. *arXiv*:2602.10785. [doi:10.48550/arXiv.2602.10785](https://doi.org/10.48550/arXiv.2602.10785)

---

### `02_lstm_transformer.ipynb` — LSTM & Transformer for Return Forecasting

**What you will implement:**
- LSTM model for Bitcoin daily return forecasting
- Transformer / Informer architecture for financial time series
- Feature engineering with purged walk-forward CV (no look-ahead)
- SHAP value computation for model explainability
- MADL-based training loss vs MSE comparison
- Hybrid volatility model: VIX + GARCH + LSTM

**Key pitfalls covered:**
1. Look-ahead bias in feature construction
2. Data leakage via standard k-fold CV
3. Non-stationarity and regime changes

**References:**
- Michańków, Sakowski & Ślepaczuk (2024). Generalized MADL (GMADL) — definition. *arXiv*:2412.18405. [doi:10.48550/arXiv.2412.18405](https://doi.org/10.48550/arXiv.2412.18405)
- Michańków, Sakowski & Ślepaczuk (2026). Alternative Loss Function in Evaluation of Algorithmic Investing Strategies Based on Transformer Models. *Pattern Recognition Letters*, 203, 170–176. [doi:10.1016/j.patrec.2026.03.007](https://doi.org/10.1016/j.patrec.2026.03.007)
- Kashif, K. & Ślepaczuk, R. (2025). LSTM-ARIMA as a Hybrid Approach in Algorithmic Investment Strategies. *Knowledge-Based Systems*, 320, 113563. [doi:10.1016/j.knosys.2025.113563](https://doi.org/10.1016/j.knosys.2025.113563)
- Płachta, K. & Ślepaczuk, R. (2025). Machine Learning for Daily Return Direction Forecasting: A Comparative Study with Explainable AI Insights. *SSRN*:5272156. [ssrn.com/abstract=5272156](https://ssrn.com/abstract=5272156)

---

### `03_drl_portfolio.ipynb` — Deep Reinforcement Learning for Portfolio Management

**What you will implement:**
- Markov Decision Process (MDP) formulation for portfolio management
- PPO (Proximal Policy Optimisation) agent for multi-asset allocation
- SAC (Soft Actor-Critic) for continuous portfolio weights
- WFO-based evaluation of DRL strategies
- Reward function design: risk-adjusted return vs reward hacking
- Comparison vs Markowitz + ML enhanced portfolio

**Key concepts:**
- State: portfolio weights, returns, volatility, macro features
- Action: target allocation vector (simplex constraint)
- Reward: log-return minus transaction cost penalty

**References:**
- Kashif & Ślepaczuk (2026). DRL for Diversified Portfolio Management. *arXiv*:2605.17307. [doi:10.48550/arXiv.2605.17307](https://doi.org/10.48550/arXiv.2605.17307)
- Ślusarczyk & Ślepaczuk (2025). ML-Enhanced Markowitz. *Journal of Big Data*, 12, 127. [doi:10.1186/s40537-025-01164-z](https://doi.org/10.1186/s40537-025-01164-z)

---

## Python Dependencies

| Package | Purpose |
|---------|---------|
| `numpy`, `pandas` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | ML utilities, preprocessing |
| `xgboost` | Gradient boosting |
| `shap` | Model explainability |
| `torch` | PyTorch — LSTM, Transformer, DRL |
| `yfinance` | Market data download |
| `scipy`, `statsmodels` | Statistical testing |
| `arch` | GARCH volatility models |
| `gymnasium` | RL environment interface |
| `stable-baselines3` | PPO, SAC implementations |

**Python version:** ≥ 3.11  
**GPU:** Strongly recommended for notebooks 02 and 03 (CUDA 11.8+)

---

## Building the Slides (LaTeX)

The slides are compiled from Beamer LaTeX source.

**Requirements:** TeX Live 2024+ or TinyTeX with `pdflatex`

```bash
cd slides
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # run twice for cross-references
```

The output is `main.pdf`. Timestamped copies are in `slides/presentation_YYYYMMDD_HHMM.pdf`.

---

## Funding and Affiliation

This course was developed under the [Spinaker](https://nawa.gov.pl/spinaker) programme funded by the Polish National Agency for Academic Exchange (NAWA) with co-financing from the European Social Fund Plus (FERS).

**Quantitative Finance Research Group (QFRG)**  
Department of Quantitative Finance and Machine Learning  
Faculty of Economic Sciences, University of Warsaw  
[wne.uw.edu.pl/QFRG](https://www.wne.uw.edu.pl/en/faculty/structure/departments/quantitative-finance)

---

## License

© 2026 Robert Ślepaczuk. All rights reserved. Course materials may be used for educational purposes with attribution.
