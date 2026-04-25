# Pi-Nexsus: A Lightweight LLM Post-Training Library for Algorithmic Trading

[![documentation](https://img.shields.io/badge/documentation-blue)](https://tunix.readthedocs.io/en/latest/index.html)

**Pi-Nexsus** is a JAX-based library designed to streamline the post-training of Large Language Models, with first-class support for algorithmic trading workflows and MQL (MetaQuotes Language) integration. It provides efficient and scalable support for:

- **SOTA Training performance on TPUs**
- **Supervised Fine-Tuning (SFT)**
- **Reinforcement Learning (RL)**
- **Agentic RL**
- **MQL Strategy Learning & BodyFail Database Integration**

Pi-Nexsus leverages the power of JAX for accelerated computation and seamless integration with JAX-based modeling frameworks like [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html), and integrates with high-performance inference engines like vLLM and SGLang-JAX for rollout.

**Current Status: V2 Release**

Pi-Nexsus is under active development. The team is actively working on expanding its capabilities, usability, and performance — including expanded MQL4/MQL5 support and BodyFail database schema coverage. Stay tuned for upcoming updates and new features!

---

## High Level Architecture

Pi-Nexsus serves as a state-of-the-art post-training library within the JAX training stack, positioned to leverage foundational tools like Flax, Optax, Orbax, etc. for efficient model refinement. It sits as an intermediate layer between these core utilities and optimized models like MaxText and MaxDiffusion, streamlining tuning workflows on top of the XLA and JAX infrastructure.

---

## Key Features

### Supervised Fine-Tuning (SFT)
- Full Weights Fine-Tuning
- PEFT — Parameter-Efficient Fine-Tuning (LoRA)
- DPO — Direct Preference Optimization
- ORPO — Odds Ratio Preference Optimization

### Reinforcement Learning (RL)
- PPO — Proximal Policy Optimization
- GRPO — Group Relative Policy Optimization
- GSPO-Token — Token-level Group Sequence Policy Optimization
- DAPO — Direct Alignment via Preference Optimization
- Dr.GRPO — Distributionally Robust GRPO

### Agentic RL
- Multi-turn tool use
- Asynchronous rollout for high-throughput trajectory collection
- Trajectory batching and grouping

### MQL Integration *(New in Pi-Nexsus)*
- Native MQL4/MQL5 strategy parsing and fine-tuning support
- **BodyFail Database** — a structured dataset of all-body candlestick failure signals for model training and backtesting
- MQL signal reward shaping for RL agents
- Integration with MetaTrader 4/5 environments for live rollout validation

---

## BodyFail Database

The `bodyfail` database is a new addition introduced in Pi-Nexsus. It captures and stores candlestick body failure events — instances where expected body-based breakout or reversal patterns fail to follow through — across all major trading instruments and timeframes.

### Schema Overview

```sql
-- bodyfail_events: core failure signal records
CREATE TABLE bodyfail_events (
    id            BIGINT PRIMARY KEY AUTO_INCREMENT,
    symbol        VARCHAR(20)     NOT NULL,
    timeframe     VARCHAR(10)     NOT NULL,   -- e.g. M1, M5, H1, D1
    open_time     DATETIME        NOT NULL,
    open          DOUBLE          NOT NULL,
    high          DOUBLE          NOT NULL,
    low           DOUBLE          NOT NULL,
    close         DOUBLE          NOT NULL,
    body_size     DOUBLE          NOT NULL,
    direction     ENUM('BULL','BEAR') NOT NULL,
    fail_type     VARCHAR(50)     NOT NULL,   -- e.g. "engulf_fail", "pin_fail"
    fail_score    FLOAT           NOT NULL,   -- model-assigned confidence [0,1]
    recorded_at   DATETIME        DEFAULT CURRENT_TIMESTAMP
);

-- bodyfail_labels: human/model annotated ground truth
CREATE TABLE bodyfail_labels (
    id            BIGINT PRIMARY KEY AUTO_INCREMENT,
    event_id      BIGINT          NOT NULL REFERENCES bodyfail_events(id),
    label         ENUM('TRUE_FAIL','FALSE_FAIL','AMBIGUOUS') NOT NULL,
    annotator     VARCHAR(100),
    annotation_ts DATETIME        DEFAULT CURRENT_TIMESTAMP
);

-- bodyfail_stats: aggregated failure stats per symbol/timeframe
CREATE TABLE bodyfail_stats (
    id            BIGINT PRIMARY KEY AUTO_INCREMENT,
    symbol        VARCHAR(20)     NOT NULL,
    timeframe     VARCHAR(10)     NOT NULL,
    total_events  INT             DEFAULT 0,
    true_fails    INT             DEFAULT 0,
    false_fails   INT             DEFAULT 0,
    fail_rate     FLOAT,
    updated_at    DATETIME        DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_symbol_tf (symbol, timeframe)
);
```

The BodyFail database covers **all body-type failure signals** including but not limited to: engulfing failures, pin bar reversals, inside bar breaks, and doji indecision failures — across all available symbols and timeframes.

---

## News

- **[2026/04]** Pi-Nexsus rebranded from pi-tunikapps. MQL BodyFail database introduced.
- **[2026/04]** Gemma4 models are supported. Stay tuned for upcoming MQL training recipes.
- **[2026/01]** Efficient kernel execution added (splash attention, GMM MoE).
- **[2025/12]** Agentic RL Training released with multi-turn agent-env interaction, tool usage, and async rollout.

---

## Framework & Infra Highlights

### Modularity
- Components are designed to be reusable and composable
- Easy to customize and extend for MQL strategy domains

### Performance & Efficiency
- Native vLLM and SGLang-JAX on TPU integration for performant rollout
- Native MaxText model integration for high-performance kernels
- Micro-batching support for component-level efficient execution

### Stability
- Seamless multi-host distributed training with Pathways (scales to thousands of devices)
- Checkpointing and Fault Tolerance

---

## Getting Started

**Installation:** Install Pi-Nexsus and run your first training job:

```bash
pip install pi-nexsus
```

For TPU users integrating `vllm` and `tpu-inference`, two setup paths are supported:

- **Docker image builds** — use the `Dockerfile` and install pinned dependencies from `requirements/requirements.txt` and `requirements/special_requirements.txt`.
- **Local TPU VM or developer-machine installs** — use `scripts/install_tunix_vllm_requirement.sh`, which installs the same requirement files outside Docker.

These are separate entry points. If you are building the Docker image, you do not need to run the install script inside the container build.

---

## Supported Models

Pi-Nexsus supports a growing list of models including Gemma, Llama, and Qwen families. Fine-tuned MQL-domain checkpoints are planned for upcoming releases.

---

## MQL File Structure

Pi-Nexsus introduces a new `mql/` directory at the repo root for all MetaQuotes Language assets:

```
mql/
├── bodyfail/
│   ├── bodyfail_collector.mq5      # EA to collect bodyfail events into DB
│   ├── bodyfail_labels_export.mq5  # Export labelled signals for training
│   └── bodyfail_stats.mq5          # On-chart stats dashboard indicator
├── signals/
│   └── nexsus_signal_bridge.mq5    # Bridge: model inference → MT5 signal
└── README.md
```

---

## Contributing and Feedback

Contributions are welcome! As Pi-Nexsus is in early development, the contribution process is still being formalized. You can make feature requests, report issues, and ask questions in the GitHub discussion forum.

For MQL-specific contributions (new failure pattern types, additional database schema tables, MT4/MT5 Expert Advisors), please open an issue with the `mql` label.

---

## Citing Pi-Nexsus

```bibtex
@misc{pi-nexsus2026,
  title={Pi-Nexsus: LLM Post-Training with MQL BodyFail Integration},
  author={Tsukimarf and contributors},
  year={2026},
  howpublished={\url{https://github.com/Tsukimarf/pi-nexsus}},
}
```

---

## Acknowledgements

Thank you to all contributors and to the original Tunix/Tune-in-JAX project by Google for the foundational architecture.
