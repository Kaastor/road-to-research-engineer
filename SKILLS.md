# road-to-research-engineer

**Legend:** **\[Must]** = core for the role • **\[Strong]** = highly valuable • **\[Awareness]** = be able to discuss/triage

---

# Core modeling & math

* **Foundations (LA/Stats/Opt) — \[Must].**
  Understand matrix calculus, distributions, estimation, and optimization so you can reason about losses, regularization, and convergence and design fair evaluations.
  **SLM focus:** small-compute scaling intuition; prefer RMSNorm; mind RoPE θ-scaling; consider **μP/μTransfer** for width transfer; regularization that matters at small scale (label smoothing, dropout schedules, stochastic depth).
  *Stack:* NumPy + SciPy (numerics only).
  *Stack add (SLM):* μP/μTransfer (**Awareness**); simple scaling-law sims in **NumPy**.

* **Paper → code & reproduction — \[Must].**
  Turn papers into clean, testable code; match reported results; document gaps and ablations.
  **SLM focus:** distillation, quantization, linear/SSM attention, and tiny baselines (≤300M) before 1–3B; keep a small, fast eval harness locally.
  *Stack:* **PyTorch** + **Hugging Face Transformers/Datasets**.
  *Stack add (SLM):* **lm-eval-harness** with a **GSM8K-mini** pack.

---

# Programming & frameworks

* **Python mastery — \[Must].**
  Write idiomatic, maintainable code; vectorize; profile hotspots; respect interfaces and typing.
  **SLM focus:** optimize CPU paths (tokenization, mmap iterators), and use fast I/O for dataset prep.
  *Stack:* **Python 3.11**, **NumPy**, **polars**, **Pydantic v2**, **mypy**, **ruff**, **cProfile**.
  *Stack add (SLM):* **memory-profiler**.

* **Deep learning stacks — \[Must].**
  Be fluent in PyTorch; have working familiarity with JAX/Flax to read/port code.
  **SLM focus:** target single-GPU/CPU and edge runtimes.
  *Stack:* **PyTorch**, **torchmetrics**; JAX/Flax (**Awareness**).
  *Stack add (SLM):* **llama.cpp (gguf)** for edge/CPU; **FlashAttention** as the single fused-attention backend (if GPU).

* **Structured I/O & tool use — \[Must].**
  Constrained decoding + function calling for production-grade outputs.
  *Stack:* **Outlines** (grammar-based decoding) + **Pydantic v2** (JSON Schema validation).
  *Awareness:* OpenAI-compatible tool spec.

* **Agentic patterns — \[Strong].**
  Planning, tool chaining, scratchpad use—even for SLMs.
  *Stack:* **LangGraph** (light ReAct-style traces & budgeted planners).

---

# Experiment design & evaluation

* **Baselines & ablations — \[Must].**
  Establish strong baselines, change one thing at a time, and report confidence intervals.
  **SLM focus:** fix a strict *compute budget* (e.g., one A100 for 24–72h); optimize *tokens seen* and data quality before params; track *tokens/param* and *loss vs tokens* curves.
  *Stack:* **Hydra** (configs), **Optuna** (sweeps), **SciPy** (CIs).

* **RAG eval as a product metric — \[Must].**
  Measure end-to-end answer quality and latency; relate retrieval hits to final outcomes.
  *Metrics:* Hit\@k vs answer quality, nDCG/MRR, hallucination rate under retrieval miss; budgeted latency.
  *Stack:* **lm-eval-harness** hooks + **custom E2E harness**, plots with **Matplotlib**.

* **Error analysis — \[Strong].**
  Build error taxonomies, slice by cohort, and tie failure modes to fixes.
  **SLM focus:** slice by prompt length, rare-token rate, domain, and RAG hit/miss (if using retrieval).
  *Stack:* **polars**, **Matplotlib**, **sklearn.metrics**.
  *Stack add (SLM):* **wordfreq**/**textstat** (**Awareness**).

* **Safety & robustness evals — \[Must].**
  Separate from quality metrics; gate releases.
  *Add:* jailbreak tests, toxicity/privacy checks, prompt-leak probes, canary token tests, PII detectors.
  *Stack:* simple **toxicity classifier** + **regex-based PII detector**; **AdvBench/XSTest** (**Awareness**).

* **Stats & A/B basics — \[Strong].**
  Know significance, power, and common pitfalls; design online tests with guardrails.
  **SLM focus:** paired bootstrap on identical prompt sets (small effects at small scale).
  *Stack:* **StatsModels**.

---

# Data & preprocessing

* **Curation & quality — \[Must].**
  Collect/label, dedupe, filter, and audit for bias; keep provenance.
  **SLM focus:** dense, high-quality sources; small teacher-generated synthetic with strict filters; **de-dup synthetic data against evals** and enforce per-generator safety filters.
  *Stack:* **Hugging Face Datasets** + **text-dedup** (MinHash/LSH).
  *Stack add (SLM):* **ftfy**, **langdetect**.

* **Formats & loaders — \[Must].**
  Stream large datasets efficiently; avoid Python bottlenecks.
  **SLM focus:** pre-pack token sequences (pack multiple short samples per block) and use mmap for zero-copy reads.
  *Stack:* **Parquet/Arrow**, **fsspec**, **PyTorch DataLoader**, **mmap**.
  *Stack add (SLM):* token-packed **Arrow** shards on local NVMe.

* **Preprocessing pipelines — \[Must].**
  Tokenize and transform reliably for text (and others as needed).
  **SLM focus:** vocab 16–32k; robust byte coverage; **freeze tokenizer early**; store and log vocab hash with runs.
  *Stack:* **sentencepiece** (tokenizer).

---

# Training systems

* **Training loops — \[Must].**
  Modular, restartable loops with clear hooks, metrics, and checkpoints.
  **SLM focus:** curriculum (short→long), loss-aware sampling, clean SFT/distillation loops.
  *Stack:* **PyTorch + Accelerate**, **TensorBoard/W\&B** (choose one—see Tracking).
  *Stack add (SLM):* **safetensors-only** checkpoints; token cursor saved with sampler/optimizer state.

* **Distributed training — \[Strong].**
  Scale across GPUs/nodes; understand communication patterns and memory partitioning.
  **SLM focus:** prefer simple data parallel; use **FSDP ZeRO-2/3** only >2B params or tight VRAM; consider CPU/NVMe offload for 2–3B with I/O care; checkpoint sharding + async uploads.
  *Stack:* **PyTorch FSDP** (NCCL).

* **Parallelism strategies — \[Strong].**
  Choose data/tensor/pipeline parallelism; shard states and checkpoints safely.
  **SLM focus:** avoid pipeline unless necessary; rely on grad checkpointing + accumulation.

* **Optimizers & numerics (small-scale hygiene) — \[Must].**
  Defaults that converge fast and stay stable at small widths.
  *Stack:* **AdamW**, cosine/OneCycle LR (pick **cosine**), LR-range tests, optional EMA, gradient-noise-scale checks.

---

# Performance & efficiency

* **Profiling & kernels (practical) — \[Strong].**
  Find bottlenecks and eliminate them before scaling.
  **SLM focus:** optimize decode path first (KV cache, fused RMSNorm/MatMul); on CPU, tune thread pools and batch size.
  *Stack:* **PyTorch Profiler**; single fused-attention backend: **FlashAttention**.

* **Throughput & memory — \[Must].**
  Use mixed precision and memory-saving techniques without destabilizing training.
  **SLM focus:** train **bf16/fp16** + grad checkpointing; infer with weight-only INT4/8; consider KV-cache quant/paging.
  *Stack:* **AMP (bf16)**, **gradient checkpointing**, **gradient accumulation**.
  *Stack add (SLM):* **AutoAWQ** (weight-only INT4) and log quantization choices.

* **Sequence-length scaling hygiene — \[Strong].**
  Long-context at small widths needs care.
  *Add:* RoPE θ/NTK-aware scaling, attention sinks; sanity-check with **LongBench**/**L-Eval** (**Awareness**).

---

# Reproducibility & experiment management

* **Determinism & versioning — \[Must].**
  Make runs repeatable across machines and time.
  **SLM focus:** pin tokenizer/version; hash packed-id datasets; log data-mixture YAML and prompt templates.
  *Stack:* seeding (random/cuda/cudnn), env pinning with **uv**, **Docker**, **W\&B Artifacts**, **Git** tags.

* **Tracking & sweeps — \[Must].**
  Centralize metrics, artifacts, and configs; automate hyperparameter search.
  **SLM focus:** always log tokens consumed, effective batch size, LR schedule, and teacher model/version for distillation.
  *Stack:* **Weights & Biases** + **Optuna**.

* **Release hygiene — \[Strong].**
  Provide Model/Data Cards, a contamination report snippet, and an eval reproducibility seed pack.

---

# Software engineering

* **Git hygiene — \[Must].**
  Clear branches, small PRs, code review discipline, meaningful messages.
  **SLM focus:** keep `/recipes` for pretrain/SFT/distill/RAG with `README.md + config.yaml` per recipe.
  *Stack:* **Git**, **pre-commit** with **ruff**, GitHub PRs, **CODEOWNERS**.

* **Testing & CI — \[Must].**
  Unit/integration tests for data, models, and training loops; run on every PR.
  **SLM focus:** golden-prompt tests (dozen prompts with expected ranges) and unit tests for token packing + quantized loaders.
  *Stack:* **pytest**, **coverage.py**, **GitHub Actions**.

---

# Orchestration & fault tolerance  *(merged)*

* **Schedulers & clusters — \[Strong].**
  Launch/manage jobs on shared compute; handle quotas and queues; optimize for cheap bursts (spot GPUs, Apple Silicon dev boxes); shard by data; frequent checkpoints.
  *Stack:* **SLURM** (sbatch/srun).

* **Fault tolerance & elastic/preemptible workflows — \[Must].**
  Design for preemption and failure; keep progress and resume safely.
  **SLM focus:** resume by token cursor; object-store checkpoints; idempotent data steps; short shards + frequent checkpoints; shard `index.json`; async uploads; SIGTERM handlers.
  *Stack:* **safetensors** + sharded checkpoints with `index.json`, S3/GS/Azure object storage, **tenacity** (retry/backoff), job arrays, **TorchElastic** (SLURM).

---

# Inference & serving

* **Prototype APIs — \[Must].**
  Stand up low-latency inference services with batching and streaming.
  **SLM focus:** CPU-first with small batch + speculative decoding (draft model); streaming via SSE.
  *Stack:* **FastAPI + Uvicorn**, SSE; simple in-process batcher (or server below).

* **Structured I/O integration — \[Must].**
  Enforce JSON/typed outputs; schema-first prompts; function calling.
  *Stack:* **Outlines** + **Pydantic v2** (JSON Schema).

* **Retrieval, embeddings & reranking — \[Must].**
  Treat retrieval quality as a first-class concern.
  *Stack:* **FAISS**; **bge-small** retriever; **bge-reranker-base** (tiny cross-encoder).
  *Datasets:* **BEIR** (plus in-house); **LanceDB** eval sets (**Awareness**).
  *Metrics:* Recall\@k, nDCG, MRR.

* **Export & compilers — \[Strong].**
  Convert models to portable, optimized runtimes.
  **SLM focus:** target edge: ONNX Runtime (CPU/Mobile); publish **gguf** for llama.cpp.
  *Stack:* **ONNX + ONNX Runtime**; **gguf** export.

* **LLM serving essentials — \[Must].**
  Efficient long-context decoding and caching.
  **SLM focus:** low-memory KV caches, sliding-window attention, paged KV; **vLLM** for server, **llama.cpp** for edge.
  *Stack:* **vLLM**, **tiktoken** (tokenization), speculative/parallel decoding (**Awareness**).

* **On-device measurement & compile paths — \[Strong].**
  Make “edge-ready” measurable.
  *Add:* **ONNX Runtime Mobile AOT** graphs; cold vs warm start; simple battery/CPU harness.

---

# Compression & deployment optimization

* **Smaller/faster with minimal quality loss — \[Must].**
  Choose the right mix of quantization, distillation, and pruning; measure tradeoffs.
  **SLM focus (pipeline):** Distill → **LoRA** (PEFT) → **INT4 AutoAWQ** → (optional) 2:4 sparsity → KV-cache quant; re-evaluate after each step.
  *Stack:* **LoRA** (PEFT), **AutoAWQ** (INT4), standard KD in **PyTorch**.

* **QAT & FP8 paths — \[Awareness]/\[Strong] (hw-dependent).**
  QAT for edge accelerators when available; FP8 with **TransformerEngine** (**Awareness**).

---

# Observability & resilience  *(merged)*

* **Metrics, logs, traces + safety nets — \[Must].**
  Know what’s happening in training and serving—at a glance—and protect latency SLOs.
  **SLM focus:** track TTFT, tokens/s (CPU/GPU), peak RSS, energy/battery; timeouts, circuit breakers, and rate limits; correlation IDs.
  *Stack:* **Prometheus** (metrics), **Grafana** (dashboards), **OpenTelemetry** (traces), **structlog** (structured logs), **pybreaker** (circuit breaker), **Envoy** (gateway & rate limiting), **Redis** (token-bucket RL).

* **Resilient rollouts — \[Strong].**
  Canary by device class (x86/ARM) and quantization level (fp16/int8/int4); keep fallback routes (RAG answer or higher-tier model).
  *Stack:* **Argo Rollouts + Argo CD** (serving on Kubernetes).

---

# Security, privacy & responsible AI (essentials)

* **Protect users and data — \[Must].**
  Handle PII correctly, lock down credentials, and sanity-check for obvious harms/biases.
  **SLM focus:** highlight on-device inference to reduce data egress; maintain jailbreak/safety prompt lists tuned for short contexts.
  *Stack:* secrets via **Vault**, IAM/RBAC, baseline PII scrubbing (**regex**), lightweight red-team scripts/checklists.

---

# Cost & SLO thinking

* **Design to budgets and targets — \[Must].**
  Track \$/1k requests, GPU-hours, p50/p95/p99 latency, and uptime; make tradeoffs explicit.
  **SLM focus:** set latency & memory budgets per device; optimize for CPU cost and battery on mobile; expose model “build flavors” (fp16/int8/int4).
  *Stack:* Cloud billing dashboard, **W\&B** for cost/latency metrics, **Grafana** SLO panels.

---

# Recommended

* **Calibration & robustness — \[Strong].**
  Report reliability (ECE/Brier), stress/OOD tests, and uncertainty; use bootstrap CIs.
  **SLM focus:** temperature/top-p calibration on short prompts; paired bootstrap on fixed prompt suites.
  *Stack:* **sklearn.calibration**, **torchmetrics**, **NumPy/SciPy** (bootstrap), **Matplotlib** (reliability plots).

* **Stability & dynamics — \[Strong].**
  Choose sane inits/normalizations; watch gradient norms/variance and activations; catch instabilities early.
  **SLM focus:** RMSNorm + cosine/OneCycle LR; clip by value or norm; monitor activation stats closely at small widths.
  *Stack:* **PyTorch** (init/norm), autograd checks, hooks, **W\&B/TensorBoard** histograms (pick **W\&B**), **torch.compile** (sanity/perf checks).

* **Interpretability & probing (practical) — \[Strong].**
  Use logit lens, linear probes, simple neuron/head ablations to diagnose regressions and distillation artifacts.

* **Licensing & contamination checks — \[Must].**
  Track data provenance/licenses and detect train–eval leakage.
  **SLM focus:** extra scrutiny when distilling from proprietary teachers; log teacher/version and filtering rules.
  *Stack:* HF Dataset/Model Cards, SPDX tags, **scancode-toolkit**/**OSS Review Toolkit** (**Awareness**), **datasketch/simhash** for overlap, lm-eval contamination scripts (**Awareness**).

* **Hardware awareness (practical) — \[Strong].**
  Understand interconnects/topology and I/O limits; fix input pipeline bottlenecks; read GPU profiles.
  **SLM focus:** CPU vectorization (AVX2/AVX-512/NEON), NUMA pinning; small-batch decode profiling.
  *Stack:* **nvidia-smi**, **Nsight Systems**, **PyTorch Profiler**, **iostat/nvme-cli/fio** (disk), **numactl** (NUMA).

* **Multilingual & code tracks — \[Strong].**
  Add evals for multilingual (XNLI/FLORES) and code (HumanEval-Plus/MBPP-lite); follow **pass\@k** discipline.

* **Elastic & preemptible patterns — \[Strong].**
  (Covered in Orchestration) Short shards + frequent checkpoints; elastic launches; object storage resumes.

---

## SLM playbook add-ons

* **Architectures to try — \[Strong]:** GQA/MQA, **RMSNorm everywhere**, downscaled **SwiGLU** FFN, **RoPE θ scaling**.
  *Awareness:* SSMs (Mamba), RWKV/Hyena, tiny MoE.

* **Distillation recipes — \[Must]:** token/logit distill; **short-rationale** distill for reasoning; **DPO** (preferred) with small RMs or heuristic rewards; verify-then-generate.

* **RAG-first design — \[Must]:** small model, strong retrieval/rerank (**FAISS + bge-small + bge-reranker-base**); concise prompts; admit uncertainty.

* **Tokenizer strategy — \[Must]:** start **32k**; test **16k** for edge; audit rare-token rates by domain (code vs natural language); **freeze early and log vocab hash**.

* **Edge & browser — \[Strong]:** CPU **llama.cpp**; ship fp16/int8/int4 builds + device sniffing.

* **Benchmarks that matter — \[Must]:** GSM8K (few-shot), ARC-Easy, HellaSwag, MT-Bench-lite, HumanEval\@1, MBPP; **plus** XNLI/FLORES (multi). Always report latency & memory.

* **Training heuristics — \[Strong]:**

  * ≤100M: pretrain 10–50B high-quality tokens; optional small SFT.
  * 300M–1B: start from a good checkpoint; domain SFT + distill; 1–5B more tokens.
  * 1–3B: distill + RAG; modest continued pretrain on target domain.

* **PEFT — \[Must]:** **LoRA** with rank search; merge + re-quantize for deployment; per-domain adapters (few MB).

---

## Portfolio tasks (interview-ready)

1. **Reproduce a ≤300M SLM — \[Must].**
   Train on 10–20B high-quality tokens with crisp ablations; export **gguf** and serve on **llama.cpp**; deliver latency/memory plots.

2. **Reasoning distillation — \[Strong].**
   Distill from a 7–8B teacher into a \~1B student with **short rationales**; report GSM8K/ARC-Easy deltas and latency; include paired-bootstrap CIs.

3. **RAG-first demo — \[Must].**
   **FAISS + bge-small** retriever + **bge-reranker-base**; end-to-end eval (hit\@k ↔ answer quality), retrieval-miss hallucination analysis; structured JSON outputs with schema validation.

---
