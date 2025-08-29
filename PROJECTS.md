## Project List Overview
The following projects are designed to cover **all** skills from your skill list through hands-on implementation. There are 12 projects, prioritized to build progressively while showing overlaps (e.g., data curation in Project 1 feeds into training in Project 3; evaluation metrics from Project 2 recur in later ones). Early projects focus on foundations, mid ones on training/evaluation, and later on deployment/optimization. Each project references overlapping skills from prior ones without redundancy. Complete them in order for best learning, but they can be done modularly.
### Project 1: Data Curation Pipeline for SLM Pretraining
**Description:** Build a pipeline to collect, clean, deduplicate, and prepare a small high-quality dataset (e.g., 1-5GB of text from public sources like Wikipedia excerpts) for SLM pretraining, including provenance tracking and bias audits.
**Skills Learned:**
- **Python mastery (write idiomatic, maintainable code; vectorize; profile hotspots; respect interfaces and typing; optimize CPU paths like tokenization/mmap iterators and fast I/O).**
- Curation & quality (collect/label, dedupe, filter, audit for bias; provenance; dense high-quality sources; de-dup synthetic data; per-generator safety filters).
- Formats & loaders (stream large datasets; avoid bottlenecks; pre-pack token sequences; mmap for zero-copy).
- Preprocessing pipelines (tokenize/transform; vocab 16–32k; byte coverage; freeze tokenizer; log vocab hash).
- Determinism & versioning (pin tokenizer/version; hash datasets; log data-mixture YAML).
- Licensing & contamination checks (track provenance/licenses; detect leakage; log filtering rules).
- Git hygiene (clear branches, small PRs, meaningful messages; /recipes with README.md + config.yaml).
**Requirements:**
- Collect 1GB+ text data from 2+ sources.
- Implement deduplication and quality filters (e.g., remove low-quality sentences).
- Tokenize with a frozen 16k vocab and pack sequences.
- Audit for bias/PII and generate a dataset card.
- Hash dataset and log in YAML.
- Commit to Git with proper structure.
**Dependencies:** Hugging Face Datasets, text-dedup (MinHash/LSH), ftfy, langdetect, sentencepiece, Git, uv (for env pinning).
### Project 2: Simple SLM Evaluation Harness
**Description:** Create a local evaluation tool to test small models on benchmarks like GSM8K-mini and HellaSwag, including custom metrics and plots.
**Skills Learned:**
- Paper → code & reproduction (turn papers into code; match results; document gaps).
- Baselines & ablations (establish baselines; change one thing; report CIs; fix compute budget; track tokens/param and loss vs tokens).
- RAG eval as a product metric (measure answer quality/latency; hit@k vs quality; nDCG/MRR; hallucination rate).
- Error analysis (error taxonomies; slice by cohort; tie failures to fixes; slice by prompt length/rare-token/domain).
- Safety & robustness evals (jailbreak tests; toxicity/privacy checks; PII detectors).
- Stats & A/B basics (significance; power; pitfalls; paired bootstrap).
- Calibration & robustness (report ECE/Brier; stress/OOD tests; uncertainty; bootstrap CIs; temperature/top-p calibration).
- **Benchmarks that matter (GSM8K (few-shot), ARC-Easy, HellaSwag, MT-Bench-lite, HumanEval@1, MBPP; plus XNLI/FLORES for multilingual; always report latency & memory).**
**Requirements:**
- Integrate 3 benchmarks (GSM8K-mini, HellaSwag, ARC-Easy).
- Compute metrics like nDCG, hallucination rate, ECE with CIs via bootstrap.
- Slice errors by prompt length and domain.
- Run safety checks (regex PII, simple toxicity).
- Generate plots for loss curves and reliability.
- Document baselines in a report.
**Dependencies:** lm-eval-harness, polars, Matplotlib, sklearn.metrics, wordfreq/textstat, StatsModels, NumPy/SciPy, torchmetrics.
### Project 3: Train a Tiny SLM from Scratch
**Description:** Pretrain a ≤100M parameter SLM on your curated dataset from Project 1, using a simple architecture like downscaled Transformer with RMSNorm.
**Skills Learned:**
- Foundations (LA/Stats/Opt) (matrix calculus; distributions; estimation; optimization; small-compute scaling; RMSNorm; RoPE θ-scaling; μP/μTransfer awareness; regularization like label smoothing/dropout).
- **Deep learning stacks (be fluent in PyTorch; working familiarity with JAX/Flax for reading/porting code; target single-GPU/CPU and edge runtimes).**
- Training loops (modular/restartable; hooks/metrics/checkpoints; curriculum short→long; loss-aware sampling; SFT/distillation loops).
- Optimizers & numerics (AdamW; cosine LR; LR-range tests; EMA; gradient-noise checks).
- Stability & dynamics (sane inits/normalizations; watch gradients/activations; clip by value/norm; monitor stats; RMSNorm + cosine LR).
- Architectures to try (GQA/MQA; RMSNorm; downscaled SwiGLU FFN; RoPE θ scaling; SSMs awareness).
- Training heuristics (≤100M: pretrain 10–50B tokens; optional SFT).
**Requirements:**
- Implement model with RMSNorm and RoPE.
- Train on 10B+ tokens with curriculum.
- Use AdamW with cosine LR and monitor gradients.
- Checkpoint with safetensors.
- Evaluate using harness from Project 2.
- Ablate one hyperparam (e.g., dropout) with CIs.
**Dependencies:** PyTorch + Accelerate, NumPy + SciPy, safetensors, μP/μTransfer (for simulation), W&B (tracking).
### Project 4: Distill a 300M SLM from a Teacher
**Description:** Distill knowledge from a 7B teacher model (e.g., open-source like Llama-7B) into a 300M student using token/logit distillation on a subset of data from Project 1.
**Skills Learned:**
- Paper → code & reproduction (SLM focus: distillation/quantization; tiny baselines ≤300M).
- Distillation recipes (token/logit distill; short-rationale for reasoning; DPO with small RMs; verify-then-generate).
- PEFT (LoRA with rank search; merge + re-quantize).
- Training heuristics (300M–1B: start from checkpoint; domain SFT + distill; 1–5B tokens).
**Requirements:**
- Set up teacher-student pair.
- Implement logit distillation with short rationales.
- Fine-tune student with LoRA.
- Merge and evaluate deltas on GSM8K/ARC-Easy using Project 2.
- Report CIs via bootstrap.
- Document gaps in reproduction.
**Dependencies:** Hugging Face Transformers/Datasets, LoRA (PEFT), PyTorch, lm-eval-harness, Optuna (sweeps).
### Project 5: Build a RAG System for Q&A
**Description:** Create an end-to-end RAG pipeline using your distilled model from Project 4, with retrieval over a knowledge base (e.g., Wikipedia chunks), reranking, and structured outputs.
**Skills Learned:**
- Retrieval, embeddings & reranking (treat retrieval as first-class; FAISS; bge-small retriever; bge-reranker-base).
- RAG-first design (small model + strong retrieval; concise prompts; admit uncertainty).
- Structured I/O & tool use (constrained decoding; function calling; production-grade outputs).
- RAG eval as a product metric (hit@k vs quality; nDCG/MRR; hallucination on miss; latency).
- Multilingual & code tracks (evals for XNLI/FLORES; HumanEval-Plus/MBPP; pass@k).
**Requirements:**
- Index 10k+ documents with FAISS and bge-small.
- Implement reranking with bge-reranker-base.
- Generate answers with structured JSON via constrained decoding.
- Eval on BEIR/in-house set with nDCG/MRR and hallucination analysis.
- Add multilingual/code benchmarks.
- Measure latency.
**Dependencies:** FAISS, Hugging Face Transformers (for bge models), Outlines (grammar-based decoding), Pydantic v2 (JSON Schema), lm-eval-harness.
### Project 6: Agentic Workflow for Task Planning
**Description:** Build a simple agent using your RAG model from Project 5 that chains tools (e.g., calculator, search) with planning and scratchpad for multi-step tasks like math problems.
**Skills Learned:**
- Agentic patterns (planning; tool chaining; scratchpad; even for SLMs).
- Structured I/O integration (enforce JSON/typed outputs; schema-first prompts; function calling).
**Requirements:**
- Implement ReAct-style agent with 2+ tools.
- Use scratchpad for reasoning traces.
- Chain with RAG from Project 5.
- Eval on budgeted tasks (e.g., GSM8K with tool calls).
- Log traces.
**Dependencies:** LangGraph (ReAct traces & planners), Pydantic v2, Outlines.
### Project 7: Optimize and Compress the SLM
**Description:** Apply compression to your 300M model from Project 4, including quantization and pruning, then re-evaluate.
**Skills Learned:**
- Smaller/faster with minimal loss (quantization/distillation/pruning; tradeoffs; pipeline: Distill → LoRA → INT4 → sparsity → KV quant).
- Throughput & memory (mixed precision; memory-saving; train bf16/fp16; infer INT4/8; KV-cache quant/paging).
- QAT & FP8 paths (awareness; QAT for edge).
**Requirements:**
- Quantize to INT4 using AutoAWQ.
- Apply 2:4 sparsity.
- Re-eval with Project 2 metrics, reporting tradeoffs.
- Measure memory/latency before/after.
**Dependencies:** AutoAWQ (INT4), PyTorch (for sparsity), gradient checkpointing/accumulation.
### Project 8: Distributed Training for 1B SLM
**Description:** Scale training of a 1B model (built on Project 3) across 2+ GPUs, using data from Project 1.
**Skills Learned:**
- Distributed training (scale GPUs/nodes; communication; memory partitioning; FSDP ZeRO-2/3 >2B; CPU offload).
- Parallelism strategies (data/tensor/pipeline; shard states/checkpoints; avoid pipeline; grad checkpointing + accumulation).
- Hardware awareness (interconnects; I/O limits; GPU profiles; CPU vectorization; NUMA pinning).
**Requirements:**
- Train with FSDP on 2 GPUs.
- Shard checkpoints.
- Profile I/O and fix bottlenecks.
- Monitor with nvidia-smi.
**Dependencies:** PyTorch FSDP (NCCL), nvidia-smi, numactl, iostat.
### Project 9: Inference Serving API
**Description:** Deploy your compressed model from Project 7 as a low-latency API with batching and streaming.
**Skills Learned:**
- Prototype APIs (low-latency inference; batching/streaming; CPU-first; speculative decoding).
- LLM serving essentials (long-context decoding; KV caches; sliding-window; paged KV; vLLM for server).
- On-device measurement (ONNX Runtime Mobile; cold/warm start; battery/CPU harness).
**Requirements:**
- Serve via API with SSE streaming.
- Implement batching and KV cache.
- Measure TTFT/tokens/s on CPU/GPU.
- Export to ONNX for edge.
**Dependencies:** FastAPI + Uvicorn, vLLM, ONNX + ONNX Runtime, tiktoken, llama.cpp (gguf for edge).
### Project 10: Experiment Management and Reproducibility Suite
**Description:** Wrap prior projects (e.g., training from Project 8) with full tracking, sweeps, and reproducibility tools.
**Skills Learned:**
- Tracking & sweeps (centralize metrics/artifacts/configs; automate HPO; log tokens/batch/LR/teacher).
- Reproducibility & experiment management (determinism; seeding; env pinning; W&B Artifacts; Git tags; release hygiene with Model/Data Cards).
- Testing & CI (unit/integration tests; run on PRs; golden-prompt tests; quantized loaders).
**Requirements:**
- Track a sweep with Optuna.
- Ensure deterministic runs with seeding.
- Add tests for data/model.
- Generate Model Card.
- Set up CI.
**Dependencies:** Weights & Biases + Optuna, Hydra (configs), pytest, coverage.py, GitHub Actions, Docker.
### Project 11: Fault-Tolerant Orchestration on Cluster
**Description:** Run distributed training from Project 8 on a cluster with preemption handling and observability.
**Skills Learned:**
- Schedulers & clusters (launch/manage jobs; quotas; spot GPUs; shard by data; frequent checkpoints).
- Fault tolerance & elastic workflows (preemption/failure; resume by token; object-store checkpoints; idempotent steps; SIGTERM handlers).
- Observability & resilience (metrics/logs/traces; TTFT/tokens/s; peak RSS; timeouts/circuit breakers; correlation IDs).
- Resilient rollouts (canary by device/quant; fallback routes).
**Requirements:**
- Launch on SLURM with job arrays.
- Implement checkpointing to S3.
- Add metrics dashboard.
- Handle preemption resumes.
- Roll out with canaries.
**Dependencies:** SLURM, TorchElastic, Prometheus, Grafana, OpenTelemetry, structlog, pybreaker, tenacity, Argo Rollouts + Argo CD.
### Project 12: Full Portfolio Demo with Security and Cost Tracking
**Description:** Integrate a RAG-agent from Projects 5-6, served via Project 9, with security, cost tracking, and interpretability probes; release as a complete app.
**Skills Learned:**
- Profiling & kernels (find bottlenecks; optimize decode; CPU thread pools; fused-attention).
- Sequence-length scaling (RoPE θ/NTK; attention sinks; LongBench awareness).
- Interpretability & probing (logit lens; linear probes; ablations for regressions).
- Security, privacy & responsible AI (PII handling; credentials lockdown; harms/biases checks; on-device to reduce egress; jailbreak lists).
- Cost & SLO thinking (track $/requests; GPU-hours; p50/95/99 latency; uptime; latency/memory budgets; model flavors).
- Export & compilers (convert to portable runtimes; gguf for llama.cpp; ONNX for edge).
- Edge & browser (CPU llama.cpp; fp16/int8/int4 builds; device sniffing).
**Requirements:**
- Add probes and ablations to diagnose issues.
- Implement PII scrubbing and safety checks.
- Track costs/latency in dashboard.
- Export to gguf/ONNX.
- Optimize for edge with profiling.
- Release with full cards and checks.
**Dependencies:** PyTorch Profiler, FlashAttention, llama.cpp (gguf), ONNX Runtime, Vault (secrets), regex (PII), W&B/Grafana (costs/SLOs), scancode-toolkit (licensing awareness).