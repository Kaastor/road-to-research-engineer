## Full PhD Plan: Efficient Agentic Systems Using Small Language Models for Edge and Resource-Constrained Applications

This updated PhD plan incorporates concrete elements like specific papers to reproduce, methods (e.g., distillation techniques), frameworks (e.g., LangGraph, FAISS), models (e.g., Phi-3-mini), and evaluation benchmarks (e.g., MT-Bench, HumanEval). These are drawn from recent literature (e.g., "Small Language Models are the Future of Agentic AI" by NVIDIA researchers, arXiv 2506.02153) to make tasks actionable. The structure remains the same, with additions integrated directly into sections for clarity.

### 1. Overview
- **Topic Summary:** This PhD investigates designing efficient agentic systems using SLMs (≤3B parameters, e.g., Phi-3-mini or Gemma-2B) for edge devices (e.g., mobile/CPU). Key elements include adapting patterns like ReAct-style planning and tool chaining (from Yao et al., 2022, arXiv 2210.03629), lightweight distillation from LLMs (e.g., knowledge distillation per "Small Language Models are the Future of Agentic AI"), RAG integration for robustness (using FAISS with bge-small embeddings), and safety evaluations (e.g., jailbreak tests from Perez et al., 2022). Research will produce prototypes, frameworks, and benchmarks emphasizing low-latency, low-memory deployment (e.g., via llama.cpp/ONNX with INT4 quantization).
- **Rationale for Fit:** Aligns with your skills roadmap (e.g., "Agentic patterns" [Strong], "Structured I/O" [Must], "Safety evals" [Must]) and portfolio tasks (e.g., reasoning distillation). It's feasible part-time: leverages pre-trained models like Microsoft's Phi series (using distillation and RL as in "One Year of Phi" blog) to minimize training time, focuses on modular prototyping/evals that fit evenings, and builds toward industry-relevant outputs like open-source agents.
- **Thesis Structure (High-Level):** Introduction to SLM agents; Literature review (covering papers like "Can Compact Language Models Search Like Agents?" arXiv 2508.20324 on distilling RAG behaviors); Methodology (distillation/agent design); Experiments/evals; Deployment case studies (e.g., mobile task automation); Conclusions and future work.
- **Expected Outcomes:** 3-5 publications (e.g., NeurIPS, ICML workshops); Open-source repo (e.g., SLM-Agent-Framework on GitHub with LangGraph integrations); Skills mastery leading to research engineer roles at places like Hugging Face or xAI.

### 2. Objectives
- **Research Objectives:**
  1. Develop SLM-based agent architectures optimized for edge constraints (e.g., latency <1s, memory <1GB, using ReAct with LangGraph on Phi-3-mini).
  2. Create distillation pipelines to transfer agentic capabilities from LLMs to SLMs with minimal quality loss (e.g., supervised fine-tuning with rank-stabilized LoRA, as in Stafford's LinkedIn post on DUKE-based distillation).
  3. Design evaluation frameworks for agent robustness, including error taxonomies (e.g., categorizing failures by reasoning depth per "Error Analysis in Language Models") and safety metrics (e.g., toxicity via Perspective API).
  4. Prototype deployable systems (e.g., Android app for task automation using ONNX Runtime Mobile) and measure real-world tradeoffs (e.g., power consumption via Android Profiler).
- **Personal Development Objectives:**
  1. Master core skills from your list (e.g., PyTorch mastery via fine-tuning Phi models, RAG integration with FAISS, error analysis using Polars for data slicing) through hands-on research.
  2. Build a portfolio (e.g., reproductions of ReAct agents, distillation demos on Hugging Face) for job applications.
  3. Network via conferences/publications to transition to full-time research engineering.

### 3. Timeline
Adjusted for part-time (half-pace of full-time 4-6 year programs). Focus on evening sessions: e.g., Mon-Wed for coding/reading, Thu-Fri for evals/analysis, weekends for writing/planning. Use tools like W&B for tracking to resume sessions easily. Milestones include qualifiers (end of Year 2) and proposal defense (end of Year 3).

- **Year 1: Foundations and Setup (Focus: Coursework + Initial Skills Building)**
  - **Months 1-6:** Apply to part-time/flexible PhD programs (e.g., online/hybrid like SJSU Gateway PhD or UIUC's part-time options; target advisors in SLM/agent research at labs like Stanford NLP or Berkeley AI, such as authors of Phi models). Secure funding (e.g., part-time TAships or industry sponsorships).
  - **Months 7-12:** Complete 2-3 courses (online/evening if possible). Start literature review: Reproduce 2-3 agent papers (e.g., "ReAct: Synergizing Reasoning and Acting" by Yao et al., 2022 via LangGraph; "Small Language Models are the Future of Agentic AI" by NVIDIA, 2025 for SLM conversion algo; one more like "Toolformer" by Schick et al., 2023)—1 paper/month. Build basic SLM agent prototype (e.g., simple tool-chaining demo with Phi-3-mini and Pydantic for structured outputs). Track with Git/W&B.
  - **Weekly Effort:** 10-15 hours; e.g., 2 evenings reading papers, 2 coding (implement ReAct loop in PyTorch), 1 analyzing (basic latency tests).
  - **Milestones:** Advisor secured; Initial repo with reproductions; Skills: "Paper → code" [Must], "Python mastery" [Must].
  - **Output:** Short report on SLM agent gaps (internal to advisor, citing gaps from NVIDIA paper).

- **Year 2: Core Skill Development and Qualifiers (Focus: Experiment Design + Baselines)**
  - **Months 1-6:** 2 more courses. Implement agentic patterns: Build ReAct-style agents with SLMs (e.g., Phi-3-mini integrated with LangGraph for planning), add structured I/O (Outlines/Pydantic for JSON parsing). Run baselines/ablations on MT-Bench-lite (short evals, <1 hour/run on CPU, comparing to Gemma-2B).
  - **Months 7-12:** Integrate lightweight distillation (e.g., short-rationale from a 7B teacher like Llama-3-8B using knowledge distillation per "Can Compact Language Models Search Like Agents?" 2025). Develop error taxonomies (slice by prompt length/domain using Polars, e.g., categorizing chain-of-thought failures). Prep for qualifiers (e.g., written exam on ML foundations).
  - **Weekly Effort:** 15 hours; e.g., 3 evenings prototyping (fine-tune with Hugging Face Transformers), 2 evals/plots (Matplotlib for ablation curves), 1 writing.
  - **Milestones:** Pass qualifiers; First conference submission (e.g., ACL workshop on agent evals, submitting ReAct reproduction results). Skills: "Agentic patterns" [Strong], "Baselines & ablations" [Must], "Error analysis" [Strong].
  - **Output:** Prototype v1 (GitHub repo with LangGraph agents); 1 workshop paper.

- **Year 3: Advanced Research and Proposal (Focus: Distillation + Robustness)**
  - **Months 1-6:** Defend research proposal (e.g., outline distillation recipes like LLM-to-SLM conversion from NVIDIA paper, and RAG integration). Experiment with DPO/heuristic rewards for alignment (using RLHF-like setup from "Phi-reasoning" models); Add RAG (FAISS with bge-small embeddings) to agents for OOD robustness (e.g., handling unseen tools).
  - **Months 7-12:** Incorporate safety evals (jailbreak tests from "Red Teaming Language Models" by Perez et al., 2022; toxicity classifiers like Hugging Face's Detoxify). Run sweeps (Optuna for hyperparams, unattended if cloud-based) and analyze with stats (SciPy for confidence intervals, paired bootstrap resampling).
  - **Weekly Effort:** 15-20 hours; e.g., 3 evenings experiments (distill with LoRA adapters), 2 analysis/safety checks (run jailbreak prompts), 1-2 writing/submissions.
  - **Milestones:** Proposal approved; 1-2 publications (e.g., NeurIPS workshop or arXiv preprint on distilled agents). Skills: "Distillation recipes" [Must], "Safety & robustness evals" [Must], "RAG-first design" [Must].
  - **Output:** Enhanced framework with RAG/safety (e.g., FAISS-integrated LangGraph); Portfolio task #2 (reasoning distillation demo on Phi-3).

- **Year 4: Deployment and Evaluation (Focus: Edge Optimization + Metrics)**
  - **Months 1-6:** Optimize for edge: Quantize models (AutoAWQ for INT4 on Gemma-2B), export to gguf/ONNX, test on CPU/mobile (llama.cpp for inference, measure on Raspberry Pi). Measure SLOs (latency/memory with PyTorch Profiler or ONNX Runtime benchmarks).
  - **Months 7-12:** Build end-to-end demos (e.g., FastAPI backend with mobile frontend for agent tasks); Conduct full evals (e.g., hallucination rates via ROUGE on synthetic data, agent success on HumanEval for code agents). Tie to error fixes via ablations (e.g., RAG vs. no-RAG).
  - **Weekly Effort:** 15-20 hours; e.g., 3 evenings deployment/testing (compile with llama.cpp), 2 evals/metrics (track with W&B), 1-2 paper revisions.
  - **Milestones:** 2 more publications (e.g., ICML or EMNLP on edge deployments). Skills: "Export & compilers" [Strong], "Throughput & memory" [Must], "On-device measurement" [Strong].
  - **Output:** Deployable prototypes (e.g., mobile agent app using ONNX Runtime Mobile); Portfolio task #3 (RAG demo with FAISS).

- **Year 5: Synthesis and Defense (Focus: Thesis + Dissemination)**
  - **Months 1-6:** Compile results into thesis chapters; Run final experiments (e.g., multilingual/code evals like XNLI for language robustness, HumanEval for code generation). Address feedback from advisor/committee.
  - **Months 7-12:** Defend dissertation; Job search (leverage portfolio for research engineer roles). Open-source full framework (e.g., with examples from "Empowering Edge AI with Small Language Models" paper); Present at conferences.
  - **Weekly Effort:** 10-15 hours; e.g., 3 evenings writing, 2 revisions/evals, 1 networking/job apps.
  - **Milestones:** Thesis defense; Total 4-5 publications. Skills: Full mastery (e.g., "Release hygiene" [Strong] via clean GitHub repos, "Interpretability & probing" [Strong] using attention visualizations).
  - **Output:** Defended PhD; Open-source SLM agent toolkit (integrated with LangGraph/FAISS); Transition to industry.

### 4. Coursework
Aim for 4-6 courses total (part-time pace: 1-2/year). Select based on program requirements; focus on evenings/online options.
- **Core:** Advanced ML (covers "Foundations LA/Stats/Opt" [Must], e.g., via assignments on linear algebra for embeddings), Deep Learning (PyTorch/JAX fluency, e.g., fine-tuning Phi models).
- **Electives:** NLP/Agents (agentic patterns, e.g., implementing ReAct), Efficient ML (SLM focus: quantization/distillation per "Small Language Models" papers), Stats for Experiments (A/B basics, CIs via SciPy).
- **Integration with Skills:** Use assignments to practice (e.g., reproduce "Toolformer" in DL course).

### 5. Research Milestones and Evaluation
- **Qualifiers (End Year 2):** Exam/portfolio on ML basics + initial prototypes (e.g., ReAct on Phi-3).
- **Proposal Defense (End Year 3):** Detailed plan with preliminary results (e.g., distillation baselines).
- **Annual Reviews:** Advisor meetings (virtual, 1/month) to track progress.
- **Evaluation Metrics:** Track skills mastery (e.g., quarterly self-audit against your list); Use W&B for experiment reproducibility (e.g., logging ablation results).

### 6. Skills Development Integration
Embed your roadmap: Start with "Core modeling & math" in Year 1 coursework (e.g., optimizing distillation loss functions); Progress to "Training systems" via distillation experiments (e.g., LoRA on Hugging Face); End with "Security & responsible AI" in safety integrations (e.g., jailbreak evals).

### 7. Publications and Dissemination
- **Target Venues:** Workshops (NeurIPS SLM/Agents, ICML Efficient ML); Journals (TACL, JAIR).
- **Plan:** 1 workshop paper/year starting Year 2 (e.g., on ReAct reproductions); 1-2 major conference papers in Years 4-5 (e.g., on edge-optimized agents).
- **Networking:** Attend 1-2 virtual conferences/year (e.g., via travel grants); Contribute to open-source (Hugging Face Hub, e.g., uploading distilled Phi variants).

### 8. Resources and Funding
- **Compute:** Personal laptop/CPU for most work (e.g., llama.cpp on local machine); Cloud (Colab/GCP spot instances, <$50/month) for occasional GPU needs (e.g., fine-tuning sweeps).
- **Tools/Software:** From your list (e.g., PyTorch for modeling, LangGraph for agents, FAISS for RAG, AutoAWQ for quantization); Free tiers of W&B/Optuna.
- **Funding:** Part-time scholarships (e.g., NSF GRFP for US programs); Industry partnerships (e.g., xAI or Meta grants for SLM research); Aim for $10-20K/year stipend if available.
- **Advisor/Labs:** Target profs like those at MIT Media Lab (agents), Hugging Face researchers, or academic SLM experts (e.g., authors of Phi models). Programs: Part-time options at SJSU, Cambridge, or online PhDs like University of Liverpool's AI program.

### 9. Risks and Mitigations
- **Time Constraints:** Risk: Burnout from evenings only. Mitigation: Strict session limits; Flexible milestones (extend to 6 years if needed).
- **Compute Access:** Risk: Limited GPU. Mitigation: Focus on CPU-friendly SLMs (e.g., Phi-3-mini via llama.cpp); Use pre-trained models.
- **Motivation:** Risk: Isolation. Mitigation: Join online communities (e.g., Reddit r/MachineLearning, Discord for SLM devs); Bi-weekly advisor check-ins.
- **Funding Gaps:** Risk: Costs. Mitigation: Start with free resources; Apply for grants early.

This plan is realistic, actionable, and directly supports your career goals. Adjust based on your chosen program/advisor. If you need applications help or refinements, let me know!