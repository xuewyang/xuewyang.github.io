---
title: "Projects"
url: "/projects/"
summary: "Selected projects I've led or worked on."
ShowReadingTime: false
ShowWordCount: false
ShowPostNavLinks: false
ShowBreadCrumbs: false
hidemeta: true
---

A selected set of projects I've led over the years. Most are work done under NDA, so descriptions stay at a high level.

---

### CoupangGPT-VL — Vision-Language E-commerce Model
*2024 — present · Coupang · Team Lead*

A unified vision-language foundation model for e-commerce. I initiated and led the
multimodal redesign of Coupang's search experience: queries and product metadata
(titles, images, descriptions, OCR text) are encoded into a shared embedding
space for multimodal retrieval.

- Built a multi-turn conversational **ShopBot** powered by CoupangGPT-VL for natural-language product discovery.
- Improved click-through rate (CTR) by **13.4%** on the redesigned search surface.
- Enabled better semantic retrieval on hard queries ("breathable shoe cabinet" → items with ventilation and airflow attributes).
- Introduced grounded summaries that explain *why* a product was recommended.

---

### CoupangGPT — Scalable E-commerce Language Model
*2023 — 2025 · Coupang · Team Lead*

Led end-to-end development of **CoupangGPT**, a multimodal reasoning GPT pretrained on 1.5 trillion tokens, images, and proprietary e-commerce data — powered by GRPO reinforcement learning and multimodal instruction tuning. Used across search relevance, search ranking, catalog, and ads quality.

- Managed a team of 5 ML engineers, a data scientist, and several product analysts.
- Partnered with infra to stand up Coupang's first multinode training cluster — **1000+ A100 / H100 / H200 GPUs** on p4 / p5 / p5en instances with EFA and InfiniBand.
- Delivered weekly progress updates to VP and Director-level stakeholders.
- Pretraining / post-training stack: PyTorch, FSDP, DeepSpeed, GRPO, DPO.

---

### Search Relevance with CoupangGPT
*2023 — 2025 · Coupang · Team Lead*

Reformulated search relevance as an instruction fine-tuning problem — does this (query, product title, metadata) tuple constitute an exact match? Drove EM accuracy from **0.55 → 0.99** through iterative distillation from larger teacher models.

| Timeline | Model | EM Accuracy |
|----------|-------|-------------|
| Q4 2023 | DCN (1.3M) on 0.2M pairs | 0.55 |
| Q1 2024 | RoBERTa (123M) on 0.2M human labels | 0.78 |
| Q2 2024 | RoBERTa distilled from CoupangGPT-8B | 0.87 |
| Q3 2024 | RoBERTa distilled from CoupangGPT-27B | 0.93 |
| Q4 2024 | RoBERTa distilled from CoupangGPT-27B | 0.96 |
| Q1 2025 | RoBERTa + reasoning traces from 27B | 0.97 |
| Q2 2025 | CoupangGPT-1B + reasoning (student) | **0.99** |

Coordinated 100+ human annotators with PMs to nail down labeling guidelines. The EM lift translated into a measurable lift in purchase rate.

---

### Query Understanding with CoupangGPT
*2024 — 2025 · Coupang · Team Lead*

A multi-task query understanding pipeline — spelling correction, brand detection, query category prediction, query expansion — trained without any human labels for the bootstrap round.

- Used **Gemini-3-Pro** via advanced prompt engineering to generate zero-human-label training data.
- Scaled via knowledge distillation: Gemini-3-Pro → CoupangGPT student.
- The distilled student matched or surpassed the teacher on internal evals.
- Shipped a **2–3% lift in PPC** (purchases per customer).

---

### OGPT — On-Device Chatbot for OPPO Smartphones
*2021 — 2023 · OPPO · Tech Lead*

A high-performance chatbot deployed on smartphones, serving **10M+ monthly users**.

- Applied DPO for instruction-following alignment.
- Distributed training across multiple nodes with FSDP, Lightning, DeepSpeed.
- Integrated multiple LLMs with LangChain; advanced prompt engineering to extend utility across use cases.

---

### Semantic Search for OPPO Photo Album
*2021 — 2023 · OPPO*

Replaced keyword matching on the smartphone Photo Album with a vision-language semantic search stack. Contrastive pretraining of text-image pairs, transformer-based query encoding, and nearest-neighbor retrieval delivered a **10% F1 improvement** over the previous system.

---

### Fashion Captioning & FACAD
*2019 — 2020 · Stony Brook · PhD research*

Released the FACAD dataset and a reinforcement-learning-with-semantic-rewards
method for generating accurate fashion captions, published at ECCV 2020.

- [GitHub — xuewyang/Fashion_Captioning](https://github.com/xuewyang/Fashion_Captioning)
- [Benchmark repo](https://github.com/xuewyang/Fashion-Image-Captioning-Benchmark)

---

### Open Source
*2024 — present*

Active contributor to **[Axolotl](https://github.com/axolotl-ai-cloud/axolotl)** and **[ms-swift](https://github.com/modelscope/ms-swift)** — scalable post-training, instruction tuning, and RL training pipelines.
