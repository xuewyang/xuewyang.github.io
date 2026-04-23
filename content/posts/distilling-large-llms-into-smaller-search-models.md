---
title: "Distilling large LLMs into small search models"
date: 2026-04-15
tags: ["llm", "distillation", "search", "relevance"]
summary: "How we took a 27B teacher all the way down to a 1B student — and ended up with better accuracy than the teacher on our internal eval set."
draft: true
---

> **Draft.** I'm still deciding how much of this to make public. Reach out if you want a preview.

One of the most counter-intuitive results from our search relevance work at Coupang was this: a 1B parameter student model, distilled from a 27B teacher with reasoning traces, outperformed the teacher on our internal exact-match eval set.

Here's the timeline, roughly:

| Model | Date | Exact-Match Accuracy |
|-------|------|----------------------|
| DCN (1.3M params) on 0.2M title+query pairs | Q4 2023 | 0.55 |
| RoBERTa (123M) on 0.2M human labels | Q1 2024 | 0.78 |
| RoBERTa distilled from CoupangGPT-8B | Q2 2024 | 0.87 |
| RoBERTa distilled from CoupangGPT-27B | Q3 2024 | 0.93 |
| RoBERTa distilled from CoupangGPT-27B | Q4 2024 | 0.96 |
| RoBERTa distilled from CoupangGPT-27B w/ reasoning | Q1 2025 | 0.97 |
| CoupangGPT-1B distilled from CoupangGPT-27B w/ reasoning | Q2 2025 | **0.99** |

The point isn't the numbers — it's what made each jump possible. I'll unpack that in the full post.
