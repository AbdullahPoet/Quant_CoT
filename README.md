# DeepSeek-R1-Distill-Qwen-7B: Quantization Effect on Chain-of-Thought Reasoning

## Introduction

This experiment studies how **GGUF quantization affects reasoning behavior, accuracy, output length, truncation, throughput, memory footprint, and overthinking** in `DeepSeek-R1-Distill-Qwen-7B`.

The benchmark compares the original **BF16** model with five quantized variants:

`Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, and `Q3_K_M`.

The central question is not only whether lower precision changes final-answer accuracy, but also whether it changes **how much the reasoning model thinks before answering**. This is especially relevant for reasoning LLMs because a smaller model artifact can be faster and cheaper to run while still producing long reasoning traces.

> **Important evaluation note:** the current run contains a substantial answer-parsing issue for `math500`. Therefore, the absolute accuracy values should be treated as preliminary until the parser is corrected and the saved generations are rescored.

---

## Experimental Setup

- **Model:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- **Baseline precision:** BF16
- **Quantized formats:** GGUF
- **Variants:** Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M
- **Original benchmark:** 552 questions
- **Evaluation subset:** 276 questions
- **Total inference rows:** 1,656
- **Maximum generation tokens:** 8,192
- **Inference framework:** vLLM for BF16 / quantized execution pipeline used by the notebook
- **Sampling strategy:** approximately half of the original benchmark, stratified by dataset/category and difficulty

### Benchmark Composition

| Dataset | Category | Sampled Questions |
|---|---|---:|
| Date Understanding | Temporal reasoning | 50 |
| LogiQA2 | Logical reasoning | 50 |
| Math500 | Mathematical reasoning | 50 |
| SimpleQA Verified | Factual QA | 50 |
| StrategyQA | Commonsense multistep reasoning | 50 |
| Misguided Attention | Overthinking traps | 26 |
| **Total** |  | **276** |

---

## Main Comparison

| Precision | Accuracy | Δ Accuracy vs BF16 | Mean Reasoning Tokens | Truncation | Parse Success | Overthinking* | Artifact Size | Output Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BF16** | **37.6%** | — | 1,571 | 9.42% | 71.74% | 24.04% | 14.19 GB | 63.46 tok/s |
| Q8_0 | 35.2% | -2.4 pp | 1,582 | 9.06% | 68.12% | 17.14% | 7.54 GB | 64.35 tok/s |
| Q6_K | 35.6% | -2.0 pp | 1,570 | 10.51% | 68.84% | 20.00% | 5.82 GB | 69.90 tok/s |
| **Q5_K_M** | **37.6%** | **0.0 pp** | **1,569** | **9.06%** | **70.29%** | 17.42% | **5.07 GB** | **77.31 tok/s** |
| Q4_K_M | 34.8% | -2.8 pp | 1,776 | 13.04% | 67.75% | 12.14% | 4.36 GB | **102.01 tok/s** |
| Q3_K_M | 35.6% | -2.0 pp | 1,728 | 12.32% | 69.20% | 15.52% | 3.55 GB | 89.60 tok/s |

\*Overthinking rate is calculated only where an overthinking label was available.

---

## Key Observations

### Q5_K_M preserved measured accuracy while greatly reducing size

`Q5_K_M` matched the measured BF16 accuracy:

- BF16: **37.6%**
- Q5_K_M: **37.6%**

At the same time, the model artifact decreased from:

- **14.19 GB → 5.07 GB**
- approximately **64% smaller**

Its average reasoning length also remained almost unchanged:

- BF16: **1,571 reasoning tokens**
- Q5_K_M: **1,569 reasoning tokens**

In this run, Q5_K_M therefore shows the strongest compression result without an observed loss in aggregate measured accuracy.

---

### More reasoning tokens did not mean higher accuracy

The more aggressively quantized Q4 and Q3 variants generated longer reasoning traces:

| Precision | Reasoning Token Change vs BF16 | Accuracy Change vs BF16 |
|---|---:|---:|
| Q4_K_M | **+13.01%** | **-2.8 pp** |
| Q3_K_M | **+9.96%** | **-2.0 pp** |

This is an important result for the experiment:

> **Longer chain-of-thought generation is not necessarily associated with better final-answer accuracy.**

Q4_K_M generated the most reasoning tokens on average while producing the lowest aggregate measured accuracy.

---

### Quantization improved inference throughput

Output-token throughput increased as quantization became more aggressive:

| Precision | Output Throughput |
|---|---:|
| BF16 | 63.46 tok/s |
| Q8_0 | 64.35 tok/s |
| Q6_K | 69.90 tok/s |
| Q5_K_M | 77.31 tok/s |
| Q4_K_M | **102.01 tok/s** |
| Q3_K_M | 89.60 tok/s |

Q4_K_M produced approximately **61% higher token throughput than BF16** in this run.

---

## Statistical Comparison Against BF16

Exact McNemar tests were used to compare per-question correctness between BF16 and each quantized variant.

| Precision | Correct → Wrong | Wrong → Correct | Discordant Pairs | Exact p-value |
|---|---:|---:|---:|---:|
| Q8_0 | 27 | 21 | 48 | 0.4709 |
| Q6_K | 24 | 19 | 43 | 0.5424 |
| Q5_K_M | 21 | 21 | 42 | 1.0000 |
| Q4_K_M | 24 | 17 | 41 | 0.3489 |
| Q3_K_M | 22 | 17 | 39 | 0.5224 |

At the conventional `α = 0.05` threshold, none of these comparisons are statistically significant.

This means the current sample does **not provide sufficient evidence that any quantized variant has a different correctness rate from BF16**, even though the observed point estimates differ.

---

## Dataset-Level Accuracy

Misguided Attention is excluded from this accuracy table because it is evaluated primarily as an overthinking/trap benchmark rather than with the same deterministic correctness scoring used for the other datasets.

| Precision | Date Understanding | Math500 | SimpleQA | StrategyQA | LogiQA2 |
|---|---:|---:|---:|---:|---:|
| BF16 | 80% | 0% | 2% | 58% | 48% |
| Q8_0 | 60% | 0% | 2% | 56% | 58% |
| Q6_K | 70% | 0% | 4% | 58% | 46% |
| Q5_K_M | 80% | 0% | 4% | **66%** | 38% |
| Q4_K_M | 66% | 4% | 2% | 54% | 48% |
| Q3_K_M | 78% | 4% | **6%** | 54% | 36% |

### Math500 Parser Warning

The apparent Math500 accuracy is not currently reliable because answer parsing almost completely failed:

| Precision | Math500 Parse Success |
|---|---:|
| BF16 | 2% |
| Q8_0 | 0% |
| Q6_K | 2% |
| Q5_K_M | 0% |
| Q4_K_M | 6% |
| Q3_K_M | 4% |

Because generated answers were frequently not extracted correctly, a `0–4%` measured accuracy should **not** be interpreted as the model's true Math500 performance.

The recommended next step is to **repair the Math500 answer parser and rescore the already saved generations without rerunning inference**.

---

## Truncation and Long Reasoning

The benchmark also shows that aggressive quantization can increase the probability that reasoning reaches the generation limit.

Overall truncation:

- BF16: **9.42%**
- Q5_K_M: **9.06%**
- Q4_K_M: **13.04%**
- Q3_K_M: **12.32%**

Several Misguided Attention generations reached approximately **8,192 reasoning tokens** and terminated with:

```text
finish_reason = length
parse_status = missing_final_answer
```

This is a useful failure mode for studying **reasoning-budget exhaustion**: the model can spend its entire token budget reasoning and fail to produce a final answer.

---

## Misguided Attention / Overthinking

Across all 1,656 generations:

- **188** were marked as overthinking
- **870** were marked as not overthinking
- **598** did not receive an overthinking label

Among valid labeled cases, the aggregate rates were:

| Precision | Overthinking Rate |
|---|---:|
| BF16 | 24.04% |
| Q8_0 | 17.14% |
| Q6_K | 20.00% |
| Q5_K_M | 17.42% |
| Q4_K_M | 12.14% |
| Q3_K_M | 15.52% |

An interesting pattern appears here: Q4/Q3 produced **longer reasoning traces overall**, while their current overthinking-label rates were lower than BF16.

This suggests that **reasoning length and overthinking are not equivalent metrics**. A model may generate more tokens without necessarily triggering the benchmark's definition of overthinking.

---

## Model Size Reduction

| Precision | Artifact Size | Reduction vs BF16 |
|---|---:|---:|
| BF16 | 14.19 GB | — |
| Q8_0 | 7.54 GB | ~46.8% |
| Q6_K | 5.82 GB | ~58.9% |
| Q5_K_M | 5.07 GB | ~64.3% |
| Q4_K_M | 4.36 GB | ~69.3% |
| Q3_K_M | 3.55 GB | ~75.0% |

The results show a large reduction in storage requirements without a monotonic collapse in measured benchmark accuracy.

---

## Current Interpretation

The current experiment provides three notable signals:

1. **Quantization degradation is not monotonic.**  
   Lower precision does not automatically produce proportionally lower measured accuracy.

2. **Moderate quantization can preserve reasoning performance.**  
   Q5_K_M matched BF16 aggregate measured accuracy while being roughly 64% smaller and providing higher output throughput.

3. **More generated reasoning can be counterproductive.**  
   Q4_K_M and Q3_K_M generated substantially more reasoning tokens than BF16 without improving accuracy, while also showing higher truncation rates.

These findings support further investigation into the relationship between:

```text
quantization
      ↓
reasoning behavior
      ↓
reasoning-token consumption
      ↓
truncation / overthinking
      ↓
final-answer quality
```

---

## Important Limitations

The current results should be interpreted with the following limitations:

- Math500 answer extraction requires correction.
- Parse success differs across precision variants.
- Some generations reach the 8,192-token output limit.
- Overthinking labels are unavailable for a substantial subset of generations.
- Artifact size is a useful storage measure but is not equivalent to actual runtime GPU-memory consumption under vLLM.
- The benchmark uses a 276-question stratified subset rather than the complete 552-question pool.
- Statistical non-significance does not prove that two precision formats are equivalent; it only indicates that a difference was not detected with the current sample and test.

---