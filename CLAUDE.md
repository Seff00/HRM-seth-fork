# Hierarchical Reasoning Model (HRM) - Codebase Documentation

## Project Overview

This is an implementation of the **Hierarchical Reasoning Model (HRM)** from the paper "Hierarchical Reasoning Model" (arXiv:2506.21734v3).

HRM is a brain-inspired neural architecture that achieves exceptional performance on complex reasoning tasks with minimal training data (~1000 examples) and only ~27M parameters.

### Key Results
- **ARC-AGI-1**: 40.3% (beats o3-mini-high at 34.5%, Claude 3.7 at 21.2%)
- **Sudoku-Extreme**: 55% (other models: 0%)
- **Maze-Hard (30×30)**: 74.5% (other models: 0%)

## Architecture Details

### Model Configuration (from `config/arch/hrm_v1.yaml`)

```yaml
H_layers: 4              # High-level module layers
L_layers: 4              # Low-level module layers
H_cycles: 2              # High-level cycles per forward pass
L_cycles: 2              # Low-level cycles per forward pass
hidden_size: 512
num_heads: 8
expansion: 4
halt_max_steps: 16       # Maximum ACT forward passes
```

### Parameter Breakdown

**Total Model: ~27M parameters**

#### Per Transformer Block (~3.41M params)
1. **Attention Module** (~1.05M params)
   - `qkv_proj`: 512 → 1536 = 786,432 params
   - `o_proj`: 512 → 512 = 262,144 params

2. **SwiGLU MLP** (~2.36M params)
   - `gate_up_proj`: 512 → 3072 = 1,572,864 params
   - `down_proj`: 1536 → 512 = 786,432 params

#### Module Distribution
- **H-module (4 layers)**: ~13.6M parameters
- **L-module (4 layers)**: ~13.6M parameters
- **Embeddings + Heads**: ~0.8M parameters

### Effective Computational Depth

The model achieves deep computation through:
- **Base depth per pass**: H_cycles × L_cycles = 2 × 2 = 4 steps
- **Each step**: 4 transformer layers
- **With ACT**: up to 16 forward passes
- **Maximum effective depth**: 4 × 4 × 16 = **256 equivalent transformer layers**

## Codebase Structure

```
├── models/
│   ├── hrm/
│   │   └── hrm_act_v1.py          # Main HRM implementation with ACT
│   ├── layers.py                   # Transformer blocks, Attention, SwiGLU
│   ├── losses.py                   # Loss functions including ACT loss
│   ├── sparse_embedding.py         # Sparse embedding layer
│   └── common.py                   # Utility functions
├── dataset/
│   ├── build_arc_dataset.py       # ARC dataset generation
│   ├── build_maze_dataset.py      # Maze dataset generation
│   ├── build_sudoku_dataset.py    # Sudoku dataset generation
│   └── common.py                   # Dataset utilities
├── config/
│   ├── arch/
│   │   └── hrm_v1.yaml            # Model architecture config
│   └── cfg_pretrain.yaml          # Training hyperparameters
├── pretrain.py                     # Training script
├── evaluate.py                     # Evaluation script
├── puzzle_dataset.py               # Dataset loader
└── run300.sh                       # Bash script for experiments
```

## Key Implementation Details

### 1. Hierarchical Architecture (`models/hrm/hrm_act_v1.py`)

The model has two main recurrent modules:
- **H-module** (High-level): Slow, abstract planning
- **L-module** (Low-level): Fast, detailed computation

```python
# Simplified forward flow (from hrm_act_v1.py:192-198)
for H_step in range(H_cycles):
    for L_step in range(L_cycles):
        z_L = L_level(z_L, z_H + input_embeddings)
    z_H = H_level(z_H, z_L)
```

### 2. Hierarchical Convergence

The L-module converges to a local equilibrium during each cycle, then gets "reset" when the H-module updates with new context. This prevents premature convergence that plagues standard RNNs.

### 3. One-Step Gradient Approximation (Two Levels)

The model uses gradient approximation at **two distinct levels**:

#### 3.1 Inner Loop: Cycle-Level Approximation
Within each ACT segment, only the final H/L cycle gets gradients (`hrm_act_v1.py:189-204`):

```python
# No-grad iterations
with torch.no_grad():
    z_H, z_L = carry.z_H, carry.z_L
    for _H_step in range(H_cycles):  # e.g., 0, 1
        for _L_step in range(L_cycles):  # e.g., 0, 1
            if not (last_iteration):
                z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        if not (last_H_step):
            z_H = self.H_level(z_H, z_L, **seq_info)

# Only final iteration gets gradients
z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)  # 1-step grad
z_H = self.H_level(z_H, z_L, **seq_info)  # 1-step grad
```

With H_cycles=2, L_cycles=2: This means 3 iterations run with no_grad, and only the 4th iteration computes gradients.

#### 3.2 Outer Loop: Segment-Level Approximation
Between ACT segments, the carry state is **detached** (`hrm_act_v1.py:207`):

```python
# New carry has no gradient connection to previous segment
new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(
    z_H=z_H.detach(),  # Breaks gradient flow
    z_L=z_L.detach()
)
```

This prevents gradients from flowing backward across segments during Adaptive Computation Time (ACT).

#### Why Two Levels?

Both serve the same purpose but at different scales:
- **Inner**: Avoid BPTT across H/L cycles (saves memory, faster convergence)
- **Outer**: Avoid BPTT across ACT segments (enables deep supervision)
- **Combined effect**: O(1) memory complexity, frequent gradient updates, forces model to learn useful representations at each step

### 4. Adaptive Computation Time (ACT)

Uses Q-learning to decide when to halt computation:
- **Q-head**: Predicts Q(halt) and Q(continue)
- **Exploration**: Random minimum steps with probability ε
- **Training**: Uses deep supervision at each segment

### 5. Deep Supervision

The model is trained with supervision at each forward pass segment, providing more frequent feedback and regularization. The detach mechanism (section 3.2) is crucial for enabling this - without it, gradients would accumulate across all segments.

### 6. ARC-AGI Puzzle Embeddings

Each ARC-AGI puzzle is assigned a learnable sparse embedding that acts as a "puzzle identifier" (`hrm_act_v1.py:116-120`):

```python
self.puzzle_emb = CastedSparseEmbedding(
    num_puzzle_identifiers,  # One per unique puzzle
    puzzle_emb_ndim,         # Embedding dimension
    batch_size=batch_size,
    init_std=0,              # Zero-initialized
    cast_to=forward_dtype
)
```

**How it works**:
1. Each puzzle gets a unique ID (e.g., puzzle "007bbfb7" → ID 42)
2. All examples from the same puzzle (demonstration pairs + test inputs) share this ID
3. The embedding is **prepended** to the input sequence (`hrm_act_v1.py:158`):
   ```python
   embedding = torch.cat((
       puzzle_embedding.view(-1, puzzle_emb_len, hidden_size),
       token_embedding  # Grid input
   ), dim=-2)
   ```
4. Model learns to associate puzzle_id → transformation rule
5. Has its own optimizer with separate learning rate (config: `puzzle_emb_lr=1e-2`)

**Dataset structure** (`build_arc_dataset.py:199-201`):
- All examples for a puzzle are grouped together
- Puzzle identifiers are saved to map back to original names
- Training uses augmentation (1000 variants per puzzle with color permutations + dihedral transforms)

### 7. Puzzle Embedding Architecture Discrepancy ⚠️

**CRITICAL FINDING**: The implementation of puzzle embeddings differs from the paper's description.

#### What the Paper Says (Page 12)

> "Each task example is prepended with a learnable special token that represents the puzzle it belongs to."

This wording suggests the puzzle ID token is **prepended to the input sequence BEFORE embedding**, implying a single embedding layer would handle both puzzle IDs and grid tokens.

#### What the Code Actually Does

The code uses **two separate embedding layers**:

1. **Grid token embedding** (`hrm_act_v1.py:112`):
   ```python
   self.embed_tokens = CastedEmbedding(
       self.config.vocab_size,
       self.config.hidden_size,
       ...
   )
   ```

2. **Puzzle ID embedding** (`hrm_act_v1.py:119-120`):
   ```python
   self.puzzle_emb = CastedSparseEmbedding(
       self.config.num_puzzle_identifiers,
       self.config.puzzle_emb_ndim,
       batch_size=batch_size,
       init_std=0,
       cast_to=forward_dtype
   )
   ```

#### The Concatenation Process (`_input_embeddings` method, lines 146-166)

```python
# Step 1: Embed grid tokens
embedding = self.embed_tokens(input.to(torch.int32))  # Line 148

# Step 2: Embed puzzle ID separately
puzzle_embedding = self.puzzle_emb(puzzle_identifiers)  # Line 152

# Step 3: Concatenate AFTER both are embedded
embedding = torch.cat((
    puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size),
    embedding  # grid token embeddings
), dim=-2)  # Line 158
```

#### Implication

The puzzle ID is **embedded separately, then prepended** to the already-embedded grid tokens. This is a meaningful architectural difference:

- **Paper description**: Prepend token ID → Single embedding layer
- **Code implementation**: Two separate embeddings → Concatenate embeddings

This two-embedding approach allows the puzzle ID to have its own learned embedding space separate from the grid token vocabulary, which may provide more flexibility in learning puzzle-specific representations.

## Training Details

### Key Hyperparameters (`config/cfg_pretrain.yaml`)
- **Batch size**: 768
- **Learning rate**: 1e-4 (with 2000 step warmup)
- **Optimizer**: Adam-atan2 (scale-invariant variant)
- **Weight decay**: 0.1
- **Architecture**: Post-Norm with RMSNorm
- **No pre-training**: Trained from scratch

### Data Efficiency
- **ARC-AGI**: ~1000 examples (with augmentation)
- **Sudoku-Extreme**: 1000 examples
- **Maze-Hard**: 1000 examples

## Evaluation: Paper vs Code Gap ⚠️

**CRITICAL FINDING**: The codebase does NOT implement the test-time optimization procedure described in the paper.

### What the Paper Says (Page 12)

> "During evaluation, we first **optimize the puzzle embeddings on the demonstration examples** using the same training procedure. We then freeze the embeddings and generate predictions for the test examples."

This describes a **two-stage test-time learning** process:
1. **Stage 1**: Optimize puzzle embedding on demonstration pairs (with gradient descent)
2. **Stage 2**: Freeze embedding, run inference on test inputs
3. Use 1000 augmented variants + majority voting

### What the Code Actually Does

The implementation uses a **simplified evaluation** without test-time optimization:

**`evaluate.py:60-61`:**
```python
train_state.model.eval()  # Set to eval mode
metrics = evaluate(config, train_state, eval_loader, eval_metadata, ...)
```

**`pretrain.py:266-288`:**
```python
def evaluate(...):
    with torch.inference_mode():  # NO GRADIENTS!
        for set_name, batch, global_batch_size in eval_loader:
            carry = train_state.model.initial_carry(batch)

            # Pure inference on frozen pre-trained model
            while True:
                carry, _, metrics, preds, all_finish = train_state.model(carry, batch, ...)
                if all_finish:
                    break
```

**`arc_eval.ipynb`:**
- Only does post-processing (inverse augmentation, cropping, majority voting)
- No optimization code

### How Demonstration Pairs Are Used

**CRITICAL**: The code **completely ignores** demonstration pairs during evaluation.

**Dataset creation** (`build_arc_dataset.py:148-178`):
```python
# Demonstration and test pairs are separated into different splits
train_examples_dest = ("train", "all")  # Demo pairs → "train" split
test_examples_map = {
    "evaluation": [(1.0, ("test", "all"))],  # Test pairs → "test" split
}

dest_mapping = {
    "train": train_examples_dest,   # ARC "train" examples → training dataset
    "test": test_examples_dest      # ARC "test" examples → test dataset
}
```

**During evaluation** (`evaluate.py:42`):
```python
# Only loads the "test" split - demonstration pairs are never loaded
eval_loader, eval_metadata = create_dataloader(
    config,
    "test",  # ← Only test pairs, NOT demonstrations
    test_set_mode=True,
    ...
)
```

**What actually happens at test time**:
1. Model loads only test input-output pairs (no demonstration pairs)
2. Each test pair has 1000 augmented variants in the dataset
3. Pure inference runs on all variants with frozen weights
4. Predictions go through inverse augmentation + majority voting (`arc_eval.ipynb`)

### Comparison Table

| Aspect | Paper Description | Code Implementation |
|--------|------------------|---------------------|
| Demo pairs at test time | Optimize puzzle embedding (gradient descent) | **Never loaded or used** |
| Test pairs usage | Freeze embedding, run inference | Run inference on 1000 augmented variants |
| Gradient computation | Yes (for demo pairs) | No (`torch.inference_mode()`) |
| Process | Two-stage (adapt on demos → infer on tests) | Single-stage (pure inference on tests only) |
| Augmentation + voting | ✅ Described (1000 variants) | ✅ Implemented (`arc_eval.ipynb`) |
| Puzzle embedding source | Test-time optimization | **Pre-trained only (frozen)** |

### Implication

This is an **even more significant departure** from the paper than just "no test-time optimization":

1. **Demonstration pairs are completely unused** - the model never sees them at test time
2. **Pure zero-shot inference** - relies entirely on pre-trained puzzle embeddings
3. **No adaptation whatsoever** - no gradient updates, no fine-tuning, not even seeing the demo pairs
4. **Test-time augmentation only** - the only test-time technique is running 1000 augmented variants + voting

This makes the results **more impressive** from a generalization standpoint - the model solves puzzles without any test-time examples showing the transformation rule. However, it's a major discrepancy from the paper's described methodology.

## Brain Correspondence

The model exhibits a **dimensionality hierarchy** similar to the mouse cortex:
- **High-level module (zH)**: Participation Ratio = 89.95 (high-dimensional)
- **Low-level module (zL)**: Participation Ratio = 30.22 (low-dimensional)
- **Ratio**: ~2.98 (similar to biological cortex at ~2.25)

This emergent property arises during training and correlates with the model's ability to handle diverse, complex tasks.

## Key Innovations

1. **Hierarchical convergence**: Prevents premature convergence while maintaining stability
2. **O(1) memory training**: Efficient gradient approximation without BPTT
3. **Adaptive depth**: Dynamically allocates computation based on problem difficulty
4. **Small-sample learning**: Achieves strong results with ~1000 examples
5. **No CoT required**: Performs latent reasoning without chain-of-thought

## Discussion Summary

### Questions Answered

1. **Parameter counts**: Not stated in paper; calculated from config to be ~27M total (~13.6M per H/L module)

2. **Layer structure**:
   - H-module: 4 transformer layers
   - L-module: 4 transformer layers
   - Each layer: ~3.41M params (Attention + SwiGLU MLP)

3. **Gradient flow mechanisms**:
   - **Inner loop**: Only final H/L cycle iteration gets gradients within each ACT segment
   - **Outer loop**: Detach between ACT segments prevents cross-segment gradient flow
   - Both serve same purpose: avoid BPTT, enable frequent updates, force intermediate learning

4. **Deep supervision with detach**:
   - Detach (`z_H.detach()`, `z_L.detach()`) breaks computation graph between segments
   - Enables supervision at each segment without accumulating gradients across all segments
   - Provides frequent feedback signal during training

5. **ARC-AGI input structure**:
   - Each puzzle gets a learnable embedding (puzzle_id → embedding vector)
   - **During training**: Only demonstration pairs used; model learns puzzle embeddings from them
   - **During evaluation**: Only test inputs used; same puzzle_id retrieves the frozen embedding learned during training
   - Embedding is prepended to grid input: `[puzzle_emb, grid_tokens]`
   - Model learns to use embedding to identify which transformation rule applies

6. **Inference procedure**:
   - **In paper**: Two-stage test-time process (optimize puzzle embedding on demos → freeze → predict on tests)
   - **In code**: Pure zero-shot inference (no test-time optimization, no demonstration pairs)
   - Model uses puzzle embeddings learned during pre-training (frozen)
   - Test inputs are run through model with 1000 augmented variants
   - Predictions are evaluated against ground truth test outputs
   - Augmentation + voting (1000 variants, top 2 majority vote) is implemented

### Key Insights

- The model achieves **256 equivalent transformer layers** through recurrence (4×4×16)
- Uses **two levels** of gradient approximation (inner: cycle-level, outer: segment-level)
- **Major evaluation discrepancy**: Code completely ignores demonstration pairs at test time
  - Paper: Two-stage test-time learning (optimize on demos → infer on tests)
  - Code: Zero-shot inference (frozen embeddings, no demos, only test inputs)
- **Training vs Evaluation split**:
  - Training: Only demonstration pairs; learns puzzle embeddings
  - Evaluation: Only test inputs; uses frozen embeddings from training
- This is actually a **stronger result**: model solves puzzles without seeing any demonstration pairs at test time
- **Puzzle embedding architecture**: Uses two separate embedding layers (one for puzzle IDs, one for grid tokens) that are concatenated, rather than a single embedding layer as the paper's wording suggests

## References

- Paper: arXiv:2506.21734v3 [cs.AI] "Hierarchical Reasoning Model"
- Authors: Guan Wang et al., Sapient Intelligence, Singapore
- Code: github.com/sapientinc/HRM
- Discord: https://discord.gg/sapient

---

**Last Updated**: 2025-12-02 (based on codebase at commit a9f2279)
