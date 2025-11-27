# Metric-Based Success Standards for Dynamic Curriculum Phases

## Overview

This document serves as the "Pass/Fail" criteria for the **Curriculum Controller**. Unlike static training, where completion is defined by epoch count, this dynamic framework defines completion by **state stability**.

The Controller must verify these expectations before triggering a phase transition. If these metrics are not met, the system should trigger a **Rollback** or **Patience Extension** rather than advancing.

## Phase 1: Structural Anchoring (The Skeleton)

Context: Training on Top-20% Degree Centrality nodes (Hubs) in Lorentz Hyperbolic space.

Objective: Embed the hierarchical backbone without noise.

| Metric | Success Expectation (The "Green Light") | Failure Mode (The "Red Flag") |
|------------------------|------------------------|------------------------|
| Hub Validation Loss | Asymptotic Convergence: The loss curve must flatten. Specifically, the loss velocity $V_{\mathcal{L}}$ (1st derivative) should be $< 1e^{-4}$ for 5 consecutive checkpoints. | Oscillation: High variance in loss indicates the learning rate is too high for the curvature $c$. |
| Lorentz Distortion ($D_{\mathcal{L}}$) | Tree-Like Fidelity ($< 0.15$): Since Hubs form the "roots" of the hierarchy, the embedding distances should near-perfectly match graph distances. High distortion here means the hierarchy is broken. | High Distortion ($> 0.3$): Indicates the curvature $c$ is too low (space is too flat) to accommodate the tree structure. |
| Cluster Separation | Silhouette Score $> 0.5$: Major communities (e.g., Science vs. Sports) should be distinct clusters in the hyperbolic space. | "Ball of Wool": A score near 0 implies all hubs are clustered at the origin (North Pole). |

**Transition Logic:**

*Advance when Hub Loss stabilizes AND Distortion is low. If Distortion remains high, increase Hyperboloid curvature (*$c$) before advancing.

## Phase 2: Relational Expansion (The Flesh)

Context: Adding Tail nodes and optimizing 1-to-N relations with Model-Aware Adaptive Margins.

Objective: Generalize from Hubs to Tails without breaking the backbone.

| Metric | Success Expectation (The "Green Light") | Failure Mode (The "Red Flag") |
|------------------------|------------------------|------------------------|
| Generalization Gap ($\Delta\mathcal{L}$) | Stable Gap ($< 0.1$): The difference between Train Loss and Val Loss should remain narrow. This proves the model isn't just memorizing the sparse tail nodes. | Exploding Gap ($> 0.25$): Immediate sign of overfitting on tail nodes. Requires increasing the adaptive margin weight $\lambda$. |
| Hits\@10 (Tails) | Positive Slope: Performance on tail entities must show a monotonically increasing trend. | Flatline: If Hub performance stays high but Tail performance doesn't move, the gradient isn't propagating to the leaves. |
| Curvature Dynamics | Stabilization of $c$: If $c$ is trainable, it should settle into a range (typically decreasing $c$ slightly to expand volume). | Runaway Curvature: If $c \to 0$ (becoming Euclidean) or $c \to \infty$, the manifold optimization has collapsed. |

**Transition Logic:**

*Advance when Tail Hits\@10 improves by 15% over baseline AND Generalization Gap remains within tolerance (*$\delta$).

## Phase 3: Semantic Discrimination (Refinement)

Context: Full graph with Generative Hard Negative Mining in Product Space (Lorentz $\times$ Euclidean).

Objective: Fine-grained separation of semantically similar entities.

| Metric | Success Expectation (The "Green Light") | Failure Mode (The "Red Flag") |
|------------------------|------------------------|------------------------|
| Embedding Uniformity | High Entropy: The distribution of embeddings on the manifold should be uniform (avoiding "clumping"). Measured via Gaussian Potential; we expect $\log U \approx -2$. | Dimensional Collapse: If embeddings collapse into a lower-dimensional subspace, Hard Negative Mining is failing (negatives are too weak). |
| MRR (Global) | "S-Curve" Saturation: We expect a sharp rise in Mean Reciprocal Rank as hard negatives correct the decision boundaries, followed by a hard plateau. | Regression: If MRR drops after introducing hard negatives, the negatives are "False Negatives" (valid links marked as negative). |
| Hardness Ratio | Gradient Magnitude Maintenance: The average gradient norm should remain non-zero. Hard negatives exist to prevent vanishing gradients. | Vanishing Gradients: If the loss on hard negatives drops to 0, the Generator is not producing difficult enough samples. |

**Transition Logic:**

*Advance to Phase 4 when Global MRR plateaus for \>10 epochs. If Uniformity degrades, Rollback to Phase 2.*

## Phase 4: Global Stabilization (Anti-Forgetting)

Context: Full dataset re-training using Self-Distillation. The Phase 3 model acts as the "Teacher" to constrain the Phase 4 "Student."

Objective: Restore global manifold continuity and repair any structural damage caused by aggressive hard negative mining.

| Metric | Success Expectation (The "Green Light") | Failure Mode (The "Red Flag") |
|------------------------|------------------------|------------------------|
| Backward Transfer (BWT) | Minimal Degradation ($> -5\%$): We measure the accuracy on the Phase 1 Test Set (Hubs). The model should retain its knowledge of the skeleton. | Catastrophic Forgetting ($< -10\%$): Performance on Hubs drops significantly. This means the model has overwritten structural knowledge to memorize hard semantic edge cases. |
| Teacher-Student KL Div | Convergence to 0: The student should align with the teacher's probability distribution on "easy" samples while outperforming it on "hard" ones. | Divergence: Indicates the student is drifting too far from the stable trajectory established in previous phases. |
| Lorentz Distortion ($D_{\mathcal{L}}$) | Re-Stabilization: Distortion usually spikes in Phase 3. In Phase 4, it should decrease back towards Phase 2 levels ($< 0.2$). | Permanent Distortion: If distortion remains high, the manifold has been permanently warped by the Euclidean product component. |

**Completion Logic:**

*Training Complete when BWT is stable and Global MRR is within 1% of Phase 3 peak.*

## Summary of Control Loop Logic

1.  **Check Velocity:** Is the model learning? (If Velocity $\approx 0 \to$ Check Plateau).

2.  **Check Stability:** Is the Gap widening? (If Gap $> \delta \to$ **Rollback**).

3.  **Check Geometry:** Is the manifold healthy? (If Distortion high $\to$ Adjust Curvature).

4.  **Check Topology:** Are we effectively using the space? (If Uniformity low $\to$ Boost Negative Sampling).

5.  **Check Memory (Phase 4):** Did we forget the basics? (If BWT low $\to$ Increase Distillation Weight).