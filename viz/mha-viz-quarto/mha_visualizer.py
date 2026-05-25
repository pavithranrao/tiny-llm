# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "marimo",
#   "numpy",
#   "plotly",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.title("Multi-Head Attention Visualizer")


@app.cell(hide_code=True)
def title_md(mo):
    mo.md(
        r"""
    # Multi-Head Attention — Step by Step

    Walk through **every operation** in multi-head attention, from raw input to
    final output. Each step includes:

    - 📊 **Visual** — interactive heatmaps with real numbers
    - 📖 **Analogy** — a real-world comparison to build intuition
    - ❓ **Why?** — the purpose of this operation
    - ⚠️ **What if we didn't?** — what goes wrong without it
    - 🔑 **Key Insight** — the core takeaway

    We use a small example: **4 tokens, embedding dim 6, 2 heads** (head_dim = 3).
    """
    )


@app.cell(hide_code=True)
def imports():
    import math
    import random

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return math, mo, np, go, make_subplots, random


@app.cell(hide_code=True)
def math_utils(math, random):
    def softmax_vals(vals):
        mx = max(vals)
        exps = [math.exp(v - mx) for v in vals]
        s = sum(exps)
        return [e / s for e in exps]

    def rand_matrix(rows, cols, lo=-2.0, hi=2.0):
        return [[random.uniform(lo, hi) for _c in range(cols)] for _r in range(rows)]

    def matmul(a, b):
        ra, ca, cb = len(a), len(a[0]), len(b[0])
        return [
            [sum(a[ai][ak] * b[ak][bj] for ak in range(ca)) for bj in range(cb)]
            for ai in range(ra)
        ]

    def mat_transpose(m):
        return [[m[mi][mj] for mi in range(len(m))] for mj in range(len(m[0]))]

    return softmax_vals, rand_matrix, matmul, mat_transpose


@app.cell(hide_code=True)
def mha_class(rand_matrix, matmul, mat_transpose, softmax_vals, math, random):
    class MHA:
        """Full multi-head attention — stores all intermediate results."""

        def __init__(self, seq_len=4, embed_dim=6, num_heads=2, seed=42):
            random.seed(seed)
            self.seq_len = seq_len
            self.embed_dim = embed_dim
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads

            self.Q_in = rand_matrix(seq_len, embed_dim)
            self.K_in = rand_matrix(seq_len, embed_dim)
            self.V_in = rand_matrix(seq_len, embed_dim)

            self.Wq = rand_matrix(embed_dim, embed_dim, -1, 1)
            self.Wk = rand_matrix(embed_dim, embed_dim, -1, 1)
            self.Wv = rand_matrix(embed_dim, embed_dim, -1, 1)
            self.Wo = rand_matrix(embed_dim, embed_dim, -1, 1)

            self.Q = matmul(self.Q_in, mat_transpose(self.Wq))
            self.K = matmul(self.K_in, mat_transpose(self.Wk))
            self.V = matmul(self.V_in, mat_transpose(self.Wv))

            hd = self.head_dim
            self.Q_heads = [
                [row[h * hd : (h + 1) * hd] for h in range(num_heads)]
                for row in self.Q
            ]
            self.K_heads = [
                [row[h * hd : (h + 1) * hd] for h in range(num_heads)]
                for row in self.K
            ]
            self.V_heads = [
                [row[h * hd : (h + 1) * hd] for h in range(num_heads)]
                for row in self.V
            ]

            self.attn_weights = []
            self.attn_out = []
            scale = 1.0 / math.sqrt(hd)
            for h in range(num_heads):
                qh = [self.Q_heads[s][h] for s in range(seq_len)]
                kh = [self.K_heads[s][h] for s in range(seq_len)]
                vh = [self.V_heads[s][h] for s in range(seq_len)]
                scores = matmul(qh, mat_transpose(kh))
                scores = [[v * scale for v in row] for row in scores]
                w = [softmax_vals(row) for row in scores]
                self.attn_weights.append(w)
                self.attn_out.append(matmul(w, vh))

            self.merged = []
            for s in range(seq_len):
                row = []
                for h in range(num_heads):
                    row.extend(self.attn_out[h][s])
                self.merged.append(row)

            self.output = matmul(self.merged, mat_transpose(self.Wo))

    return (MHA,)


@app.cell
def compute(MHA):
    mha = MHA(seq_len=4, embed_dim=6, num_heads=2, seed=42)
    print(
        f"seq_len={mha.seq_len}  embed_dim={mha.embed_dim}  "
        f"num_heads={mha.num_heads}  head_dim={mha.head_dim}"
    )
    print("✅ MHA computed — all intermediates stored.")
    return (mha,)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def viz_helpers(np, go, make_subplots):
    def heatmap(data, title, colorscale="Blues", decimals=2):
        hm_arr = np.array(data)
        hm_fig = go.Figure(
            go.Heatmap(
                z=hm_arr,
                text=[[f"{v:.{decimals}f}" for v in row] for row in hm_arr],
                texttemplate="%{text}",
                textfont={"size": 9, "family": "monospace"},
                colorscale=colorscale,
                showscale=False,
            )
        )
        hm_fig.update_layout(
            title=title,
            width=max(280, hm_arr.shape[1] * 55 + 80),
            height=max(220, hm_arr.shape[0] * 55 + 80),
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
        )
        return hm_fig

    def side_by_side(matrices, titles, colorscales, main_title=""):
        sb_n = len(matrices)
        sb_fig = make_subplots(
            rows=1, cols=sb_n, subplot_titles=titles, horizontal_spacing=0.08,
        )
        for sb_i, (sb_mat, sb_cs) in enumerate(zip(matrices, colorscales)):
            sb_arr = np.array(sb_mat)
            sb_fig.add_trace(
                go.Heatmap(
                    z=sb_arr,
                    text=[[f"{v:.2f}" for v in row] for row in sb_arr],
                    texttemplate="%{text}",
                    textfont={"size": 8, "family": "monospace"},
                    colorscale=sb_cs, showscale=False,
                ),
                row=1, col=sb_i + 1,
            )
        sb_fig.update_layout(
            title=main_title, width=220 * sb_n + 80, height=300,
            margin=dict(l=20, r=20, t=70, b=20), showlegend=False,
        )
        sb_fig.update_xaxes(showticklabels=False)
        sb_fig.update_yaxes(showticklabels=False)
        return sb_fig

    def attn_heatmap(weights, seq_len):
        ah_labels = [f"tok {t}" for t in range(seq_len)]
        ah_fig = go.Figure()
        ah_nh = len(weights)
        for ah_h in range(ah_nh):
            ah_arr = np.array(weights[ah_h])
            ah_fig.add_trace(
                go.Heatmap(
                    z=ah_arr, x=ah_labels, y=ah_labels,
                    text=[[f"{v:.3f}" for v in row] for row in ah_arr],
                    texttemplate="%{text}", textfont={"size": 10},
                    colorscale="Oranges", zmin=0, zmax=1,
                    colorbar={"title": "Weight"},
                    visible=(ah_h == 0), name=f"Head {ah_h}",
                )
            )
        ah_buttons = [
            dict(
                label=f"Head {bh}",
                method="update",
                args=[{"visible": [bj == bh for bj in range(ah_nh)]}],
            )
            for bh in range(ah_nh)
        ]
        ah_fig.update_layout(
            updatemenus=[dict(
                buttons=ah_buttons, direction="down", showactive=True,
                x=0.5, xanchor="center", y=1.15, yanchor="top",
            )],
            title="Attention Weights (dropdown to switch heads)",
            xaxis_title="Key", yaxis_title="Query",
            width=520, height=500,
        )
        return ah_fig

    return heatmap, side_by_side, attn_heatmap


# ---------------------------------------------------------------------------
# Step-by-step
# ---------------------------------------------------------------------------


@app.cell(hide_code=True)
def pipeline_overview(go):
    po_steps = [
        "Input\nQ,K,V", "Project\nxWᵀ", "Reshape\nheads",
        "Transpose\nheads 1st", "Attention\nQKᵀ/√d·V",
        "Merge\nconcat", "Output\n×Wo",
    ]
    po_fig = go.Figure()
    po_fig.add_trace(
        go.Scatter(
            x=list(range(7)), y=[0] * 7,
            mode="markers+text",
            marker=dict(size=30, color=["#8b5cf6"] * 7, symbol="square"),
            text=po_steps, textposition="top center", textfont=dict(size=10),
        )
    )
    for po_i in range(6):
        po_fig.add_annotation(
            x=po_i + 1 - 0.12, y=0, ax=po_i + 0.12, ay=0,
            arrowhead=2, arrowsize=1.5, arrowcolor="#64748b", arrowwidth=2,
        )
    po_fig.update_layout(
        title="The 7 Steps of Multi-Head Attention",
        width=850, height=180,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=50, b=50),
        plot_bgcolor="white", showlegend=False,
    )
    po_fig.show()


# --- Step 0 ---------------------------------------------------------------


@app.cell(hide_code=True)
def step0_md(mo):
    mo.md(
        r"""
    ---

    ## Step 0: Input — What are Q, K, V?

    Three input matrices of shape `(seq_len × embed_dim)`, plus four learned weight matrices.

    > 📖 **Analogy**: Think of a library: you walk in with a question (**Query**), the books have titles on the spine (**Keys**) and content inside (**Values**).

    > 🔑 **Key Insight**: The weight matrices (Wq, Wk, Wv) are learned during training — they transform raw embeddings into useful query/key/value spaces.
    """
    )


@app.cell
def step0_viz(side_by_side, mha):
    side_by_side(
        [mha.Q_in, mha.K_in, mha.V_in],
        ["Query (L×E)", "Key (L×E)", "Value (L×E)"],
        ["Blues", "Greens", "Purples"],
        "Input Matrices",
    ).show()

    side_by_side(
        [mha.Wq, mha.Wk, mha.Wv, mha.Wo],
        ["Wq (E×E)", "Wk (E×E)", "Wv (E×E)", "Wo (E×E)"],
        ["Blues", "Greens", "Purples", "Reds"],
        "Learned Weight Matrices",
    ).show()


# --- Step 1 ---------------------------------------------------------------


@app.cell(hide_code=True)
def step1_md(mo):
    mo.md(
        r"""
    ---

    ## Step 1: Linear Projection — Creating Different Views

    Each input is multiplied by its weight matrix: **Q = Input × Wqᵀ**, **K = Input × Wkᵀ**, **V = Input × Wvᵀ**.

    > 📖 **Analogy**: Like putting on different colored glasses — red for Query, green for Key, blue for Value.

    > ⚠️ **What if we didn't?** Without projection, you'd compare raw token embeddings directly. No flexibility to learn *what aspects* of a token matter.

    > 🔑 **Key Insight**: y = xWᵀ — the weight matrix rotates the vector into a new space where matching is more meaningful.
    """
    )


@app.cell
def step1_viz(side_by_side, mha):
    for s1_label, s1_inp, s1_proj, s1_cs in [
        ("Query", mha.Q_in, mha.Q, "Blues"),
        ("Key", mha.K_in, mha.K, "Greens"),
        ("Value", mha.V_in, mha.V, "Purples"),
    ]:
        side_by_side(
            [s1_inp, s1_proj],
            [f"{s1_label} Input", f"{s1_label} Projected"],
            [s1_cs, s1_cs],
            f"{s1_label}: Input → Projected  (× W{s1_label[0].lower()}ᵀ)",
        ).show()


# --- Step 2 ---------------------------------------------------------------


@app.cell(hide_code=True)
def step2_md(mo):
    mo.md(
        r"""
    ---

    ## Step 2: Reshape — Splitting into Attention Heads

    Each projected vector is split into `num_heads` chunks of size `head_dim`.
    For our example: a 6-dim vector → 2 chunks of 3 dims.

    > 📖 **Analogy**: A team of 2 analysts, each gets their own slice of information. Analyst 1 looks at dims 0-2, Analyst 2 at dims 3-5.

    > ⚠️ **What if we didn't?** With 1 head, all attention mixes into one pattern. Multiple heads learn SEVERAL independent relationship types at once.

    > 🔑 **Key Insight**: head_dim = embed_dim / num_heads. We redistribute capacity, not lose it.
    """
    )


@app.cell
def step2_viz(np, go, make_subplots, mha):
    s2_fig = make_subplots(
        rows=mha.seq_len, cols=mha.num_heads,
        vertical_spacing=0.06, horizontal_spacing=0.1,
        subplot_titles=["Head 0", "Head 1"] * mha.seq_len,
    )
    for s2_s in range(mha.seq_len):
        for s2_h in range(mha.num_heads):
            s2_arr = np.array(mha.Q_heads[s2_s][s2_h]).reshape(1, -1)
            s2_fig.add_trace(
                go.Heatmap(
                    z=s2_arr,
                    text=[[f"{v:.2f}" for v in mha.Q_heads[s2_s][s2_h]]],
                    texttemplate="%{text}",
                    textfont={"size": 10, "family": "monospace"},
                    colorscale="Blues" if s2_h == 0 else "Tealgrn",
                    showscale=False,
                ),
                row=s2_s + 1, col=s2_h + 1,
            )
    s2_fig.update_layout(
        title="Q reshaped: each token split into 2 heads  (blue=H0, teal=H1)",
        width=420, height=120 * mha.seq_len + 60,
        showlegend=False, margin=dict(l=20, r=20, t=60, b=20),
    )
    s2_fig.update_xaxes(showticklabels=False)
    s2_fig.update_yaxes(showticklabels=False)
    s2_fig.show()


# --- Step 3 ---------------------------------------------------------------


@app.cell(hide_code=True)
def step3_md(mo):
    mo.md(
        r"""
    ---

    ## Step 3: Transpose — Making Heads Independent

    Reshape gives `(seq_len, num_heads, head_dim)`. Transpose to `(num_heads, seq_len, head_dim)` so each head becomes an independent batch.

    > 📖 **Analogy**: Splitting a conference room into separate breakout rooms — each head gets its own room for independent conversations.

    > ⚠️ **What if we didn't?** QKᵀ would mix head and sequence indices — total nonsense!

    > 🔑 **Key Insight**: Shape (H, L, D) means: for each of H heads, compute attention over L tokens each with D dimensions.
    """
    )


@app.cell
def step3_viz(go):
    s3_fig = go.Figure()
    s3_fig.add_trace(
        go.Scatter(
            x=[0] * 4, y=list(range(3, -1, -1)),
            mode="markers+text",
            marker=dict(size=25, color="#3b82f6", symbol="square"),
            text=["T0:[h0|h1]", "T1:[h0|h1]", "T2:[h0|h1]", "T3:[h0|h1]"],
            textposition="middle right", textfont=dict(size=11),
            name="(Seq, Heads, Dim)",
        )
    )
    s3_fig.add_annotation(
        x=0.5, y=1.5, ax=0.15, ay=1.5,
        arrowhead=2, arrowsize=2, arrowcolor="#22c55e", arrowwidth=3,
    )
    s3_fig.add_annotation(
        x=0.33, y=1.85, text="transpose",
        showarrow=False, font=dict(size=14, color="#22c55e", family="monospace"),
    )
    s3_fig.add_trace(
        go.Scatter(
            x=[1] * 2, y=[3, 0],
            mode="markers+text",
            marker=dict(size=25, color="#22c55e", symbol="square"),
            text=["Head 0: [T0,T1,T2,T3]", "Head 1: [T0,T1,T2,T3]"],
            textposition="middle right", textfont=dict(size=11),
            name="(Heads, Seq, Dim)",
        )
    )
    s3_fig.update_layout(
        title="Transpose: (Seq, Heads, Dim) → (Heads, Seq, Dim)",
        width=680, height=320,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=True, legend=dict(x=0.55, y=0.3),
        plot_bgcolor="white", margin=dict(l=20, r=20, t=60, b=20),
    )
    s3_fig.show()


# --- Step 4 ---------------------------------------------------------------


@app.cell(hide_code=True)
def step4_md(mo):
    mo.md(
        r"""
    ---

    ## Step 4: Attention — Where the Magic Happens ✨

    For each head **independently**:

    1. **Scores** = QKᵀ  (dot product of every query with every key)
    2. **Scale** = scores / √d
    3. **Weights** = softmax(scores)  (→ probabilities summing to 1)
    4. **Output** = weights × V  (weighted average of values)

    > 📖 **Analogy**: Each token asks "which other tokens are relevant to me?" Attention weights are the answer — a probability distribution over all tokens.

    > ⚠️ **What if we didn't scale?** With large dims, QKᵀ blows up → softmax becomes argmax → always picks ONE token. No smooth blending!

    > 🔑 **Key Insight**: softmax(QKᵀ/√d) × V — each output token is a blend of all inputs, weighted by relevance.
    """
    )


@app.cell
def step4_weights(attn_heatmap, mha, np):
    attn_heatmap(mha.attn_weights, mha.seq_len).show()

    for s4w_h in range(mha.num_heads):
        print(f"\n=== Head {s4w_h} Attention Weights ===")
        s4w_arr = np.array(mha.attn_weights[s4w_h])
        for s4w_i in range(mha.seq_len):
            s4w_row = "  ".join(f"{s4w_arr[s4w_i, j]:.3f}" for j in range(mha.seq_len))
            print(f"  Token {s4w_i}: [{s4w_row}]  (sum={s4w_arr[s4w_i].sum():.3f})")
        print("  (Each row sums to 1.0 — a probability distribution!)")


@app.cell
def step4_output(heatmap, mha):
    for s4o_h in range(mha.num_heads):
        heatmap(
            mha.attn_out[s4o_h],
            f"Attention Output — Head {s4o_h}  (L × head_dim)",
            colorscale="Greens",
        ).show()


# --- Step 5 ---------------------------------------------------------------


@app.cell(hide_code=True)
def step5_md(mo):
    mo.md(
        r"""
    ---

    ## Step 5: Merge — Reassembling Perspectives

    Concatenate all head outputs back into `(seq_len × embed_dim)`.
    For our example: Head 0's 3 dims + Head 1's 3 dims = 6 dims per token.

    > 📖 **Analogy**: Reassembling a jigsaw puzzle — each head's perspective stitches into a complete picture.

    > 🔑 **Key Insight**: (H, L, D) → (L, H×D) = (L, embed_dim). Just memory rearrangement, no computation.
    """
    )


@app.cell
def step5_viz(side_by_side, mha):
    side_by_side(
        [mha.attn_out[0], mha.attn_out[1], mha.merged],
        ["Head 0 output", "Head 1 output", "Merged (concat)"],
        ["Greens", "Tealgrn", "Purples"],
        f"Head outputs → Merged  ({mha.num_heads} × {mha.head_dim} = {mha.embed_dim} dims)",
    ).show()


# --- Step 6 ---------------------------------------------------------------


@app.cell(hide_code=True)
def step6_md(mo):
    mo.md(
        r"""
    ---

    ## Step 6: Output Projection — Final Mix

    Multiply the merged matrix by Wo: **Output = Merged × Woᵀ**

    > 📖 **Analogy**: A master painter (Wo) mixes the colors from all analysts into a coherent final painting.

    > ⚠️ **What if we didn't?** The model would have no way to learn that "head 1 is about syntax and should be weighted more."

    > 🔑 **Key Insight**: The ENTIRE MHA block: Output = Concat(head₁, head₂, ...) × Wo, where each headᵢ = softmax(QᵢKᵢᵀ/√d) × Vᵢ.
    """
    )


@app.cell
def step6_viz(side_by_side, mha):
    side_by_side(
        [mha.merged, mha.Wo, mha.output],
        ["Merged (L×E)", "Wo (E×E)", "Final Output (L×E)"],
        ["Purples", "Reds", "Oranges"],
        "Final: Merged × Woᵀ = Output",
    ).show()

    print(f"\n✅ Output shape: ({mha.seq_len}, {mha.embed_dim}) — same as input!")
    print("Multi-head attention is a residual-ready transformation.")


# --- Summary --------------------------------------------------------------


@app.cell(hide_code=True)
def summary_md(mo):
    mo.md(
        r"""
    ---

    ## Summary: The Complete MHA Pipeline

    | Step | Operation | Shape Transform |
    |------|-----------|-----------------|
    | 0 | Input Q, K, V | (L, E) × 3 |
    | 1 | Linear projection | (L, E) → (L, E) × 3 |
    | 2 | Reshape / split heads | (L, E) → (L, H, D) |
    | 3 | Transpose | (L, H, D) → (H, L, D) |
    | 4 | Scaled dot-product attention | → weights (H, L, L), output (H, L, D) |
    | 5 | Merge (concatenate heads) | (H, L, D) → (L, E) |
    | 6 | Output projection | (L, E) × Woᵀ → (L, E) |

    ### The Full Formula

    $$\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \cdot W_o$$

    where each head:

    $$\text{head}_i = \text{softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$$

    > **All** operations are differentiable — the entire block learns end-to-end via backpropagation. That's the beauty of the Transformer architecture!
    """
    )


if __name__ == "__main__":
    app.run()
