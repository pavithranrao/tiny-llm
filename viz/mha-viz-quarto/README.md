# Multi-Head Attention Visualizer (Marimo + Quarto)

A step-by-step interactive visualization of multi-head attention, built with [Marimo](https://marimo.io/).

## Run interactively

```bash
# From project root
uv run marimo edit viz/mha-viz-quarto/mha_visualizer.py
```

Opens a reactive notebook in your browser at `http://localhost:2718`.

## Export to static HTML

```bash
uv run marimo export html viz/mha-viz-quarto/mha_visualizer.py \
  --output viz/mha-viz-quarto/mha_visualizer.html
```

## Export to Quarto

```bash
# First install quarto (if not already installed)
# See https://quarto.org/docs/get-started/

# Export as Markdown that Quarto can render
uv run marimo export md viz/mha-viz-quarto/mha_visualizer.py \
  --output viz/mha-viz-quarto/mha_visualizer.qmd

# Render with Quarto
quarto render viz/mha-viz-quarto/mha_visualizer.qmd
```

Or export directly to HTML via Quarto's notebook support:

```bash
quarto render viz/mha-viz-quarto/mha_visualizer.py
```

## Publish to GitHub Pages

```bash
# Export to docs/ for GitHub Pages
uv run marimo export html viz/mha-viz-quarto/mha_visualizer.py \
  --output docs/mha-viz/index.html

# Or use Quarto to publish
quarto publish ghpages viz/mha-viz-quarto/
```

## What it covers

The 7 steps of multi-head attention, with:
- **Interactive Plotly heatmaps** for every matrix
- **Attention weight heatmap** with dropdown to switch heads
- **Conceptual explanations** (analogy, why, what-if, key insight) per step
- **Shape annotations** showing dimension transforms

| Step | Operation |
|------|-----------|
| 0 | Input: Q, K, V matrices |
| 1 | Linear projection (× Wq, Wk, Wv) |
| 2 | Reshape into attention heads |
| 3 | Transpose for head independence |
| 4 | Scaled dot-product attention |
| 5 | Merge (concatenate heads) |
| 6 | Output projection (× Wo) |
