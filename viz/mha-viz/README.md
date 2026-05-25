# Multi-Head Attention Visualizer (D3.js)

A step-by-step interactive visualization of multi-head attention, built with D3.js.

## View

Open the HTML file directly in any browser — no server or build step needed:

```bash
# From project root
open viz/mha-viz/mha_d3_viz.html
# or
xdg-open viz/mha-viz/mha_d3_viz.html
```

Or serve it:

```bash
python3 -m http.server 8765 --directory viz/mha-viz
# then open http://localhost:8765/mha_d3_viz.html
```

## What it covers

The 7 steps of multi-head attention, with:

- **D3.js heatmaps** with real computed numbers for every matrix
- **Interactive head selector** to switch attention weight views
- **Visual reshape diagram** showing how vectors split into heads
- **Transpose diagram** showing the dimension rearrangement
- **Callout boxes** (analogy, insight, warning) per step
- **Scroll-triggered animations** as you read through

| Step | Operation |
|------|-----------|
| 0 | Input: Q, K, V matrices |
| 1 | Linear projection (× Wq, Wk, Wv) |
| 2 | Reshape into attention heads |
| 3 | Transpose for head independence |
| 4 | Scaled dot-product attention |
| 5 | Merge (concatenate heads) |
| 6 | Output projection (× Wo) |
