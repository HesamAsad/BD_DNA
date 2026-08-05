# Architecture slides

`bd3lm_architectures.html` is a self-contained reveal.js deck covering the
BD3-LM block-diffusion objective and the two long-context backbones in this
repository: the dual-stream Transformer (`models/dit_dual.py`) and the
leakage-safe bidirectional SSM (`models/bidirectional_ssm.py`).

Open it directly in a browser -- no server needed:

    firefox docs/slides/bd3lm_architectures.html

Speaker notes: `s`. Overview: `esc`. Print to PDF: append `?print-pdf` to the
URL and use the browser's print dialog.

## Vendored libraries

The deck loads reveal.js and MathJax from `vendor/`, which is gitignored so the
repository does not carry 2.3 MB of third-party bundles. Re-fetch them with:

    cd docs/slides && mkdir -p vendor/theme
    B=https://cdn.jsdelivr.net/npm/reveal.js@5.1.0/dist
    curl -sSfL -o vendor/reveal.css "$B/reveal.css"
    curl -sSfL -o vendor/reveal.js  "$B/reveal.js"
    curl -sSfL -o vendor/theme/white.css "$B/theme/white.css"
    curl -sSfL -o vendor/tex-svg.js "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"
    sed -i '/@import url(.\/fonts\/source-sans-pro\/source-sans-pro.css);/d' vendor/theme/white.css

The `sed` drops the theme's webfont import so the deck makes no network
requests at all.
