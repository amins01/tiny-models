# Tiny Models
Small generative models built and trained from scratch to run locally in the browser.

## Current Models
- **Semantic model:** DiT that generates a landscape image from a drawing (from a semantic palette). More info in the [blog post](https://amins01.github.io/tiny-models/#/semantic)

https://github.com/user-attachments/assets/cc07057f-9766-4138-8420-d5b643c4151c

- **Edit model:** DiT for text-based image editing. For the text encoder, I used a pretrained [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
<!-- ![edit](/assets/) -->

## Project Layout
- `frontend/` contains the code for the web app (React + TS) and deployed assets (e.g., models). Models of size > 100 MB must be split into parts in order to get around GitHub's file size limit (the path to each part must be included in `frontend/constants.ts`)
- `pipelines/` contains the code for modeling, training, sampling, and exporting to produce the ONNX checkpoints
