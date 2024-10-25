# Interpretation of Multimodal LLMs Trained for Medical Domain
This repository consists of several jupyter notebooks with different LLM interpretation techniques
## Problem
Multimodal AI models like LLaVA-Med, which integrate medical text and images, are difficult to interpret, posing risks to transparency and trust in clinical decision-making. Without clear understanding of how these models process and combine different inputs, their application in healthcare is limited.
## Solution
This project focuses on interpreting the internal mechanisms of the LLaVA-Med model by applying classical interpretation methods and analyzing model representations (e.g., linearity, contextualization). This will enhance transparency and ensure safer, more reliable use in medical research and diagnostics.
## Goal
The main goal of this project is to improve the interpretability of the multimodal LLaVA-Med model, ensuring its safe and transparent application in medical research and diagnostics.
## Way to Reproduce
```bash
git clone https://github.com/rntprch/Interpretation-multimodal-LLMs-trained-for-medical-domain.git
cd llava_env
docker build -t llava_model .
cd ../
docker run -it --gpus='"device=1"' -p 10000:10000 -p 40000:40000 -p 7860:7860 -p 510:510 -v $(pwd):/workspace --name llava_container llava_model
```
After creating docker environment connect to running container with VS code or any interpreter in interactive mode.
## Authros
**Rinat Prochii**, Skoletch

**Fedor Gubanov**, Skoltech

**Iana Kulichenko**, Skoltech

**Polina Druzhinina**, Skoltech
