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
## Statistics
### Linear Score
Source: https://github.com/AIRI-Institute/LLM-Microscope

#### Full Embedding (Text + Image)
![image](https://github.com/user-attachments/assets/f9a4eb82-e8c9-4808-ab5e-76229d395bcd)
![image](https://github.com/user-attachments/assets/8b23ede0-d5c7-414b-9bbc-368943641d34)
#### Text Embedding
![image](https://github.com/user-attachments/assets/7463a5ba-4bb9-4558-b176-679e3349b7fd)
![image](https://github.com/user-attachments/assets/56272bdd-b835-4bc8-9dba-ca676940ce1e)
#### Visual Embedding
![image](https://github.com/user-attachments/assets/aadd60b9-df02-4466-bcd9-eaa1e63d9d40)
![image](https://github.com/user-attachments/assets/47902474-7de8-489c-9aea-3197e4798be0)

### Anisotropy & Intrinsic Dimension
Source: https://aclanthology.org/2024.findings-eacl.58.pdf

#### Full Embedding (Text + Image)
![image](https://github.com/user-attachments/assets/4c877446-a702-4dd0-9557-f46d62e3575b)

#### Text Embedding
![image](https://github.com/user-attachments/assets/9535a47e-2177-4686-b6bb-e8db814abe6f)
![image](https://github.com/user-attachments/assets/af41b78a-7a9b-462b-bda1-39f601a6f8f0)
#### Visual Embedding
![image](https://github.com/user-attachments/assets/f2e73387-2241-4853-a039-539ca1db4006)
![image](https://github.com/user-attachments/assets/1cc2cc2a-eccb-4829-9d3f-cdff3a0779e1)

### Contextualization
Source: https://aclanthology.org/D19-1006.pdf
![image](https://github.com/user-attachments/assets/a3ffa06c-fa68-4cbd-8292-eabc20d68058)

### Top Ten Tokens Among Different Layers
Source: https://multimodal-interpretability.csail.mit.edu/Multimodal-Neurons-in-Text-Only-Transformers/
![image](https://github.com/user-attachments/assets/f74424ce-2e14-47b4-bb8a-5a57db47bb72)

### GradCam
Source: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
Model recieves same question for each image and some parts of answer were pregenerated in order to recieve only meaningful tokens.

#### Attention Distribution Based on the Question Tokens
![image](https://github.com/user-attachments/assets/43d701fc-f000-4e37-ab48-db10c8d79e7b)
#### Attention Distribution Based on the Last Predicted Token
![image](https://github.com/user-attachments/assets/0a78693d-c968-4f3a-b4d5-b5b18b5f9dd4)


## Authros
**Rinat Prochii**, Skoletch

**Fedor Gubanov**, Skoltech

**Iana Kulichenko**, Skoltech

**Polina Druzhinina**, Skoltech
