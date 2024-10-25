# NVIDIA's CUDA base image
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

# Install miniconda manually
RUN apt-get update && apt-get install -y wget bzip2 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Path for miniconda
ENV PATH=/opt/miniconda/bin:$PATH

# Set the working directory inside the container
WORKDIR /model

# Clone the LLaVA-Med repository
RUN apt-get update && apt-get install -y git && \
    git clone https://github.com/microsoft/LLaVA-Med.git && \
    cd LLaVA-Med

# Install text editors
RUN apt-get install -y vim nano

# Set the working directory to LLaVA-Med
WORKDIR /model/LLaVA-Med

# Create and activate the Conda environment
RUN conda init bash && conda create -n llava-med python=3.10 -y

# Activate the Conda environment and install necessary packages
RUN /bin/bash -c "source activate llava-med && \
    pip install --upgrade pip && \
    pip install -e ."

RUN echo "conda activate llava-med" >> ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

# Instal additional libraries if needed
COPY requirements.txt .
RUN pip3 install -r requirements.txt

RUN pip3 install jupyter jupyterlab
RUN pip3 install --no-cache-dir ipywidgets

RUN mkdir -p /model/LLaVA-Med/llava/data/pmc /model/LLaVA-Med/llava/data/images

RUN wget "https://hanoverprod.z21.web.core.windows.net/med_llava/eval/llava_med_eval_qa50_qa.jsonl" \
	-O /model/LLaVA-Med/llava/data/llava_med_eval_qa50_qa.jsonl

RUN wget "https://hanoverprod.z21.web.core.windows.net/med_llava/multimodal_chat_eval/llava_med_test_image_urls.jsonl" \
	-O /model/LLaVA-Med/llava/data/llava_med_test_image_urls.jsonl

COPY download_images.py /model/LLaVA-Med/llava/data/download_images.py

RUN python /workspace/LLaVA-Med/llava/data/download_images.py \
    --input_path /workspace/LLaVA-Med/llava/data/llava_med_test_image_urls.jsonl \
    --pmc_output_path /workspace/LLaVA-Med/llava/data/pmc \
    --images_output_path /workspace/LLaVA-Med/llava/data/images


WORKDIR /workspace

EXPOSE 10000 40000 7860 510

# Set the default command to activate the Conda environment when starting the container
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=510", "--no-browser", "--allow-root"]
