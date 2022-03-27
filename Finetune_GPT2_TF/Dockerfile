# Docker file to run a container that will run the finetune_gpt2_alt.py 
# in Python 3 for Huggingface (no GPU).

# Load tensorflow image for huggingface and Python 3.
FROM huggingface/transformers-gpu

# Set locale for variable (pulled from dockerfile in original OpenAI
# GPT2 repository).
ENV LANG=C.UTF-8

# Create a directory in the docker container. Set the working directory
# in the container to that newly created directory and then add all
# files from the current directory in the host to the working directory
# in the container.
RUN mkdir /finetune-gpt2
WORKDIR /finetune-gpt2
ADD . /finetune-gpt2

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Download the data.
#RUN curl -LO https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
#RUN tar -xf LJSpeech-1.1.tar.bz2
#RUN mkdir datasets
#RUN cp -r LJSpeech-1.1 datasets/

# Run the finetune_gpt2_alt.py program.
RUN ["python3", "preprocess_stories.py"]
CMD ["python3", "finetune_gpt2_alt.py"]