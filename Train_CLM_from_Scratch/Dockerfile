# Docker file to run a container that will run the 
# training_clm_from_scratch.py in Python 3 for Huggingface (no GPU).

# Load tensorflow image for huggingface and Python 3.
FROM huggingface/transformers-gpu

# Set locale for variable (pulled from dockerfile in original OpenAI
# GPT2 repository).
ENV LANG=C.UTF-8

# Create a directory in the docker container. Set the working directory
# in the container to that newly created directory and then add all
# files from the current directory in the host to the working directory
# in the container.
RUN mkdir /train-clm
WORKDIR /train-clm
ADD . /train-clm

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Run the training_clm_from_scratch.py program.
CMD ["python3", "training_clm_from_scratch.py"]