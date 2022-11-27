# Docker file to run a container that will run the text2imageSD.py in
# Python 3 (no GPU).

# Load Python 3.7 image.
FROM python:3.7.14

# Set locale for variable (pulled from dockerfile in original OpenAI
# GPT2 repository).
ENV LANG=C.UTF-8

# Create a directory in the docker container. Set the working directory
# in the container to that newly created directory and then add all
# files from the current directory in the host to the working directory
# in the container.
RUN mkdir /run-sd
WORKDIR /run-sd
ADD . /run-sd

# Set up a volume so that the current directory in the host is
# connected to the working directory in the container.

# Uncomment to download model to local storage and run it from there
# instead of relying on Huggingface Hub.
#RUN apt-get update -y
#RUN apt-get upgrade -y
#RUN apt-get install -y git
#RUN apt-get install -y git-lfs
#RUN git lfs install
#RUN git clone https://huggingface.co/CompVis/stable-diffusion-v1-4

# Install all required modules in the requirements.txt file.
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt

# Run the text2imageSD.py program.
CMD ["python3", "text2imageSD.py"]