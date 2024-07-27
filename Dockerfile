FROM nvcr.io/nvidia/tensorflow:22.11-tf1-py3

WORKDIR /app

COPY requirements.txt .

RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT python3 models/main.py --model lstm --mode train