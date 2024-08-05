FROM nvcr.io/nvidia/tensorflow:22.11-tf2-py3

RUN apt-get update && apt-get install -y python3-pip && apt-get install -y python3-venv
RUN python3 -m venv /env
RUN /bin/bash -c "source /env/bin/activate"

COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY ./models /models

EXPOSE 8888

ENTRYPOINT ["python3", "models/main.py", "--mode", "train", "--model", "lstm"]