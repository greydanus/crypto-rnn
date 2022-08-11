FROM python:2

COPY . /opt/crypto-rnn
WORKDIR /opt/crypto-rnn

RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "main.py" ]
