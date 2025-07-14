FROM python:3.12.10-bookworm
RUN python -m pip install --upgrade pip
WORKDIR /workspace
COPY requirements.txt /workspace
RUN pip install -r requirements.txt

