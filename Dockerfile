FROM python:3.9-slim

WORKDIR /notebooks

COPY requirements.txt .

RUN pip install -r requirements.txt jupyterlab

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]