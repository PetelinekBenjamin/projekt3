# Uporabite sliko z Python 3.9
FROM python:3.9

# Postavite delovno mapo v Dockerju
WORKDIR /app

# Kopirajte python skripto v delovno mapo
COPY notebooks/Projekt/api.py .

# Kopirajte modele in scalerje v delovno mapo
COPY models /app/models

COPY data/processed/train_prod.csv /app/data/processed/

# Namesti zahtevane knjižnice z določenimi verzijami
RUN pip install --no-cache-dir \
    tensorflow==2.16.1 \
    scikit-learn==1.5.0 \
    pandas==2.2.2 \
    requests==2.32.2 \
    great-expectations==0.18.15 \
    matplotlib==3.9.0 \
    mlflow==2.13.0 \
    flake8==7.0.0 \
    evidently==0.4.25 \
    black==24.4.2 \
    jupyter==1.0.0 \
    dvc==3.51.1 \
    dvc-s3==3.2.0 \
    pymongo==4.6.3 \
    flask_cors==3.0.10

# Nastavite okoljske spremenljivke za CPU uporabo (ker vaš model uporablja CPU)
ENV CUDA_VISIBLE_DEVICES -1

# Izpostavite potrebne vrata
EXPOSE 5000

# Zaženite vaš Flask API
CMD ["python", "api.py"]
