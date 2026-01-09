
FROM python:3.9-slim


WORKDIR /app



RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY . .


# (Önceki satırlar aynı kalsın...)

# Portu 7860 yapıyoruz çünkü Hugging Face bu portu sever
EXPOSE 7860

# Kullanıcı yetkisiyle çalıştır (Hugging Face güvenlik kuralı)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]