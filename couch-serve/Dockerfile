FROM python:3.14-slim

LABEL org.opencontainers.image.source=https://github.com/Amarcode-Studios/Windlet.Templates

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --no-compile -r requirements.txt

COPY model.py tokeniser.py serve.py ./

ENV PYTHONUNBUFFERED=True \
    TORCH_DISABLE_NATIVE_JIT=1 \
    HOME=/tmp

EXPOSE 8080
USER 65534:65534
ENTRYPOINT ["python", "serve.py"]
CMD ["--checkpoint", "/input/checkpoint.pt"]
