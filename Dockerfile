FROM jitesoft/tesseract-ocr as tesseract

FROM pytorch/pytorch as runtime

WORKDIR /app
# install pycocotools
# Essentials: developer tools, build tools, OpenBLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip openssh-client wget \
    build-essential cmake \
    libopenblas-dev ffmpeg libsm6 libxext6
RUN pip install cython && git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

COPY ./requirements-docker.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY --from=tesseract /usr/local/bin/tesseract /usr/local/bin/tesseract

COPY . .
CMD ["python3", "main.py"]