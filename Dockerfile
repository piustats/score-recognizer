FROM pytorch/pytorch as runtime

WORKDIR /app
# Essentials: developer tools, build tools, OpenBLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim unzip openssh-client wget \
    build-essential cmake software-properties-common \
    libopenblas-dev ffmpeg libsm6 libxext6
# Install tesseract
RUN add-apt-repository -y ppa:alex-p/tesseract-ocr5 && apt-get update && apt-get install -y --no-install-recommends tesseract-ocr

RUN pip install cython && git clone https://github.com/cocodataset/cocoapi.git \
 && cd cocoapi/PythonAPI \
 && python setup.py build_ext install

COPY ./requirements-docker.txt ./requirements.txt

RUN pip install -r requirements.txt

# Look at the readme to download both of these models
COPY models/faster-rcnn-model-small ./models/faster-rcnn-model-small
COPY models/tesseract-karnivore-model.zip ./models/tesseract-karnivore-model.zip
RUN unzip ./models/tesseract-karnivore-model.zip

ENV CNN_MODEL=/app/models/faster-rcnn-model-small
ENV TESSERACT_MODEL=/app/models/tess-model


COPY . .

EXPOSE 80
CMD ["python3", "server.py"]