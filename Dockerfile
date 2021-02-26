FROM python:3.8
MAINTAINER Masashi Shibata <shibata_masashi@cyberagent.co.jp>

RUN pip install --upgrade pip

ADD ./requirements.txt /usr/src/requirements.txt
RUN pip install --no-cache-dir -r /usr/src/requirements.txt

# Installs google cloud sdk, this is mostly for using gsutil to export model.
# See https://cloud.google.com/ai-platform/training/docs/custom-containers-training
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup
# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

ADD ./tmp/train.csv /usr/src/tmp/train.csv
ADD ./main.py /usr/src/main.py
WORKDIR /usr/src

CMD ["python", "benchmark_aiplatform.py"]