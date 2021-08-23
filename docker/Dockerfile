FROM nvcr.io/nvidia/pytorch:21.04-py3
ARG PYTHON_DEPS=""
# ARG PYTHON_VERSION=3.9

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Recife

RUN ln -s /usr/bin/python3 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsasl2-dev \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    wget

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt



# RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
#     chmod +x ~/miniconda.sh && \
#     ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
# #     /opt/conda/bin/conda install -y -c pytorch magma-cuda100 && \
# #     /opt/conda/bin/conda clean -ya
# #ENV PATH /opt/conda/bin:$PATH
# ENV PATH="/opt/conda/bin:${PATH}"
# # # This must be done before pip so that requirements.txt is available
# # WORKDIR /opt/pytorch
# # COPY . .

# # RUN git submodule sync && git submodule update --init --recursive
# # RUN TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
# #     CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
# #     pip install -v .

# # RUN if [ "$WITH_TORCHVISION" = "1" ] ; then git clone https://github.com/pytorch/vision.git && cd vision && pip install -v . ; else echo "building without torchvision" ; fi

# # RUN pip3 install --no-cache-dir -U install setuptools pip
# # RUN pip3 install --no-cache-dir cupy-cuda102==9.0.0a2 scipy optuna
# # RUN pip3 install --upgrade pip && pip3 install torch==1.7.0+cu110 # torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
# # RUN pip3 install torch torchvision
# # RUN pip3 install torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# RUN pip3 install numba==0.50.1 \
#     tqdm==4.42.1 \
#     seaborn==0.10.0 \
#     pytest==5.3.5 \
#     tabulate==0.8.3 \
#     pandas==1.1.4 \
#     Cython==0.29.15 \
#     psutil==5.6.7 \
#     numpy \
#     statsmodels==0.11.0 \
#     mlflow==1.9.1 \
#     SQLAlchemy==1.3.13 \
#     scipy==1.4.1 \
#     joblib==0.14.1 \
#     Click==7.0 \
#     matplotlib==3.1.3 \
#     confuse==1.4.0 \
#     dill==0.3.3 \
#     holoviews==1.13.5 \
#     imblearn \
#     KDEpy==1.0.10 \
#     pandas_bokeh==0.5.2 \
#     pandas_profiling \
#     plotly==4.13.0 \
#     pycm==3.0 \
#     ray==1.0.1.post1 \
#     scikit_learn==0.23.2 \
#     fuzzy-c-means \
#     cupy-cuda111 \
#     xgboost \
#     pylama
# # pip3 install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html && \
# # pip3 install torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
# # RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# # RUN pip3 install torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# WORKDIR /workspace
# RUN chmod -R a+w .

# # Create a non-root user to use if preferred - see https://aka.ms/vscode-remote/containers/non-root-user.
# #    groupadd --gid $USER_GID $USERNAME && \
# #    useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME && \
# # [Optional] Add sudo support for the non-root user
# #    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
# #    chmod 0440 /etc/sudoers.d/$USERNAME
# #USER $USERNAME
# # RUN cd /tmp && bash -c "$(wget -q -O - https://linux.kite.com/dls/linux/current)"
# # RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# # ARG USERNAME=fcf
# # ARG USER_UID=1000
# # ARG USER_GID=$USER_UID

# # RUN groupadd --gid $USER_GID $USERNAME && \
# #     useradd -ms /bin/bash $USERNAME --uid $USER_UID --gid $USER_GID

# # USER $USERNAME


# ############### AIRFLOW


# # Never prompt the user for choices on installation/configuration of packages
# ENV DEBIAN_FRONTEND noninteractive
# ENV TERM linux

# # Airflow
# # ENV AIRFLOW_HOME=${AIRFLOW_USER_HOME}

# # Define en_US.
# ENV LANGUAGE en_US.UTF-8
# ENV LANG en_US.UTF-8
# ENV LC_ALL en_US.UTF-8
# ENV LC_CTYPE en_US.UTF-8
# ENV LC_MESSAGES en_US.UTF-8

# # Disable noisy "Handling signal" log messages:
# # ENV GUNICORN_CMD_ARGS --log-level WARNING

# RUN set -ex \
#     && buildDeps=' \
#     freetds-dev \
#     libkrb5-dev \
#     libsasl2-dev \
#     libssl-dev \
#     libffi-dev \
#     libpq-dev \
#     git \
#     ' \
#     && apt-get update -yqq \
#     && apt-get upgrade -yqq \
#     && apt-get install -yqq --no-install-recommends \
#     $buildDeps \
#     freetds-bin \
#     build-essential \
#     default-libmysqlclient-dev \
#     apt-utils \
#     curl \
#     rsync \
#     netcat \
#     locales \
#     && sed -i 's/^# en_US.UTF-8 UTF-8$/en_US.UTF-8 UTF-8/g' /etc/locale.gen \
#     && locale-gen \
#     && update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 \
#     # && useradd -ms /bin/bash -d ${AIRFLOW_USER_HOME} airflow \
#     && pip install -U pip setuptools wheel \
#     && pip install pytz \
#     && pip install pyOpenSSL \
#     && pip install ndg-httpsclient \
#     && pip install pyasn1 \
#     # && pip install apache-airflow[crypto,celery,postgres,hive,jdbc,mysql,ssh${AIRFLOW_DEPS:+,}${AIRFLOW_DEPS}]==${AIRFLOW_VERSION} \
#     && pip install 'redis==3.2' \
#     && if [ -n "${PYTHON_DEPS}" ]; then pip install ${PYTHON_DEPS}; fi \
#     # && apt-get purge --auto-remove -yqq $buildDeps \
#     && apt-get autoremove -yqq --purge \
#     && apt-get clean
# #    && rm -rf \
# #        /var/lib/apt/lists/* \
# #        /tmp/* \
# #        /var/tmp/* \
# #        /usr/share/man \
# #        /usr/share/doc \
# #        /usr/share/doc-base

# # COPY .devcontainer/script/entrypoint.sh /entrypoint.sh
# # COPY .devcontainer/config/airflow.cfg ${AIRFLOW_USER_HOME}/airflow.cfg

# # RUN pip install apache-airflow[s3,crypto,celery,postgres,hive,jdbc,mysql,ssh${AIRFLOW_DEPS:+,}${AIRFLOW_DEPS}]==${AIRFLOW_VERSION} retry

# COPY requirements.txt /requirements.txt
# RUN pip install -r /requirements.txt
# COPY .devcontainer/script/entrypoint.sh /entrypoint.sh
# # COPY .devcontainer/config/airflow.cfg ${AIRFLOW_USER_HOME}/airflow.cfg
# # COPY script/entrypoint.sh /entrypoint.sh
# # COPY config/airflow.cfg ${AIRFLOW_USER_HOME}/airflow.cfg
# # ENV PYTHONPATH "/usr/local/airflow/dags/src:${PYTHONPATH}:$AIRFLOW_HOME"

# # RUN chown -R airflow: ${AIRFLOW_USER_HOME}

# EXPOSE 8080 5555 8793 6379

# # USER airflow
# # WORKDIR ${AIRFLOW_USER_HOME}
# # ENTRYPOINT ["/entrypoint.sh"]
# # CMD ["webserver"]
