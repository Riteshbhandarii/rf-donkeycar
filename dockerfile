FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    openssh-server \
    git \
    htop \
    python3.12 \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 22 8887

RUN mkdir /var/run/sshd

RUN echo 'Match User racer' >> /etc/ssh/sshd_config.d/10-racer.conf
RUN echo '  PasswordAuthentication yes' >> /etc/ssh/sshd_config.d/10-racer.conf

RUN useradd -ms /bin/bash racer
RUN echo "racer:racer" | chpasswd

WORKDIR /home/racer

COPY ./requirements.txt /home/racer/requirements.txt

RUN python3.12 -m venv /home/racer/racervenv

RUN /home/racer/racervenv/bin/pip install --upgrade pip
RUN /home/racer/racervenv/bin/pip install -r /home/racer/requirements.txt

RUN mkdir -p /home/racer/project

RUN chown -R racer:racer /home/racer

RUN chmod -R 777 /home/racer/project

RUN echo 'source /home/racer/racervenv/bin/activate' >> /home/racer/.bashrc

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]