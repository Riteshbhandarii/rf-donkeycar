Dockerized environment for RL training with jetRacer

Build and run container with docker compose

docker compose up -d --build    (first time)
docker compose up -d            (other times)

Login with SSH to container

ssh racer@localhost

password is racer

The Python virtual environment "racervenv" is created automatically
during the Docker image build and activated automatically when
logging into the container.

The Docker container provides the common system environment for the course.
Python packages are installed into the racervenv virtual environment
inside the container.

The virtual environment allows additional Python packages to be installed
without root privileges.