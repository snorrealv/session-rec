# syntax=docker/dockerfile:1

# Use the official Miniconda3 image as the base image
FROM continuumio/miniconda3

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/develop/develop-images/dockerfile_best-practices/#user
ARG UID=10001
# RUN adduser \
#     --disabled-password \
#     --gecos "" \
#     --home "/nonexistent" \
#     --shell "/sbin/nologin" \
#     --no-create-home \
#     --uid "${UID}" \
#     appuser

RUN adduser -ms /bin/bash -u 1001 appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.
# Leverage a bind mount to environment_cpu.yml to avoid having to copy it into
# this layer.
COPY docker/cpu/environment_cpu.yml .
COPY --chown=appuser:appuser . /app
# Create the Conda environment
RUN conda env create -f environment_cpu.yml

# Activate the Conda environment
SHELL ["conda", "run", "-n", "srec37", "/bin/bash", "-c"]

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
# CMD ["conda", "run", "-n", "srec37", "python3", "main.py"]
