# =============================================================================
# CompuCell3D worker image for AWS Batch.
#
# This is the piece that makes Option 3 work: the web application stays on Render
# (pip-only, 512 MB) and this image carries the engine it cannot host. The image is
# large (~2-4 GB) because CompuCell3D pulls VTK, which is exactly why it does not
# belong in the web dyno.
#
# CompuCell3D is installed from its OWN official conda channel. It is not on PyPI,
# so pip cannot be used here -- that constraint is the entire reason this file
# exists rather than a line in requirements.txt.
#
# Build and push:
#   docker build -t cc3d-worker deploy/cc3d
#   aws ecr create-repository --repository-name cc3d-worker --region us-east-2
#   aws ecr get-login-password --region us-east-2 \
#     | docker login --username AWS --password-stdin <acct>.dkr.ecr.us-east-2.amazonaws.com
#   docker tag cc3d-worker <acct>.dkr.ecr.us-east-2.amazonaws.com/cc3d-worker:latest
#   docker push <acct>.dkr.ecr.us-east-2.amazonaws.com/cc3d-worker:latest
# =============================================================================
FROM continuumio/miniconda3:24.9.2-0

LABEL org.opencontainers.image.title="CompuCell3D batch worker"
LABEL org.opencontainers.image.description="Headless CompuCell3D runner for BioSimulateAI"

# Non-interactive, and fail the build loudly rather than producing a broken image.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CONDA_ALWAYS_YES=true

# VTK needs these shared libraries present even for offscreen use. Installing them
# here rather than discovering the lack of them at run time.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libgl1 \
        libglu1-mesa \
        libxrender1 \
        libxext6 \
        libsm6 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The official channel. Both python AND compucell3d are pinned: the engine version is
# the one thing this image exists to carry, and leaving it unpinned under
# `channel_priority flexible` meant a rebuild could silently change the simulator
# under results that were already published. 4.6.0 is the version a real Batch run
# was verified against (its log line reads "cc3d 4.6.0").
# To move versions, change this deliberately and re-run the build's smoke test.
ARG CC3D_VERSION=4.6.0
RUN conda config --set channel_priority flexible \
    && conda install -n base -c compucell3d -c conda-forge \
        python=3.10 \
        "compucell3d=${CC3D_VERSION}" \
        boto3 \
    && conda clean --all --force-pkgs-dirs

# Prove the engine is importable AND that the numeric stack it needs actually loads.
# `import cc3d` alone passed on an image whose simulations could not run, so vtk is
# asserted here too -- it is the fragile piece under offscreen rendering.
RUN python -c "import cc3d, vtk, boto3; print('CompuCell3D', getattr(cc3d, '__version__', 'unknown'))"

WORKDIR /app
COPY cc3d_job_runner.py /app/cc3d_job_runner.py

# Offscreen rendering: there is no display on a Batch host.
ENV QT_QPA_PLATFORM=offscreen \
    LIBGL_ALWAYS_SOFTWARE=1 \
    MPLBACKEND=Agg

# Spot interruption contract, made explicit in the image.
#
# These jobs run on spot capacity: EC2 publishes a two-minute notice on the instance
# metadata service, then Docker sends SIGTERM to PID 1, then SIGKILL 30 seconds
# later. CompuCell3D writes no checkpoint, so a Batch retry restarts the simulation
# from step 0 rather than resuming it. The runner therefore CAPS RUN LENGTH instead
# of checkpointing, and on interruption uploads partial output to cells.partial.csv
# plus an honest status.json -- it never lets partial data reach cells.csv.
#
# The default cap matches REMOTE_RUN_LENGTH_CAP_MINUTES in cc3d_remote.py, so the
# validation the application performs before submitting and the ceiling the container
# enforces agree. Override per job through the Batch container environment.
ENV CC3D_MAX_MINUTES=90

# Explicit, though SIGTERM is Docker's default: the graceful-shutdown path is the
# whole point of this image's entrypoint, so the signal it depends on is not left
# implicit. The exec form below also matters -- python must be PID 1 to receive it,
# which a shell-form ENTRYPOINT would prevent.
STOPSIGNAL SIGTERM

ENTRYPOINT ["python", "/app/cc3d_job_runner.py"]
