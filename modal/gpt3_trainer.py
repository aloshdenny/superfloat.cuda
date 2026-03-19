import modal

app = modal.App("train-gpt3-cuda")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "make", "gcc", "wget", "curl")
    .run_commands(
        "git clone --branch modal --single-branch https://github.com/aloshdenny/superfloat.cuda.git /workspace",
    )
    .workdir("/workspace")
)

volume = modal.Volume.from_name("fineweb100b-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 24,
    volumes={"/workspace/fineweb100B": volume},
    memory=131072
)
def train():
    import subprocess, os, sys

    os.chdir("/workspace")

    def run(cmd, **kwargs):
        """Run a command and stream output line-by-line to Modal logs."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout
            text=True,
            bufsize=1,                  # line-buffered
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            **kwargs
        )
        for line in process.stdout:
            print(line, end="", flush=True)
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

    marker = "/workspace/fineweb100B/.dataset_ready"

    if not os.path.exists(marker):
        print("==> Downloading FineWeb 100B dataset...", flush=True)
        run(["chmod", "+x", "dev/data/fineweb.sh"])
        run(["bash", "dev/data/fineweb.sh", "-100"])
        open(marker, "w").close()
        volume.commit()
    else:
        print("==> Dataset already cached.", flush=True)

    print("==> Compiling train_gpt3.cu...", flush=True)
    run(["make", "train_gpt3cu"])

    print("==> Starting GPT-3 training...", flush=True)
    run([
        "./train_gpt3cu",
        "-i", "/workspace/fineweb100B/fineweb_train_*.bin",
        "-j", "/workspace/fineweb100B/fineweb_val_*.bin",
        "-v", "1",
    ])


@app.local_entrypoint()
def main():
    train.remote()