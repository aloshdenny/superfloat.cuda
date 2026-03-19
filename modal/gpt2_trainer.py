import modal

app = modal.App("train-gpt2-cuda")

# Build the image once — dataset download is separate (see below)
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "make", "gcc", "wget", "curl")
    .pip_install("torch", "numpy", "tiktoken", "requests", "tqdm", "transformers", "datasets")  # fineweb.py deps
    .run_commands(
        "git clone https://github.com/aloshdenny/superfloat.cuda.git /workspace",
    )
    .workdir("/workspace")
)

# Use a persistent volume so the dataset survives across runs
volume = modal.Volume.from_name("fineweb-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 24,          # 6 hours
    volumes={"/workspace/data": volume},
)
def train():
    import subprocess, os

    os.chdir("/workspace")

    # Step 1: Download dataset (skips if already cached in the volume)
    marker = "/workspace/data/.dataset_ready"
    if not os.path.exists(marker):
        print("==> Downloading FineWeb dataset...")
        subprocess.run(
            ["python", "dev/data/fineweb.py"],
            check=True
        )
        open(marker, "w").close()
        volume.commit()  # persist to volume
    else:
        print("==> Dataset already cached, skipping download.")

    # Step 2: Compile
    print("==> Compiling train_gpt2.cu...")
    subprocess.run(["make", "train_gpt2cu"], check=True)

    # Step 3: Train
    print("==> Starting training...")
    subprocess.run(["./train_gpt2cu"], check=True)


@app.local_entrypoint()
def main():
    train.remote()