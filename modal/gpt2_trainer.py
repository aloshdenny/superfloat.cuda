import modal

app = modal.App("train-gpt2-cuda")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "make", "gcc", "wget", "curl")
    .run_commands(
        "git clone --branch modal --single-branch https://github.com/aloshdenny/superfloat.cuda.git /workspace",
    )
    .workdir("/workspace")
)

# Volume for 10B dataset
volume = modal.Volume.from_name("fineweb10b-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="B200",
    timeout=60 * 60 * 24,
    volumes={"/workspace/fineweb10B": volume},
)
def train():
    import subprocess, os

    os.chdir("/workspace")

    marker = "/workspace/fineweb10B/.dataset_ready"

    # Step 1: Download dataset
    if not os.path.exists(marker):
        print("==> Downloading FineWeb 10B dataset...")

        subprocess.run(["chmod", "+x", "fineweb.sh"], check=True)

        # Example: first 100 shards (adjust as needed)
        subprocess.run(
            ["bash", "dev/data/fineweb.sh", "-10"],
            check=True
        )

        open(marker, "w").close()
        volume.commit()
    else:
        print("==> Dataset already cached.")

    # Step 2: Compile
    print("==> Compiling train_gpt2.cu...")
    subprocess.run(["make", "train_gpt2cu"], check=True)

    # Step 3: Train
    print("==> Starting training...")
    subprocess.run(["./train_gpt2cu"], check=True)


@app.local_entrypoint()
def main():
    train.remote()