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
)
def train():
    import subprocess, os

    os.chdir("/workspace")

    marker = "/workspace/fineweb100B/.dataset_ready"

    # Step 1: Download dataset
    if not os.path.exists(marker):
        print("==> Downloading FineWeb 100B dataset...")

        subprocess.run(["chmod", "+x", "fineweb.sh"], check=True)

        # Example: 200 shards (or remove arg for full dataset)
        subprocess.run(
            ["bash", "dev/data/fineweb.sh", "-100"],
            check=True
        )

        open(marker, "w").close()
        volume.commit()
    else:
        print("==> Dataset already cached.")

    # Step 2: Compile
    print("==> Compiling train_gpt3.cu...")
    subprocess.run(["make", "train_gpt3cu"], check=True)

    # Step 3: Train
    print("==> Starting GPT-3 training...")
    subprocess.run(["./train_gpt3cu"], check=True)


@app.local_entrypoint()
def main():
    train.remote()