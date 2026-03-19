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

    # Step 1: Download dataset (skipped if already cached)
    marker = "/workspace/fineweb100B/.dataset_ready"
    if not os.path.exists(marker):
        print("==> Downloading FineWeb100B dataset...")

        # Make the shell script executable and run it
        # Passing a shard count arg is optional — omit for all 1028 shards
        subprocess.run(["chmod", "+x", "fineweb.sh"], check=True)
        subprocess.run(
            ["bash", "fineweb.sh", "100"],  # change or remove "100" to get more/all shards
            check=True
        )

        open(marker, "w").close()
        volume.commit()
    else:
        print("==> Dataset already cached, skipping download.")

    # Step 2: Compile
    print("==> Compiling train_gpt3.cu...")
    subprocess.run(["make", "train_gpt3cu"], check=True)

    # Step 3: Train
    print("==> Starting GPT-3 training...")
    subprocess.run(["./train_gpt3cu"], check=True)


@app.local_entrypoint()
def main():
    train.remote()