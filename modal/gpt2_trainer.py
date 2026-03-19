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

    marker = "/workspace/dev/data/fineweb10B/.dataset_ready"

    if not os.path.exists(marker):
        print("==> Downloading FineWeb 10B dataset...")

        subprocess.run(["chmod", "+x", "dev/data/fineweb.sh"], check=True)
        subprocess.run(["bash", "dev/data/fineweb.sh", "-10"], check=True)

        # Verify files are real before marking done
        import glob
        bins = glob.glob("/workspace/dev/data/fineweb10B/*.bin")
        real = [f for f in bins if os.path.getsize(f) > 1024]  # must be >1KB
        if not real:
            raise RuntimeError("Download failed — .bin files are empty or missing. Check the shell script output.")

        open(marker, "w").close()
        volume.commit()
    else:
        print("==> Dataset already cached.")

    # Step 2: Compile
    subprocess.run(["make", "train_gpt2cu"], check=True)

    # Step 3: Train
    subprocess.run(["./train_gpt2cu"], check=True)


@app.local_entrypoint()
def main():
    train.remote()