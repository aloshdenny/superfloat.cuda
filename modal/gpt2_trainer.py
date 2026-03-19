import modal

app = modal.App("train-gpt2-cuda")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "make", "gcc", "wget", "curl")
    .pip_install("huggingface_hub")
    .run_commands(
        "git clone --branch modal --single-branch https://github.com/aloshdenny/superfloat.cuda.git /workspace",
    )
    .workdir("/workspace")
)

# Volume for 10B dataset
volume = modal.Volume.from_name("fineweb10b-data", create_if_missing=True)

@app.function(
    image=(
        modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
        .apt_install("git", "make", "gcc", "wget", "curl")
        .pip_install("huggingface_hub")
        .run_commands(
            "git clone --branch modal --single-branch https://github.com/aloshdenny/superfloat.cuda.git /workspace",
        )
        .workdir("/workspace")
    ),
    gpu="B200",
    timeout=60 * 60 * 24,
    volumes={"/workspace/fineweb10B": volume},
    memory=131072
)
def train():
    import subprocess, os
    from huggingface_hub import hf_hub_download, list_repo_files

    os.chdir("/workspace")
    marker = "/workspace/fineweb10B/.dataset_ready"

    if not os.path.exists(marker):
        print("==> Downloading FineWeb 10B dataset via huggingface_hub...", flush=True)

        save_dir = "/workspace/fineweb10B"
        os.makedirs(save_dir, exist_ok=True)

        repo_id = "chrisdryden/FineWebTokenizedGPT2"

        # List all .bin files in the repo
        all_files = [
            f for f in list_repo_files(repo_id, repo_type="dataset")
            if f.endswith(".bin")
        ]
        print(f"Found {len(all_files)} .bin files in repo", flush=True)

        for i, filename in enumerate(sorted(all_files)):
            dest = os.path.join(save_dir, os.path.basename(filename))
            if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
                print(f"[{i+1}/{len(all_files)}] Skipping {filename} (already exists)", flush=True)
                continue
            print(f"[{i+1}/{len(all_files)}] Downloading {filename}...", flush=True)
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=save_dir,
                local_dir_use_symlinks=False,
            )

        open(marker, "w").close()
        volume.commit()
    else:
        print("==> Dataset already cached.", flush=True)

    # Step 2: Compile
    print("==> Compiling train_gpt2.cu...", flush=True)
    subprocess.run(["make", "train_gpt2cu"], check=True)

    # Step 3: Train
    print("==> Starting GPT-2 training...", flush=True)
    subprocess.run([
        "./train_gpt2cu",
        "-i", "/workspace/fineweb10B/fineweb_train_*.bin",
        "-j", "/workspace/fineweb10B/fineweb_val_*.bin",
    ], check=True)