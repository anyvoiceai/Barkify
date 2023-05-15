from glob import glob

def Bestckpt(exp_root_dir):
    ckpts = glob(f"{exp_root_dir}/lightning_logs/*/checkpoints/*")
    ckpts = sorted(ckpts, key=lambda x:x.split("/")[-1].split("=")[1].split("-")[0])
    ckpts = ckpts[-1] if len(ckpts) > 0 else None
    return ckpts