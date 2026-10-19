"""Copy BUSI into GET's layout: Dataset/busi/{images,masks}/<class>_<n>.png (names from the split pickle)."""
import os, sys, shutil, pickle

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/Dataset_BUSI_with_GT")
DST = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "..", "GET", "Dataset", "busi")
d = pickle.load(open(os.path.join(DST, "busi_train_test_names.pkl"), "rb"))
names = d["train"]["name_list"] + d["test"]["name_list"]
os.makedirs(f"{DST}/images", exist_ok=True); os.makedirs(f"{DST}/masks", exist_ok=True)
missing = []
for n in names:
    cls, idx = n.rsplit("_", 1)
    img, msk = f"{SRC}/{cls}/{cls} ({idx}).png", f"{SRC}/{cls}/{cls} ({idx})_mask.png"
    if os.path.exists(img) and os.path.exists(msk):
        shutil.copy(img, f"{DST}/images/{n}.png"); shutil.copy(msk, f"{DST}/masks/{n}.png")
    else:
        missing.append(n)
print(f"copied {len(names) - len(missing)}/{len(names)}; missing: {missing[:10]}")
