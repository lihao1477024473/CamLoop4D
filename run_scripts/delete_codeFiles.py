from pathlib import Path
import shutil

for code_dir in Path("/root/autodl-tmp/som-change/lh").glob("*/code"):
    if code_dir.is_dir():
        print(code_dir)
        # continue
        for item in code_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"已清空: {code_dir}")


