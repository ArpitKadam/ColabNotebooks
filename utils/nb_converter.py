import os
import sys
import nbformat

main_dir = os.getcwd()

def fix_notebook_metadata(notebook_rel_path):
    notebook_path = os.path.join(main_dir, notebook_rel_path)

    if not os.path.exists(notebook_path):
        return

    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        print(f"Skipping {notebook_rel_path}: {e}")
        return

    widgets = nb.metadata.get("widgets")

    if isinstance(widgets, dict) and "state" not in widgets:
        nb.metadata.pop("widgets")
        print(f"Fixed widgets metadata: {notebook_rel_path}")

        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

if __name__ == "__main__":
    for path in sys.argv[1:]:
        if path.endswith(".ipynb"):
            fix_notebook_metadata(path)
