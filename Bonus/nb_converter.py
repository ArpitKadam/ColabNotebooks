import os
import sys
import nbformat

main_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def fix_notebook_metadata(notebook_rel_path):
    notebook_path = os.path.join(main_dir, notebook_rel_path)

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

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
