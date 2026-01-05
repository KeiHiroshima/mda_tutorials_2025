import nbformat
import re
import os

def convert_py_to_ipynb(py_path, ipynb_path):
    with open(py_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by # %%
    # Note: The first block might be empty or imports if no cell marker at start,
    # but our file starts with # %% [markdown]

    # Simple splitting
    parts = re.split(r'\n# %%\s*', content)

    cells = []

    # Handle the first part if it has content (e.g. before first marker)
    # in Jupytext format top metadata is usually preserved but we can skip if empty

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        # Check type
        if part.startswith("[markdown]"):
            cell_type = "markdown"
            source = part[len("[markdown]"):].strip()
            # Remove comment hashes from markdown source if they are line comments
            # Jupytext usually keeps semantic markdown in comments # # Header
            # Let's clean it up: remove leading `# ` from lines
            cleaned_source = []
            for line in source.splitlines():
                if line.startswith("# "):
                    cleaned_source.append(line[2:])
                elif line == "#":
                    cleaned_source.append("")
                else:
                    # If it doesn't start with #, it might be raw text or code that shouldn't be there
                    # In our file, all markdown is commented with #
                    cleaned_source.append(line)
            source = "\n".join(cleaned_source)
        else:
            cell_type = "code"
            source = part

        if cell_type == "markdown":
            cells.append(nbformat.v4.new_markdown_cell(source))
        else:
            cells.append(nbformat.v4.new_code_cell(source))

    nb = nbformat.v4.new_notebook(cells=cells)

    # Write
    with open(ipynb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Converted {py_path} to {ipynb_path}")

if __name__ == "__main__":
    convert_py_to_ipynb("tutorial_12.py", "week12_tutorial.ipynb")
