"""post-processes the docs after generating the with quartodoc"""

# %% imports
import os
import re

# Note
doc_source = "/Users/mpiekenbrock/geomcover/docs_src"
qmd_files = os.listdir(doc_source)
qmd_files = [fn for fn in qmd_files if fn[-3:] == "qmd"]

for fn in qmd_files:
	with open(os.path.join(doc_source, fn), "r+", encoding="utf-8") as file:
		content = file.read()

		## Remove header info + h1 title
		pattern = r"^---\n.*---\n"
		content = re.sub(pattern, "", content, flags=re.DOTALL)
		if content[0] == "#":
			content = content[(content.find("\n\n") + 2) :]

		## Tack on title + bread-crumbs
		prepend_str = f"---\ntitle: {fn[:-4]}\nbread-crumbs: true\n---\n"
		if "Parameters" in content:
			file.seek(0, 0)
			file.write(prepend_str + content)


# Example usage:
# suffix = ".txt"
# prepend_string = "prefix_"

# prepend_to_files(directory_path, suffix, prepend_string)
