"""post-processes the docs after generating the with quartodoc"""

import os
import re

doc_source = "/Users/mpiekenbrock/geomcover/docs_src"
qmd_files = os.listdir(doc_source)
qmd_files = [fn for fn in qmd_files if fn[-3:] == "qmd"]

for fn in qmd_files:
	with open(os.path.join(doc_source, fn), "r+", encoding="utf-8") as file:
		content = file.read()
		prepend_str = f"---\ntitle: {fn[:-4]}\nbread-crumbs: true\n---\n"
		ind = [m.start() for m in re.finditer(r"---\n", content)]
		s, e = ind if len(ind) == 2 else -1, -4
		if "Parameters" in content and prepend_str not in content:
			file.seek(0, 0)
			file.write(prepend_str + content[(e + 4) :])

title: cover.set_cover_ilp

# Example usage:
# suffix = ".txt"
# prepend_string = "prefix_"

# prepend_to_files(directory_path, suffix, prepend_string)
