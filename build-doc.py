#!/usr/bin/env python3
import os
import sys
import logging
from sphinx.cmd.build import main
from bootstrap import PROJECT_DIR, PROJECT_NAME, build_project

logger = logging.getLogger(__name__)
logger.info("Project name: %s", PROJECT_NAME)
LIBPATH = build_project(PROJECT_NAME, PROJECT_DIR)

if __name__ == '__main__':
    sys.path.insert(0, LIBPATH)
    dirname = LIBPATH
    while not os.path.split(dirname)[-1].startswith("build"):
        dirname = os.path.split(dirname)[0]
    print(dirname)
    dest_dir = os.path.join(dirname, "sphinx")
    os.makedirs(dest_dir, exist_ok=True)

    argv = ["-b", "html",
            os.path.join(PROJECT_DIR,"doc","source"),
            dest_dir ]
    print("sphinx " + " ".join(argv))
    sys.exit(main(argv))
