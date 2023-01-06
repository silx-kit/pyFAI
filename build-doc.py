#!/usr/bin/env python3
import os
import sys
import logging
import bootstrap
from sphinx.cmd.build import main
logger = logging.getLogger(__name__)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_NAME = bootstrap.get_project_name(PROJECT_DIR)
logger.info("Project name: %s", PROJECT_NAME)
LIBPATH = bootstrap.build_project(PROJECT_NAME, PROJECT_DIR)
if __name__ == '__main__':
    sys.path.insert(0, LIBPATH)
    dest_dir = os.path.join(PROJECT_DIR, "build", "sphinx")
    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    argv = ["-b", "html",
            os.path.join(PROJECT_DIR,"doc","source"),
            dest_dir ]
    print(argv)
    sys.exit(main(argv))
