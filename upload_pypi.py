import subprocess
import sys


subprocess.call([sys.executable, "-m", "twine", "upload", "dist/smartprocesspool-*.tar.gz", "dist/smartprocesspool-*.whl", "--verbose"])