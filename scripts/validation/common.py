"""Small shared helpers for repository validation; no production dependencies."""
import hashlib
import os
from pathlib import Path
import signal
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_process(argv, **kwargs):
    """Bound an entire CLI subprocess tree, retaining its log on timeout."""
    timeout = kwargs.pop("timeout", 1800)
    check = kwargs.pop("check", False)
    input_value = kwargs.pop("input", None)
    if kwargs.pop("capture_output", False):
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if input_value is not None:
        kwargs["stdin"] = subprocess.PIPE
    # Validation targets Linux locally and in CI. Each CLI's grandchildren inherit
    # this process group, so a timeout cannot leave optimization workers running.
    with subprocess.Popen(argv, start_new_session=True, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input_value, timeout=timeout)
        except BaseException:
            # Descendant CLIs may themselves create process groups. Kill the
            # entire tree before its parent disappears, including those groups.
            try:
                import psutil
                for child in psutil.Process(process.pid).children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            except Exception:
                pass
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        result = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result
