import os
import subprocess
import sys

PROBE = (
    "import sys\n"
    "from larry.common.utils import reexec_with_hash_seed\n"
    "reexec_with_hash_seed(int(sys.argv[1]))\n"
    "print(hash('larry'))\n"
)


def _hash_in_fresh_process(seed: int) -> str:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(seed)], env=env, capture_output=True, text=True, check=True
    )
    return result.stdout.strip().splitlines()[-1]


def test_the_hash_seed_takes_effect_and_is_stable_across_processes() -> None:
    assert _hash_in_fresh_process(42) == _hash_in_fresh_process(42)
    assert _hash_in_fresh_process(42) != _hash_in_fresh_process(43)
