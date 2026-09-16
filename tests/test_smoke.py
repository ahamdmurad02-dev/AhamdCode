from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_app(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "app.py"), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_status_is_truthful_without_checkpoint():
    result = run_app("status")
    assert result.returncode == 0
    assert "Model: Not loaded" in result.stdout
    assert "Checkpoint: none" in result.stdout


def test_help_command_works():
    result = run_app("help")
    assert result.returncode == 0
    assert "status" in result.stdout
    assert "model" in result.stdout


def test_model_command_reports_missing_checkpoint():
    result = run_app("model")
    assert result.returncode == 1
    assert "Model: Not loaded" in result.stdout
    assert "No checkpoint found." in result.stdout


def test_model_config_is_decoder_only_contract():
    torch = __import__("torch")
    from ahamdcode.model.config import ModelConfig
    from ahamdcode.model.transformer import MiniTransformer

    cfg = ModelConfig.mini_fast()
    model = MiniTransformer(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    output = model(tokens)
    assert output["logits"].shape == (1, 8, cfg.vocab_size)
    assert model.count_parameters() > 0
