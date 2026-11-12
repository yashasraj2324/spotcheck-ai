import mlflow
import pytest
import torch
from mlflow.tracking import MlflowClient

from skin_lesion.training import checkpointing, tracking, trainer
from skin_lesion.training.trainer import run_training


def _steps(run_id: str, key: str) -> list[int]:
    return sorted(m.step for m in MlflowClient().get_metric_history(run_id, key))


def _crash_on_second_validation(monkeypatch):
    """Make the 2nd validation pass raise, simulating a crash after epoch 0 was checkpointed."""
    real, calls = trainer.evaluate, {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash")
        return real(*args, **kwargs)

    monkeypatch.setattr(trainer, "evaluate", flaky)
    return real


def test_resume_after_crash_continues_same_run_from_next_epoch(synthetic_cfg, monkeypatch):
    synthetic_cfg["train"]["epochs"] = 2
    real = _crash_on_second_validation(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_training(synthetic_cfg)

    path = checkpointing.checkpoint_path(synthetic_cfg)
    saved = checkpointing.load_epoch_checkpoint(path, torch.device("cpu"))
    assert saved["epoch"] == 0 and saved["completed"] is False
    assert _steps(saved["run_id"], "train_loss") == [0]

    monkeypatch.setattr(trainer, "evaluate", real)
    synthetic_cfg["train"]["resume"] = True  # the only config difference, and it is allowed
    result = run_training(synthetic_cfg)

    assert result["run_id"] == saved["run_id"]  # same MLflow run, not a new one
    assert _steps(result["run_id"], "train_loss") == [0, 1]  # epoch 0 not repeated
    experiment = mlflow.get_experiment_by_name(synthetic_cfg["mlflow"]["experiment"])
    assert len(MlflowClient().search_runs([experiment.experiment_id])) == 1
    assert checkpointing.load_epoch_checkpoint(path, torch.device("cpu"))["completed"] is True


def test_resume_refuses_changed_config(synthetic_cfg, monkeypatch):
    synthetic_cfg["train"]["epochs"] = 2
    _crash_on_second_validation(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash"):
        run_training(synthetic_cfg)

    synthetic_cfg["train"]["resume"] = True
    synthetic_cfg["train"]["head_lr"] = 0.5
    with pytest.raises(ValueError, match=r"train\.head_lr"):
        checkpointing.find_resume_state(synthetic_cfg, torch.device("cpu"))


def test_config_differences_ignores_only_resume_flag(synthetic_cfg):
    other = {**synthetic_cfg, "train": {**synthetic_cfg["train"], "resume": True}}
    assert checkpointing.config_differences(synthetic_cfg, other) == []
    other["model"] = {**other["model"], "dropout": 0.9}
    assert checkpointing.config_differences(synthetic_cfg, other) == ["model.dropout"]
    assert checkpointing.config_differences(None, synthetic_cfg)  # old checkpoint without config


def test_resume_without_checkpoint_returns_none(synthetic_cfg):
    synthetic_cfg["train"]["resume"] = True
    assert checkpointing.find_resume_state(synthetic_cfg, torch.device("cpu")) is None


def test_resume_disabled_ignores_existing_checkpoint(synthetic_cfg):
    path = checkpointing.checkpoint_path(synthetic_cfg)
    checkpointing.save_epoch_checkpoint(path, {"epoch": 5})
    assert checkpointing.find_resume_state(synthetic_cfg, torch.device("cpu")) is None


def test_save_is_atomic_and_roundtrips(tmp_path):
    path = tmp_path / "ck" / "last.pth"
    checkpointing.save_epoch_checkpoint(path, {"epoch": 3, "w": torch.ones(2)})
    assert not path.with_suffix(".tmp").exists()
    state = checkpointing.load_epoch_checkpoint(path, torch.device("cpu"))
    assert state["epoch"] == 3 and torch.equal(state["w"], torch.ones(2))


def test_upload_is_skipped_without_connection_string(synthetic_cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(
        checkpointing, "_container_client", lambda _: pytest.fail("must not connect to Azure")
    )
    assert checkpointing.upload_checkpoint(synthetic_cfg, tmp_path / "x.pth") is False


class _FakeContainer:
    def __init__(self):
        self.uploaded = {}

    def upload_blob(self, name, data, overwrite):
        self.uploaded[name] = (data.read(), overwrite)


def test_upload_targets_checkpoints_container(synthetic_cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true")
    fake, seen = _FakeContainer(), []
    monkeypatch.setattr(checkpointing, "_container_client", lambda c: (seen.append(c), fake)[1])
    f = tmp_path / "last.pth"
    f.write_bytes(b"abc")

    assert checkpointing.upload_checkpoint(synthetic_cfg, f) is True
    assert seen == ["checkpoints"]
    assert fake.uploaded == {f"{synthetic_cfg['run_name']}/last.pth": (b"abc", True)}


def test_upload_failure_does_not_raise(synthetic_cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true")

    def boom(_):
        raise ConnectionError("no network")

    monkeypatch.setattr(checkpointing, "_container_client", boom)
    f = tmp_path / "last.pth"
    f.write_bytes(b"abc")
    assert checkpointing.upload_checkpoint(synthetic_cfg, f) is False


def test_log_model_uses_name_on_mlflow3_and_artifact_path_on_mlflow2(monkeypatch):
    calls = []

    def v3(pytorch_model, name=None, input_example=None):
        calls.append(("v3", name, input_example.shape))

    def v2(pytorch_model, artifact_path, input_example=None):
        calls.append(("v2", artifact_path, input_example.shape))

    model = torch.nn.Linear(2, 2)
    for fake in (v3, v2):
        monkeypatch.setattr(tracking.mlflow.pytorch, "log_model", fake)
        tracking.log_pytorch_model(model, img_size=64)

    assert calls == [("v3", "model", (1, 3, 64, 64)), ("v2", "model", (1, 3, 64, 64))]
