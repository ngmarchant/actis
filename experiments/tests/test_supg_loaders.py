"""Unit tests for automatic SUPG dataset downloading and loading."""

import io
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from experiments.data_prep import (
    SUPG_BASE_URL,
    download_supg_dataset,
    load_imagenet_dataset,
    load_jackson_dataset,
    load_onto_dataset,
    load_supg_dataset,
    load_tacred_dataset,
)
from experiments.scenarios import SUPGDataset, get_scenario


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_download_supg_dataset_caches_and_validates(tmp_path):
    csv_bytes = b"id,label,proxy_score\n0,False,0.1\n1,True,0.9\n"
    requested_urls: list[str] = []

    def fake_urlopen(req):
        requested_urls.append(req.full_url)
        return _FakeResponse(csv_bytes)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        path = download_supg_dataset("supg_onto", data_dir=tmp_path)
        assert path == tmp_path / "onto" / "source.csv"
        assert path.read_bytes() == csv_bytes
        assert requested_urls == [f"{SUPG_BASE_URL}/onto/source.csv"]

        # Second call uses cached file without hitting network
        cached_path = download_supg_dataset("onto", data_dir=tmp_path)
        assert cached_path == path
        assert len(requested_urls) == 1

        # force=True re-downloads
        download_supg_dataset("onto", data_dir=tmp_path, force=True)
        assert len(requested_urls) == 2

    with pytest.raises(ValueError, match="Unknown SUPG dataset"):
        download_supg_dataset("unknown_dataset", data_dir=tmp_path)


def test_download_supg_dataset_cleans_tmp_on_error(tmp_path):
    with patch("urllib.request.urlopen", side_effect=RuntimeError("network error")):
        with pytest.raises(RuntimeError, match="network error"):
            download_supg_dataset("onto", data_dir=tmp_path)

    assert not (tmp_path / "onto" / "source.csv").exists()
    assert not (tmp_path / "onto" / "source.csv.tmp").exists()


def test_load_supg_csv_and_feather_convenience_loaders(tmp_path):
    df_sample = pd.DataFrame(
        {"id": [0, 1], "label": [False, True], "proxy_score": [0.1, 0.9]}
    )
    csv_bytes = df_sample.to_csv(index=False).encode("utf-8")

    feather_buf = io.BytesIO()
    df_sample.to_feather(feather_buf)
    feather_bytes = feather_buf.getvalue()

    def fake_urlopen(req):
        if req.full_url.endswith(".feather"):
            return _FakeResponse(feather_bytes)
        return _FakeResponse(csv_bytes)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        df_onto = load_onto_dataset(data_dir=tmp_path)
        df_imagenet = load_imagenet_dataset(data_dir=tmp_path)
        df_tacred = load_tacred_dataset(data_dir=tmp_path)
        df_jackson = load_jackson_dataset(data_dir=tmp_path)
        df_generic = load_supg_dataset("jackson", data_dir=tmp_path)

    for df in (df_onto, df_imagenet, df_tacred, df_jackson, df_generic):
        assert list(df.columns) == ["id", "label", "proxy_score"]
        assert len(df) == 2


def test_supg_scenario_auto_downloads_when_missing(tmp_path):
    csv_bytes = b"id,label,proxy_score\n0,False,0.2\n1,True,0.8\n"
    target = Path(tmp_path) / "onto" / "source.csv"

    def fake_urlopen(req):
        return _FakeResponse(csv_bytes)

    scenario = get_scenario("onto", data_path=target)
    assert isinstance(scenario, SUPGDataset)

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        pop = scenario.generate_population()

    assert target.exists()
    assert len(pop) == 2
    assert pop.scores.tolist() == pytest.approx([0.2, 0.8])
    assert pop.labels.tolist() == [False, True]
