import json

import pytest

from src import data

pytestmark = pytest.mark.skipif(
    not (data.SPLITS_PATH.exists() and (data.DATA / "spider" / "train_spider.json").exists()),
    reason="splits.json or Spider data not present",
)


@pytest.fixture(scope="module")
def splits():
    with open(data.SPLITS_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def by_id():
    return {e["id"]: e for e in data.spider_train_all()}


def test_no_database_overlap(splits, by_id):
    train_dbs = {by_id[i]["db_id"] for i in splits["train_ids"]}
    val_dbs = {by_id[i]["db_id"] for i in splits["val_ids"]}
    assert not (train_dbs & val_dbs)
    assert val_dbs <= set(splits["val_databases"])
    assert train_dbs <= set(splits["train_databases"])
    assert not (set(splits["val_databases"]) & set(splits["train_databases"]))


def test_ids_are_unique_and_disjoint(splits):
    tr, va = splits["train_ids"], splits["val_ids"]
    assert len(tr) == len(set(tr)) and len(va) == len(set(va))
    assert not (set(tr) & set(va))


def test_every_question_is_accounted_for(splits, by_id):
    d = splits["dropped"]
    n = (len(splits["train_ids"]) + len(splits["val_ids"]) + len(d["train_gold_empty"])
         + len(d["train_gold_error"]) + len(d["val_gold_error"]))
    assert n == len(by_id) == 7000
    assert splits["counts"]["train"] == len(splits["train_ids"])
    assert splits["counts"]["val"] == len(splits["val_ids"])


def test_val_size_and_spread(splits):
    assert 450 <= len(splits["val_ids"]) <= 600
    assert len(splits["val_databases"]) >= 10


def test_load_split_matches_file(splits):
    assert [e["id"] for e in data.load_split("val")] == splits["val_ids"]
    assert len(data.load_split("train")) == len(splits["train_ids"])


@pytest.mark.parametrize("split", ["test_a", "test_b"])
def test_test_splits_are_guarded(split):
    with pytest.raises(PermissionError):
        data.load_split(split)
