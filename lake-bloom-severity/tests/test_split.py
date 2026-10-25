from model.data import load_frozen_split, load_processed_frame, load_train_test
from model.paths import load_config


def test_frozen_split_has_no_location_overlap():
    cfg = load_config()
    splits = load_frozen_split(cfg)
    train = set(splits.loc[splits["split"] == "train", "location_group"])
    test = set(splits.loc[splits["split"] == "test", "location_group"])
    assert train and test
    assert train.isdisjoint(test)


def test_every_processed_location_is_assigned():
    cfg = load_config()
    df = load_processed_frame(cfg)
    splits = load_frozen_split(cfg)
    data_groups = set(df[cfg["data"]["group_col"]].unique())
    split_groups = set(splits["location_group"].unique())
    assert data_groups == split_groups


def test_load_train_test_respects_frozen_file():
    data = load_train_test()
    assert set(data.groups_train).isdisjoint(set(data.groups_test))
    assert len(data.X_train) > len(data.X_test)
    assert list(data.X_train.columns) == data.feature_cols
