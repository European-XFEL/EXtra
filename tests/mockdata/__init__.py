
from extra_data import SourceData, KeyData


def assert_equal_sourcedata(sd1, sd2):
    assert isinstance(sd1, SourceData)
    assert isinstance(sd2, SourceData)
    assert sd1.source == sd2.source
    assert sd1.train_ids == sd2.train_ids


def assert_equal_keydata(kd1, kd2):
    assert isinstance(kd1, KeyData)
    assert isinstance(kd2, KeyData)
    assert kd1.source == kd2.source
    assert kd1.key == kd2.key
    assert kd1.train_ids == kd2.train_ids
