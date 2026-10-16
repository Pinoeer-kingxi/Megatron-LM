import pytest
def test_pass():
    assert True
@pytest.mark.skip(reason="explicit CPU smoke skip")
def test_skip():
    pass
