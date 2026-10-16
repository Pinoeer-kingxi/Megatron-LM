import pytest
@pytest.fixture
def resource():
    yield
    assert False, "intentional teardown failure"
def test_teardown(resource):
    assert True
