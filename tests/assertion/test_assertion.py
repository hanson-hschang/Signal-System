from assertion import isPositiveInteger


def test_isPositiveInteger():
    assert isPositiveInteger(1) == True
    assert isPositiveInteger(0) == False
    assert isPositiveInteger(-1) == False
    assert isPositiveInteger(1.0) == True
    assert isPositiveInteger(0.0) == False
    assert isPositiveInteger(-1.0) == False
    assert isPositiveInteger(1.5) == False
    assert isPositiveInteger(0.5) == False
    assert isPositiveInteger(-1.5) == False
    assert isPositiveInteger("1") == False
