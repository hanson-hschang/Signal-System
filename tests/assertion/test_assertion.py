from assertion import isNonNegativeInteger, isPositiveInteger, isPositiveNumber


def test_isPositiveNumber():
    assert isPositiveNumber(1) == True
    assert isPositiveNumber(0) == False
    assert isPositiveNumber(-1) == False
    assert isPositiveNumber(1.0) == True
    assert isPositiveNumber(0.0) == False
    assert isPositiveNumber(-1.0) == False
    assert isPositiveNumber(1.5) == True
    assert isPositiveNumber(0.5) == True
    assert isPositiveNumber(-1.5) == False
    assert isPositiveNumber("1") == False


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


def test_isNonNegativeInteger():
    assert isNonNegativeInteger(1) == True
    assert isNonNegativeInteger(0) == True
    assert isNonNegativeInteger(-1) == False
    assert isNonNegativeInteger(1.0) == True
    assert isNonNegativeInteger(0.0) == True
    assert isNonNegativeInteger(-1.0) == False
    assert isNonNegativeInteger(1.5) == False
    assert isNonNegativeInteger(0.5) == False
    assert isNonNegativeInteger(-1.5) == False
    assert isNonNegativeInteger("1") == False
