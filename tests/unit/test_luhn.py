from common.luhn import TEST_CARD_NUMBERS, complete_luhn, is_luhn_valid


def test_stripe_test_numbers_are_luhn_valid():
    for number in TEST_CARD_NUMBERS:
        assert is_luhn_valid(number), number


def test_tampered_digit_fails():
    tampered = "4242424242424241"
    assert not is_luhn_valid(tampered)


def test_non_digit_input_fails():
    assert not is_luhn_valid("4242-4242-4242-4242")


def test_complete_luhn_produces_valid_numbers():
    for partial in ("424242123456789", "555555000000000", "37828224631000"):
        number = complete_luhn(partial)
        assert number.startswith(partial) and len(number) == len(partial) + 1
        assert is_luhn_valid(number)
