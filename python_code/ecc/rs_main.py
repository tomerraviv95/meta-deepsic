### code from site https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders

from python_code.ecc.polynomials_manipulation import init_tables, convert_binary_to_field, convert_field_to_binary
from python_code.ecc.rs_decoder import rs_calc_syndromes, rs_find_error_locator, rs_find_errors, rs_correct_errata
from python_code.ecc.rs_encoder import rs_encode_msg
import numpy as np


def encode(binary_word: np.ndarray, nsym: int):
    """
    Encodes binary word of length 1784 to a codeword of length 2040. All in numpy arrays.
    :param binary_word: length 1784 word
    :return: length 2040 codeword
    """
    init_tables()
    symbols_word = convert_binary_to_field(binary_word)
    symbols_codeword = rs_encode_msg(symbols_word, nsym=nsym)
    return convert_field_to_binary(symbols_codeword)


def decode(binary_rx: np.ndarray, nsym: int):
    """
    Decodes a given word with the Berlekamp-Massey decoder.
    :param binary_rx: the binary codewordword of length 2040
    :return: length 1784 word. If the algorithm detects more errors than can be repaired - returns 1784 first symbols of
    the binary codeword.
    """
    init_tables()
    symbols_rx = convert_binary_to_field(binary_rx.astype(int))
    synd = rs_calc_syndromes(symbols_rx, nsym=nsym)
    err_loc = rs_find_error_locator(synd, nsym=nsym)
    if err_loc is None:
        corrected_word = symbols_rx[:-nsym]
    else:
        pos = rs_find_errors(err_loc[::-1], len(symbols_rx))  # find the errors locations
        corrected_word = rs_correct_errata(symbols_rx, synd, pos)[:-nsym]
    return convert_field_to_binary(corrected_word)


if __name__ == "__main__":
    ## simple testing
    block_length = 480
    n_errors = 4
    nsym = 8
    words = np.random.randint(0, 2, [block_length])
    tx = encode(words, nsym)
    errors = np.zeros(block_length + nsym * 8).astype(int)
    errors_ind = np.random.randint(0, tx.shape, n_errors)
    errors[errors_ind] = 1
    print(f'generated errors at locations: {errors_ind // 8}')
    binary_rx = (tx + errors) % 2
    corrected_word = decode(binary_rx, nsym)
    flips_num = np.sum(np.abs(words - corrected_word))
    print("flips from original word after decoding: " + str(flips_num))
