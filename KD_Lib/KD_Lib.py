# -*- coding: utf-8 -*-
"""Main module template with example functions."""


def sum_numbers(number_list):
    """Example function. Sums a list of numbers using a for loop.

    Parameters
    ----------
    number_list : list
        List of ints or floats

    Returns
    -------
    int or float
        Sum of list

    Notes
    -----
    This is NOT good Python, just an example function for tests.
    """

    sum_val = 0
    for n in number_list:
        sum_val += n

    return sum_val


def max_number(number_list):
    """Example function. Finds max of list of numbers using a for loop.

    Parameters
    ----------
    number_list : list
        List of ints or floats

    Returns
    -------
    int or float
        Sum of list

    Notes
    -----
    Also not good Python.
    """

    cur_max = number_list[0]
    for n in number_list[1:]:
        if n > cur_max:
            cur_max = n

    return cur_max
