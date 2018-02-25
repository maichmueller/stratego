import itertools
import math
from scipy.signal import convolve2d, deconvolve

import numpy as np
import copy
import matplotlib.pyplot as plt

def smart_sub_shit(mask, div_length):
    max_val = max(mask)
    divisor = []
    quotient = []
    subdivisor = []
    found_divisor = False
    for submask in itertools.product(range(-max_val, max_val)[::-1], repeat=div_length):
        if not submask[0]:
            continue
        quotient, remainder = deconvolve(mask, submask)
        if sum(abs(remainder)) == 0:
            divisor = submask
            found_divisor = True
            break
    if not found_divisor:
        if div_length < len(mask)-1:
            smart_sub_shit(mask, div_length+1)
        else:
            print("Divisor {} can't be split anymore.".format(mask))
    else:
        if len(quotient) > 2:
            subdivisor = smart_sub_shit(quotient.astype("int"), 2)
    return {"Quotient": quotient, "Divisor": list(divisor), "Subdivisor": subdivisor}


def deconvolve_mask(mask):
    result = smart_sub_shit(mask, 2)
    quotient = result["Quotient"]
    divisor = result["Divisor"]
    subdivisor = result["Subdivisor"]
    if subdivisor:
        def get_depth_subdivisor(subdiv, depth=0):
            if not list(subdiv["Quotient"]):
                return depth
            else:
                return get_depth_subdivisor(subdiv["Subdivisor"], depth+1)
        subdiv_depth = get_depth_subdivisor(subdivisor)
        subdivisor_array = []
        current_subdiv = result
        last_quotient = None
        for i in range(subdiv_depth):
            subd = current_subdiv["Subdivisor"]
            subdivisor_array.append(subd["Divisor"])
            last_quotient = subd["Quotient"]
            current_subdiv = subd
        return [list(divisor)] + [list(subdiv) for subdiv in subdivisor_array] + [list(last_quotient)]
    else:
        return result


print(deconvolve_mask([1,2,3,4,5,7,1,4,3]))