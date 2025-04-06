__all__ = ["clamp", "tabulate"]

#Clamps a value within the specified lower and upper bounds.
def clamp(value: int, lower: int, upper: int):
    if upper < lower:
        raise ValueError(f"Cannot clamp into <{lower}, {upper}>")

    return max(lower, min(value, upper))

#Prints a formatted table of key-value pairs from a dictionary, with a header.
def tabulate(kv: dict, header: str):
    prob_dict = {k: repr(v) for k, v in kv.items()}
    
    # Find the maximum lengths of the keys, values, and header
    maxkey = max(map(len, prob_dict))
    maxval = max(map(len, prob_dict.values()))
    maxlen = max(len(header), maxkey + maxval + 1)
    
    # Print the table with headers and formatted key-value pairs
    print("+-" + "-" * maxlen + "-+")
    print("| " + header.ljust(maxlen) + " |")
    print("+-" + "-" * maxlen + "-+")

    for k, v in prob_dict.items():
        print("| " + f"{k.ljust(maxkey)} {v.rjust(maxval)}" + " |")

    print("+-" + "-" * maxlen + "-+")
    print()
