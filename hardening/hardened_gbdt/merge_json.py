import json
import pandas as pd

if __name__ == "__main__":
    with open("/home/ypo/share/dp-gbdt-evaluation/results.txt") as file:
        for (i, line) in enumerate(file):
            if i >= 100:
                break
            print(line)