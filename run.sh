#!/bin/bash

echo "__main__"

python3 ./code/POI_utils/prepare_data.py
python3 ./code/T5_for_POI_with_wx_input.py

echo "__end__"
