#!/bin/bash

python -u -m train \
--train.data_dir=/weka/proj-nmmo/runs/ \
"${@}"
