#!/bin/bash
python fcn32_obj_solve.py ../hand_training/model/HAND_iter_8000.caffemodel data.txt model/OBJ 4000 1 2>&1 | tee fcn32_obj_solve.log
