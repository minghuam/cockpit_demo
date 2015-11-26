#!/bin/bash
python fcn32_obj_solve.py ../hand_training/model/HAND_iter_3000.caffemodel data.txt model/OBJ 4000 2 2>&1 | tee fcn32_obj_solve.log
