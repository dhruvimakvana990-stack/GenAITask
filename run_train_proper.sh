#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
cd /home/growlt167/GenAITask
stdbuf -oL -eL .venv/bin/python main.py train --experiment proper 2>&1 | grep --line-buffered -vaE "oneDNN|cpu_feature|TF-TRT|external/local_xla|InitializeLog|rebuild TensorFlow|following instructions"
