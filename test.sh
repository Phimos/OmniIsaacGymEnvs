#!/bin/zsh

/home/user/.local/share/ov/pkg/isaac_sim-2022.2.1/python.sh \
 omniisaacgymenvs/scripts/rlgames_train.py \
 checkpoint=runs/ShadowHand/nn/ShadowHand.pth \
 test=True