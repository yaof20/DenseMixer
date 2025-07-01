python flops_compute.py

# outputs
# Model Training Cost Analysis Results --- Conventional Training for Qwen3-30B-A3B ---
# Number of parameters: 30,431,444,992
# Number of Forward TFLOPs per layer: 16.85
# Number of Backward TFLOPs per layer: 33.70
# Number of TFLOPs per layer: 50.54
# Peak memory cost: 157.93 GBs


# Model Training Cost Analysis Results: --- Dense Mixer Training for Qwen3-30B-A3B ---
# Number of parameters: 30,431,444,992
# Number of Forward TFLOPs per layer: 40.04
# Number of Backward TFLOPs per layer: 33.70
# Number of TFLOPs per layer: 73.74
# Peak memory cost: 164.96 GBs

# FLOPs: DenseMixer / Conventional = 1.46x
