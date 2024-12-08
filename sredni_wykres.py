import re
import numpy as np
import matplotlib.pyplot as plt

def extract_avg_losses(logs):
    losses = re.findall(r"Epoch \d+: (\d+\.\d+)", logs)
    losses = [float(loss) for loss in losses]
    return losses


def extract_training_losses(logs):
    tab_losses = []
    for log in logs:
        losses = re.findall(r"Training Loss: (\d+\.\d+)", log)
        losses = [float(loss) for loss in losses]
        tab_losses.append(losses)
    return tab_losses



logs_avg = """
Average training loss per epoch across all runs:
Epoch 1: 0.9723
Epoch 2: 0.7561
Epoch 3: 0.6295
Epoch 4: 0.6209
Epoch 5: 0.6039
Epoch 6: 0.5893
Epoch 7: 0.5795
Epoch 8: 0.5634
Epoch 9: 0.5551
Epoch 10: 0.5489
Epoch 11: 0.5435
Epoch 12: 0.5385
Epoch 13: 0.5335
Epoch 14: 0.5281
Epoch 15: 0.5200
Epoch 16: 0.5158
Epoch 17: 0.5118
Epoch 18: 0.5078
Epoch 19: 0.5042
Epoch 20: 0.5009
"""

logs = """
Training run 1/7
2024-10-31 19:40:39.208812: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:274] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
Epoch 1/20 - Training Loss: 0.6027 - Training Accuracy: 4.90%
Epoch 1/20 - Test Loss: 0.6894 - Test Accuracy: 4.84%
Epoch 2/20 - Training Loss: 0.6349 - Training Accuracy: 4.90%
Epoch 2/20 - Test Loss: 0.6862 - Test Accuracy: 4.84%
Epoch 3/20 - Training Loss: 0.6211 - Training Accuracy: 4.90%
Epoch 3/20 - Test Loss: 0.6839 - Test Accuracy: 4.84%
Epoch 4/20 - Training Loss: 0.6063 - Training Accuracy: 4.90%
Epoch 4/20 - Test Loss: 0.6818 - Test Accuracy: 4.84%
Epoch 5/20 - Training Loss: 0.5905 - Training Accuracy: 4.90%
Epoch 5/20 - Test Loss: 0.6793 - Test Accuracy: 4.84%
Epoch 6/20 - Training Loss: 0.5775 - Training Accuracy: 4.90%
Epoch 6/20 - Test Loss: 0.6760 - Test Accuracy: 4.84%
Epoch 7/20 - Training Loss: 0.5478 - Training Accuracy: 4.90%
Epoch 7/20 - Test Loss: 0.6596 - Test Accuracy: 4.84%
Epoch 8/20 - Training Loss: 0.4745 - Training Accuracy: 4.90%
Epoch 8/20 - Test Loss: 0.6392 - Test Accuracy: 4.84%
Epoch 9/20 - Training Loss: 0.4521 - Training Accuracy: 4.90%
Epoch 9/20 - Test Loss: 0.6335 - Test Accuracy: 4.84%
Epoch 10/20 - Training Loss: 0.4389 - Training Accuracy: 4.90%
Epoch 10/20 - Test Loss: 0.6280 - Test Accuracy: 4.84%
Epoch 11/20 - Training Loss: 0.4240 - Training Accuracy: 4.90%
Epoch 11/20 - Test Loss: 0.6228 - Test Accuracy: 4.84%
Epoch 12/20 - Training Loss: 0.4088 - Training Accuracy: 4.90%
Epoch 12/20 - Test Loss: 0.6177 - Test Accuracy: 4.84%
Epoch 13/20 - Training Loss: 0.3946 - Training Accuracy: 4.90%
Epoch 13/20 - Test Loss: 0.6125 - Test Accuracy: 4.84%
Epoch 14/20 - Training Loss: 0.3803 - Training Accuracy: 4.90%
Epoch 14/20 - Test Loss: 0.6073 - Test Accuracy: 4.84%
Epoch 15/20 - Training Loss: 0.3667 - Training Accuracy: 4.90%
Epoch 15/20 - Test Loss: 0.6025 - Test Accuracy: 4.84%
Epoch 16/20 - Training Loss: 0.3543 - Training Accuracy: 4.90%
Epoch 16/20 - Test Loss: 0.5979 - Test Accuracy: 4.84%
Epoch 17/20 - Training Loss: 0.3428 - Training Accuracy: 4.90%
Epoch 17/20 - Test Loss: 0.5937 - Test Accuracy: 4.84%
Epoch 18/20 - Training Loss: 0.3322 - Training Accuracy: 4.90%
Epoch 18/20 - Test Loss: 0.5897 - Test Accuracy: 4.84%
Epoch 19/20 - Training Loss: 0.3226 - Training Accuracy: 4.90%
Epoch 19/20 - Test Loss: 0.5855 - Test Accuracy: 4.84%
Epoch 20/20 - Training Loss: 0.3137 - Training Accuracy: 4.90%
Epoch 20/20 - Test Loss: 0.5814 - Test Accuracy: 4.84%
Train accuracy: 65.25885558583107
Test accuracy: 63.44086021505376
Training run 2/7
Epoch 1/20 - Training Loss: 1.1641 - Training Accuracy: 4.90%
Epoch 1/20 - Test Loss: 0.7007 - Test Accuracy: 4.84%
Epoch 2/20 - Training Loss: 0.5544 - Training Accuracy: 4.90%
Epoch 2/20 - Test Loss: 0.6962 - Test Accuracy: 4.84%
Epoch 3/20 - Training Loss: 0.5790 - Training Accuracy: 4.90%
Epoch 3/20 - Test Loss: 0.6907 - Test Accuracy: 4.84%
Epoch 4/20 - Training Loss: 0.5632 - Training Accuracy: 4.90%
Epoch 4/20 - Test Loss: 0.6868 - Test Accuracy: 4.84%
Epoch 5/20 - Training Loss: 0.5413 - Training Accuracy: 4.90%
Epoch 5/20 - Test Loss: 0.6828 - Test Accuracy: 4.84%
Epoch 6/20 - Training Loss: 0.5214 - Training Accuracy: 4.90%
Epoch 6/20 - Test Loss: 0.6777 - Test Accuracy: 4.84%
Epoch 7/20 - Training Loss: 0.5084 - Training Accuracy: 4.90%
Epoch 7/20 - Test Loss: 0.6720 - Test Accuracy: 4.84%
Epoch 8/20 - Training Loss: 0.4974 - Training Accuracy: 4.90%
Epoch 8/20 - Test Loss: 0.6663 - Test Accuracy: 4.84%
Epoch 9/20 - Training Loss: 0.4858 - Training Accuracy: 4.90%
Epoch 9/20 - Test Loss: 0.6612 - Test Accuracy: 4.84%
Epoch 10/20 - Training Loss: 0.4765 - Training Accuracy: 4.90%
Epoch 10/20 - Test Loss: 0.6562 - Test Accuracy: 4.84%
Epoch 11/20 - Training Loss: 0.4696 - Training Accuracy: 4.90%
Epoch 11/20 - Test Loss: 0.6516 - Test Accuracy: 4.84%
Epoch 12/20 - Training Loss: 0.4637 - Training Accuracy: 4.90%
Epoch 12/20 - Test Loss: 0.6475 - Test Accuracy: 4.84%
Epoch 13/20 - Training Loss: 0.4581 - Training Accuracy: 4.90%
Epoch 13/20 - Test Loss: 0.6437 - Test Accuracy: 4.84%
Epoch 14/20 - Training Loss: 0.4528 - Training Accuracy: 4.90%
Epoch 14/20 - Test Loss: 0.6399 - Test Accuracy: 4.84%
Epoch 15/20 - Training Loss: 0.4479 - Training Accuracy: 4.90%
Epoch 15/20 - Test Loss: 0.6363 - Test Accuracy: 4.84%
Epoch 16/20 - Training Loss: 0.4433 - Training Accuracy: 4.90%
Epoch 16/20 - Test Loss: 0.6329 - Test Accuracy: 4.84%
Epoch 17/20 - Training Loss: 0.4388 - Training Accuracy: 4.90%
Epoch 17/20 - Test Loss: 0.6296 - Test Accuracy: 4.84%
Epoch 18/20 - Training Loss: 0.4341 - Training Accuracy: 4.90%
Epoch 18/20 - Test Loss: 0.6263 - Test Accuracy: 4.84%
Epoch 19/20 - Training Loss: 0.4296 - Training Accuracy: 4.90%
Epoch 19/20 - Test Loss: 0.6233 - Test Accuracy: 4.84%
Epoch 20/20 - Training Loss: 0.4258 - Training Accuracy: 4.90%
Epoch 20/20 - Test Loss: 0.6201 - Test Accuracy: 4.84%
Train accuracy: 67.71117166212534
Test accuracy: 65.59139784946237
Training run 3/7
Epoch 1/20 - Training Loss: 1.9009 - Training Accuracy: 4.90%
Epoch 1/20 - Test Loss: 1.8891 - Test Accuracy: 4.84%
Epoch 2/20 - Training Loss: 1.6920 - Training Accuracy: 4.90%
Epoch 2/20 - Test Loss: 1.1041 - Test Accuracy: 4.84%
Epoch 3/20 - Training Loss: 0.7113 - Training Accuracy: 4.90%
Epoch 3/20 - Test Loss: 0.7276 - Test Accuracy: 4.84%
Epoch 4/20 - Training Loss: 0.7059 - Training Accuracy: 4.90%
Epoch 4/20 - Test Loss: 0.7166 - Test Accuracy: 4.84%
Epoch 5/20 - Training Loss: 0.6735 - Training Accuracy: 4.90%
Epoch 5/20 - Test Loss: 0.7030 - Test Accuracy: 4.84%
Epoch 6/20 - Training Loss: 0.6293 - Training Accuracy: 4.90%
Epoch 6/20 - Test Loss: 0.6974 - Test Accuracy: 4.84%
Epoch 7/20 - Training Loss: 0.6205 - Training Accuracy: 4.90%
Epoch 7/20 - Test Loss: 0.6937 - Test Accuracy: 4.84%
Epoch 8/20 - Training Loss: 0.6132 - Training Accuracy: 4.90%
Epoch 8/20 - Test Loss: 0.6907 - Test Accuracy: 4.84%
Epoch 9/20 - Training Loss: 0.6075 - Training Accuracy: 4.90%
Epoch 9/20 - Test Loss: 0.6882 - Test Accuracy: 4.84%
Epoch 10/20 - Training Loss: 0.6029 - Training Accuracy: 4.90%
Epoch 10/20 - Test Loss: 0.6854 - Test Accuracy: 4.84%
Epoch 11/20 - Training Loss: 0.5989 - Training Accuracy: 4.90%
Epoch 11/20 - Test Loss: 0.6823 - Test Accuracy: 4.84%
Epoch 12/20 - Training Loss: 0.5954 - Training Accuracy: 4.90%
Epoch 12/20 - Test Loss: 0.6793 - Test Accuracy: 4.84%
Epoch 13/20 - Training Loss: 0.5924 - Training Accuracy: 4.90%
Epoch 13/20 - Test Loss: 0.6762 - Test Accuracy: 4.84%
Epoch 14/20 - Training Loss: 0.5896 - Training Accuracy: 4.90%
Epoch 14/20 - Test Loss: 0.6735 - Test Accuracy: 4.84%
Epoch 15/20 - Training Loss: 0.5870 - Training Accuracy: 4.90%
Epoch 15/20 - Test Loss: 0.6710 - Test Accuracy: 4.84%
Epoch 16/20 - Training Loss: 0.5846 - Training Accuracy: 4.90%
Epoch 16/20 - Test Loss: 0.6687 - Test Accuracy: 4.84%
Epoch 17/20 - Training Loss: 0.5822 - Training Accuracy: 4.90%
Epoch 17/20 - Test Loss: 0.6667 - Test Accuracy: 4.84%
Epoch 18/20 - Training Loss: 0.5800 - Training Accuracy: 4.90%
Epoch 18/20 - Test Loss: 0.6649 - Test Accuracy: 4.84%
Epoch 19/20 - Training Loss: 0.5780 - Training Accuracy: 4.90%
Epoch 19/20 - Test Loss: 0.6631 - Test Accuracy: 4.84%
Epoch 20/20 - Training Loss: 0.5760 - Training Accuracy: 4.90%
Epoch 20/20 - Test Loss: 0.6614 - Test Accuracy: 4.84%
Train accuracy: 65.25885558583107
Test accuracy: 63.44086021505376
Training run 4/7
Epoch 1/20 - Training Loss: 0.7406 - Training Accuracy: 4.90%
Epoch 1/20 - Test Loss: 0.7083 - Test Accuracy: 4.84%
Epoch 2/20 - Training Loss: 0.6832 - Training Accuracy: 4.90%
Epoch 2/20 - Test Loss: 0.7093 - Test Accuracy: 4.84%
Epoch 3/20 - Training Loss: 0.6806 - Training Accuracy: 4.90%
Epoch 3/20 - Test Loss: 0.7053 - Test Accuracy: 4.84%
Epoch 4/20 - Training Loss: 0.6738 - Training Accuracy: 4.90%
Epoch 4/20 - Test Loss: 0.7024 - Test Accuracy: 4.84%
Epoch 5/20 - Training Loss: 0.6697 - Training Accuracy: 4.90%
Epoch 5/20 - Test Loss: 0.6997 - Test Accuracy: 4.84%
Epoch 6/20 - Training Loss: 0.6659 - Training Accuracy: 4.90%
Epoch 6/20 - Test Loss: 0.6974 - Test Accuracy: 4.84%
Epoch 7/20 - Training Loss: 0.6623 - Training Accuracy: 4.90%
Epoch 7/20 - Test Loss: 0.6949 - Test Accuracy: 4.84%
Epoch 8/20 - Training Loss: 0.6590 - Training Accuracy: 4.90%
Epoch 8/20 - Test Loss: 0.6926 - Test Accuracy: 4.84%
Epoch 9/20 - Training Loss: 0.6562 - Training Accuracy: 4.90%
Epoch 9/20 - Test Loss: 0.6906 - Test Accuracy: 4.84%
Epoch 10/20 - Training Loss: 0.6540 - Training Accuracy: 4.90%
Epoch 10/20 - Test Loss: 0.6889 - Test Accuracy: 4.84%
Epoch 11/20 - Training Loss: 0.6523 - Training Accuracy: 4.90%
Epoch 11/20 - Test Loss: 0.6874 - Test Accuracy: 4.84%
Epoch 12/20 - Training Loss: 0.6509 - Training Accuracy: 4.90%
Epoch 12/20 - Test Loss: 0.6861 - Test Accuracy: 4.84%
Epoch 13/20 - Training Loss: 0.6495 - Training Accuracy: 4.90%
Epoch 13/20 - Test Loss: 0.6848 - Test Accuracy: 4.84%
Epoch 14/20 - Training Loss: 0.6479 - Training Accuracy: 4.90%
Epoch 14/20 - Test Loss: 0.6836 - Test Accuracy: 4.84%
Epoch 15/20 - Training Loss: 0.6460 - Training Accuracy: 4.90%
Epoch 15/20 - Test Loss: 0.6824 - Test Accuracy: 4.84%
Epoch 16/20 - Training Loss: 0.6438 - Training Accuracy: 4.90%
Epoch 16/20 - Test Loss: 0.6812 - Test Accuracy: 4.84%
Epoch 17/20 - Training Loss: 0.6414 - Training Accuracy: 4.90%
Epoch 17/20 - Test Loss: 0.6801 - Test Accuracy: 4.84%
Epoch 18/20 - Training Loss: 0.6384 - Training Accuracy: 4.90%
Epoch 18/20 - Test Loss: 0.6791 - Test Accuracy: 4.84%
Epoch 19/20 - Training Loss: 0.6351 - Training Accuracy: 4.90%
Epoch 19/20 - Test Loss: 0.6777 - Test Accuracy: 4.84%
Epoch 20/20 - Training Loss: 0.6319 - Training Accuracy: 4.90%
Epoch 20/20 - Test Loss: 0.6763 - Test Accuracy: 4.84%
Train accuracy: 65.39509536784742
Test accuracy: 62.903225806451616
Training run 5/7
Epoch 1/20 - Training Loss: 0.6322 - Training Accuracy: 4.90%
Epoch 1/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 2/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 2/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 3/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 3/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 4/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 4/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 5/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 5/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 6/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 6/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 7/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 7/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 8/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 8/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 9/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 9/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 10/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 10/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 11/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 11/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 12/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 12/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 13/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 13/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 14/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 14/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 15/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 15/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 16/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 16/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 17/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 17/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 18/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 18/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 19/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 19/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 20/20 - Training Loss: 0.7137 - Training Accuracy: 4.90%
Epoch 20/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Train accuracy: 64.57765667574932
Test accuracy: 63.97849462365591
Training run 6/7
Epoch 1/20 - Training Loss: 0.5353 - Training Accuracy: 4.90%
Epoch 1/20 - Test Loss: 0.6047 - Test Accuracy: 4.84%
Epoch 2/20 - Training Loss: 0.4264 - Training Accuracy: 4.90%
Epoch 2/20 - Test Loss: 0.6040 - Test Accuracy: 4.84%
Epoch 3/20 - Training Loss: 0.4195 - Training Accuracy: 4.90%
Epoch 3/20 - Test Loss: 0.5990 - Test Accuracy: 4.84%
Epoch 4/20 - Training Loss: 0.4025 - Training Accuracy: 4.90%
Epoch 4/20 - Test Loss: 0.5930 - Test Accuracy: 4.84%
Epoch 5/20 - Training Loss: 0.3875 - Training Accuracy: 4.90%
Epoch 5/20 - Test Loss: 0.5856 - Test Accuracy: 4.84%
Epoch 6/20 - Training Loss: 0.3748 - Training Accuracy: 4.90%
Epoch 6/20 - Test Loss: 0.5781 - Test Accuracy: 4.84%
Epoch 7/20 - Training Loss: 0.3633 - Training Accuracy: 4.90%
Epoch 7/20 - Test Loss: 0.5711 - Test Accuracy: 4.84%
Epoch 8/20 - Training Loss: 0.3528 - Training Accuracy: 4.90%
Epoch 8/20 - Test Loss: 0.5649 - Test Accuracy: 4.84%
Epoch 9/20 - Training Loss: 0.3431 - Training Accuracy: 4.90%
Epoch 9/20 - Test Loss: 0.5593 - Test Accuracy: 4.84%
Epoch 10/20 - Training Loss: 0.3340 - Training Accuracy: 4.90%
Epoch 10/20 - Test Loss: 0.5542 - Test Accuracy: 4.84%
Epoch 11/20 - Training Loss: 0.3256 - Training Accuracy: 4.90%
Epoch 11/20 - Test Loss: 0.5496 - Test Accuracy: 4.84%
Epoch 12/20 - Training Loss: 0.3179 - Training Accuracy: 4.90%
Epoch 12/20 - Test Loss: 0.5454 - Test Accuracy: 4.84%
Epoch 13/20 - Training Loss: 0.3107 - Training Accuracy: 4.90%
Epoch 13/20 - Test Loss: 0.5414 - Test Accuracy: 4.84%
Epoch 14/20 - Training Loss: 0.3042 - Training Accuracy: 4.90%
Epoch 14/20 - Test Loss: 0.5377 - Test Accuracy: 4.84%
Epoch 15/20 - Training Loss: 0.2981 - Training Accuracy: 4.90%
Epoch 15/20 - Test Loss: 0.5342 - Test Accuracy: 4.84%
Epoch 16/20 - Training Loss: 0.2924 - Training Accuracy: 4.90%
Epoch 16/20 - Test Loss: 0.5311 - Test Accuracy: 4.84%
Epoch 17/20 - Training Loss: 0.2872 - Training Accuracy: 4.90%
Epoch 17/20 - Test Loss: 0.5281 - Test Accuracy: 4.84%
Epoch 18/20 - Training Loss: 0.2821 - Training Accuracy: 4.90%
Epoch 18/20 - Test Loss: 0.5253 - Test Accuracy: 4.84%
Epoch 19/20 - Training Loss: 0.2772 - Training Accuracy: 4.90%
Epoch 19/20 - Test Loss: 0.5226 - Test Accuracy: 4.84%
Epoch 20/20 - Training Loss: 0.2726 - Training Accuracy: 4.90%
Epoch 20/20 - Test Loss: 0.5200 - Test Accuracy: 4.84%
Train accuracy: 64.98637602179836
Test accuracy: 66.12903225806451
Training run 7/7
Epoch 1/20 - Training Loss: 1.2305 - Training Accuracy: 4.90%
Epoch 1/20 - Test Loss: 0.7189 - Test Accuracy: 4.84%
Epoch 2/20 - Training Loss: 0.5880 - Training Accuracy: 4.90%
Epoch 2/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 3/20 - Training Loss: 0.6817 - Training Accuracy: 4.90%
Epoch 3/20 - Test Loss: 0.7257 - Test Accuracy: 4.84%
Epoch 4/20 - Training Loss: 0.6807 - Training Accuracy: 4.90%
Epoch 4/20 - Test Loss: 0.7232 - Test Accuracy: 4.84%
Epoch 5/20 - Training Loss: 0.6509 - Training Accuracy: 4.90%
Epoch 5/20 - Test Loss: 0.7223 - Test Accuracy: 4.84%
Epoch 6/20 - Training Loss: 0.6422 - Training Accuracy: 4.90%
Epoch 6/20 - Test Loss: 0.7199 - Test Accuracy: 4.84%
Epoch 7/20 - Training Loss: 0.6404 - Training Accuracy: 4.90%
Epoch 7/20 - Test Loss: 0.7165 - Test Accuracy: 4.84%
Epoch 8/20 - Training Loss: 0.6334 - Training Accuracy: 4.90%
Epoch 8/20 - Test Loss: 0.7136 - Test Accuracy: 4.84%
Epoch 9/20 - Training Loss: 0.6275 - Training Accuracy: 4.90%
Epoch 9/20 - Test Loss: 0.7118 - Test Accuracy: 4.84%
Epoch 10/20 - Training Loss: 0.6226 - Training Accuracy: 4.90%
Epoch 10/20 - Test Loss: 0.7104 - Test Accuracy: 4.84%
Epoch 11/20 - Training Loss: 0.6208 - Training Accuracy: 4.90%
Epoch 11/20 - Test Loss: 0.7093 - Test Accuracy: 4.84%
Epoch 12/20 - Training Loss: 0.6195 - Training Accuracy: 4.90%
Epoch 12/20 - Test Loss: 0.7078 - Test Accuracy: 4.84%
Epoch 13/20 - Training Loss: 0.6159 - Training Accuracy: 4.90%
Epoch 13/20 - Test Loss: 0.7060 - Test Accuracy: 4.84%
Epoch 14/20 - Training Loss: 0.6079 - Training Accuracy: 4.90%
Epoch 14/20 - Test Loss: 0.7046 - Test Accuracy: 4.84%
Epoch 15/20 - Training Loss: 0.5803 - Training Accuracy: 4.90%
Epoch 15/20 - Test Loss: 0.7012 - Test Accuracy: 4.84%
Epoch 16/20 - Training Loss: 0.5787 - Training Accuracy: 4.90%
Epoch 16/20 - Test Loss: 0.6965 - Test Accuracy: 4.84%
Epoch 17/20 - Training Loss: 0.5764 - Training Accuracy: 4.90%
Epoch 17/20 - Test Loss: 0.6917 - Test Accuracy: 4.84%
Epoch 18/20 - Training Loss: 0.5744 - Training Accuracy: 4.90%
Epoch 18/20 - Test Loss: 0.6864 - Test Accuracy: 4.84%
Epoch 19/20 - Training Loss: 0.5730 - Training Accuracy: 4.90%
Epoch 19/20 - Test Loss: 0.6805 - Test Accuracy: 4.84%
Epoch 20/20 - Training Loss: 0.5727 - Training Accuracy: 4.90%
Epoch 20/20 - Test Loss: 0.6753 - Test Accuracy: 4.84%
Train accuracy: 65.66757493188011
Test accuracy: 65.05376344086021
"""

runs = logs.split("Training run")
x = extract_training_losses(runs)[1:]
avg_losses = extract_avg_losses(logs_avg)

#rysowanie wykresu
plt.figure(figsize=(10, 6))

for i, epoch_data in enumerate(x):
    plt.plot(epoch_data, label=f'Loss run {i+1}', linestyle='--')
    
plt.plot(avg_losses, label='Average Loss', color='black', linewidth=3)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Losses Across Runs')
plt.legend()

plt.savefig("sredni_lr0,001_epochs20.png")
plt.show()

