import matplotlib.pyplot as plt
import re
import os

def parse_text(input_text):
    pattern = r' loss:\s+([\d\.]+)'
    loss_values = []
    
    matches = re.findall(pattern, input_text)
    
    for match in matches:
        loss_values.append(float(match)) 
    
    return loss_values

def plot_and_save_loss(loss_values, plot_name, title):
    epochs = list(range(1, len(loss_values) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_values, marker='o', linestyle='-', color='b', label="Loss")
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.xticks(epochs)
    plt.legend()

    save_dir = "wykresy/loss"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(f"{save_dir}/{plot_name}.png")
    plt.show()


input_text = """
Epoch 1/100
587/587 [==============================] - 7s 12ms/step - loss: 2.0039 - accuracy: 0.1721 - val_loss: 1.8362 - val_accuracy: 0.4082
Epoch 2/100
587/587 [==============================] - 7s 12ms/step - loss: 1.9499 - accuracy: 0.3799 - val_loss: 1.7575 - val_accuracy: 0.7891
Epoch 3/100
587/587 [==============================] - 8s 13ms/step - loss: 1.8995 - accuracy: 0.5417 - val_loss: 1.6827 - val_accuracy: 0.8776
Epoch 4/100
587/587 [==============================] - 8s 13ms/step - loss: 1.8529 - accuracy: 0.5775 - val_loss: 1.6120 - val_accuracy: 0.9048
Epoch 5/100
587/587 [==============================] - 7s 12ms/step - loss: 1.8100 - accuracy: 0.5809 - val_loss: 1.5464 - val_accuracy: 0.9048
Epoch 6/100
587/587 [==============================] - 7s 12ms/step - loss: 1.7704 - accuracy: 0.5809 - val_loss: 1.4851 - val_accuracy: 0.9048
Epoch 7/100
587/587 [==============================] - 7s 12ms/step - loss: 1.7342 - accuracy: 0.5809 - val_loss: 1.4275 - val_accuracy: 0.9048
Epoch 8/100
587/587 [==============================] - 7s 12ms/step - loss: 1.7013 - accuracy: 0.5809 - val_loss: 1.3752 - val_accuracy: 0.9048
Epoch 9/100
587/587 [==============================] - 6s 11ms/step - loss: 1.6716 - accuracy: 0.5809 - val_loss: 1.3273 - val_accuracy: 0.9048
Epoch 10/100
587/587 [==============================] - 7s 11ms/step - loss: 1.6449 - accuracy: 0.5809 - val_loss: 1.2817 - val_accuracy: 0.9048
Epoch 11/100
587/587 [==============================] - 7s 11ms/step - loss: 1.6210 - accuracy: 0.5809 - val_loss: 1.2411 - val_accuracy: 0.9048
Epoch 12/100
587/587 [==============================] - 7s 11ms/step - loss: 1.5997 - accuracy: 0.5809 - val_loss: 1.2044 - val_accuracy: 0.9048
Epoch 13/100
587/587 [==============================] - 6s 11ms/step - loss: 1.5808 - accuracy: 0.5809 - val_loss: 1.1701 - val_accuracy: 0.9048
Epoch 14/100
587/587 [==============================] - 7s 11ms/step - loss: 1.5640 - accuracy: 0.5809 - val_loss: 1.1398 - val_accuracy: 0.9048
Epoch 15/100
587/587 [==============================] - 7s 13ms/step - loss: 1.5491 - accuracy: 0.5809 - val_loss: 1.1116 - val_accuracy: 0.9048
Epoch 16/100
587/587 [==============================] - 7s 12ms/step - loss: 1.5359 - accuracy: 0.5809 - val_loss: 1.0874 - val_accuracy: 0.9048
Epoch 17/100
587/587 [==============================] - 7s 12ms/step - loss: 1.5243 - accuracy: 0.5809 - val_loss: 1.0645 - val_accuracy: 0.9048
Epoch 18/100
587/587 [==============================] - 7s 12ms/step - loss: 1.5139 - accuracy: 0.5809 - val_loss: 1.0445 - val_accuracy: 0.9048
Epoch 19/100
587/587 [==============================] - 7s 12ms/step - loss: 1.5048 - accuracy: 0.5809 - val_loss: 1.0262 - val_accuracy: 0.9048
Epoch 20/100
587/587 [==============================] - 7s 11ms/step - loss: 1.4966 - accuracy: 0.5809 - val_loss: 1.0098 - val_accuracy: 0.9048
Epoch 21/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4894 - accuracy: 0.5809 - val_loss: 0.9950 - val_accuracy: 0.9048
Epoch 22/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4829 - accuracy: 0.5809 - val_loss: 0.9821 - val_accuracy: 0.9048
Epoch 23/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4771 - accuracy: 0.5809 - val_loss: 0.9702 - val_accuracy: 0.9048
Epoch 24/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4719 - accuracy: 0.5809 - val_loss: 0.9599 - val_accuracy: 0.9048
Epoch 25/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4672 - accuracy: 0.5809 - val_loss: 0.9493 - val_accuracy: 0.9048
Epoch 26/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4629 - accuracy: 0.5809 - val_loss: 0.9414 - val_accuracy: 0.9048
Epoch 27/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4590 - accuracy: 0.5809 - val_loss: 0.9337 - val_accuracy: 0.9048
Epoch 28/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4554 - accuracy: 0.5809 - val_loss: 0.9270 - val_accuracy: 0.9048
Epoch 29/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4521 - accuracy: 0.5809 - val_loss: 0.9209 - val_accuracy: 0.9048
Epoch 30/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4491 - accuracy: 0.5809 - val_loss: 0.9151 - val_accuracy: 0.9048
Epoch 31/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4462 - accuracy: 0.5809 - val_loss: 0.9100 - val_accuracy: 0.9048
Epoch 32/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4435 - accuracy: 0.5809 - val_loss: 0.9059 - val_accuracy: 0.9048
Epoch 33/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4410 - accuracy: 0.5809 - val_loss: 0.9018 - val_accuracy: 0.9048
Epoch 34/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4387 - accuracy: 0.5809 - val_loss: 0.8985 - val_accuracy: 0.9048
Epoch 35/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4364 - accuracy: 0.5809 - val_loss: 0.8953 - val_accuracy: 0.9048
Epoch 36/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4343 - accuracy: 0.5809 - val_loss: 0.8929 - val_accuracy: 0.9048
Epoch 37/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4322 - accuracy: 0.5809 - val_loss: 0.8902 - val_accuracy: 0.9048
Epoch 38/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4303 - accuracy: 0.5809 - val_loss: 0.8878 - val_accuracy: 0.9048
Epoch 39/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4284 - accuracy: 0.5809 - val_loss: 0.8863 - val_accuracy: 0.9048
Epoch 40/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4266 - accuracy: 0.5809 - val_loss: 0.8848 - val_accuracy: 0.9048
Epoch 41/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4249 - accuracy: 0.5809 - val_loss: 0.8831 - val_accuracy: 0.9048
Epoch 42/100
587/587 [==============================] - 8s 13ms/step - loss: 1.4232 - accuracy: 0.5809 - val_loss: 0.8821 - val_accuracy: 0.9048
Epoch 43/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4216 - accuracy: 0.5809 - val_loss: 0.8809 - val_accuracy: 0.9048
Epoch 44/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4200 - accuracy: 0.5809 - val_loss: 0.8800 - val_accuracy: 0.9048
Epoch 45/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4185 - accuracy: 0.5809 - val_loss: 0.8793 - val_accuracy: 0.9048
Epoch 46/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4170 - accuracy: 0.5809 - val_loss: 0.8788 - val_accuracy: 0.9048
Epoch 47/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4156 - accuracy: 0.5809 - val_loss: 0.8784 - val_accuracy: 0.9048
Epoch 48/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4142 - accuracy: 0.5809 - val_loss: 0.8778 - val_accuracy: 0.9048
Epoch 49/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4128 - accuracy: 0.5809 - val_loss: 0.8775 - val_accuracy: 0.9048
Epoch 50/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4115 - accuracy: 0.5809 - val_loss: 0.8774 - val_accuracy: 0.9048
Epoch 51/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4102 - accuracy: 0.5809 - val_loss: 0.8774 - val_accuracy: 0.9048
Epoch 52/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4089 - accuracy: 0.5809 - val_loss: 0.8773 - val_accuracy: 0.9048
Epoch 53/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4076 - accuracy: 0.5809 - val_loss: 0.8774 - val_accuracy: 0.9048
Epoch 54/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4064 - accuracy: 0.5809 - val_loss: 0.8774 - val_accuracy: 0.9048
Epoch 55/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4052 - accuracy: 0.5809 - val_loss: 0.8776 - val_accuracy: 0.9048
Epoch 56/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4040 - accuracy: 0.5809 - val_loss: 0.8782 - val_accuracy: 0.9048
Epoch 57/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4028 - accuracy: 0.5809 - val_loss: 0.8784 - val_accuracy: 0.9048
Epoch 58/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4017 - accuracy: 0.5809 - val_loss: 0.8786 - val_accuracy: 0.9048
Epoch 59/100
587/587 [==============================] - 7s 12ms/step - loss: 1.4006 - accuracy: 0.5809 - val_loss: 0.8791 - val_accuracy: 0.9048
Epoch 60/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3995 - accuracy: 0.5809 - val_loss: 0.8795 - val_accuracy: 0.9048
Epoch 61/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3984 - accuracy: 0.5809 - val_loss: 0.8798 - val_accuracy: 0.9048
Epoch 62/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3974 - accuracy: 0.5809 - val_loss: 0.8804 - val_accuracy: 0.9048
Epoch 63/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3963 - accuracy: 0.5809 - val_loss: 0.8804 - val_accuracy: 0.9048
Epoch 64/100
587/587 [==============================] - 7s 11ms/step - loss: 1.3953 - accuracy: 0.5809 - val_loss: 0.8810 - val_accuracy: 0.9048
Epoch 65/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3943 - accuracy: 0.5809 - val_loss: 0.8817 - val_accuracy: 0.9048
Epoch 66/100
587/587 [==============================] - 7s 11ms/step - loss: 1.3932 - accuracy: 0.5809 - val_loss: 0.8822 - val_accuracy: 0.9048
Epoch 67/100
587/587 [==============================] - 7s 11ms/step - loss: 1.3923 - accuracy: 0.5809 - val_loss: 0.8826 - val_accuracy: 0.9048
Epoch 68/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3913 - accuracy: 0.5809 - val_loss: 0.8838 - val_accuracy: 0.9048
Epoch 69/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3903 - accuracy: 0.5809 - val_loss: 0.8843 - val_accuracy: 0.9048
Epoch 70/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3894 - accuracy: 0.5809 - val_loss: 0.8849 - val_accuracy: 0.9048
Epoch 71/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3884 - accuracy: 0.5809 - val_loss: 0.8852 - val_accuracy: 0.9048
Epoch 72/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3875 - accuracy: 0.5809 - val_loss: 0.8862 - val_accuracy: 0.9048
Epoch 73/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3865 - accuracy: 0.5809 - val_loss: 0.8862 - val_accuracy: 0.9048
Epoch 74/100
587/587 [==============================] - 7s 11ms/step - loss: 1.3856 - accuracy: 0.5809 - val_loss: 0.8869 - val_accuracy: 0.9048
Epoch 75/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3847 - accuracy: 0.5809 - val_loss: 0.8878 - val_accuracy: 0.9048
Epoch 76/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3838 - accuracy: 0.5809 - val_loss: 0.8888 - val_accuracy: 0.9048
Epoch 77/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3829 - accuracy: 0.5809 - val_loss: 0.8890 - val_accuracy: 0.9048
Epoch 78/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3820 - accuracy: 0.5809 - val_loss: 0.8897 - val_accuracy: 0.9048
Epoch 79/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3811 - accuracy: 0.5809 - val_loss: 0.8903 - val_accuracy: 0.9048
Epoch 80/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3803 - accuracy: 0.5809 - val_loss: 0.8909 - val_accuracy: 0.9048
Epoch 81/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3795 - accuracy: 0.5809 - val_loss: 0.8917 - val_accuracy: 0.9048
Epoch 82/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3786 - accuracy: 0.5809 - val_loss: 0.8922 - val_accuracy: 0.9048
Epoch 83/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3777 - accuracy: 0.5809 - val_loss: 0.8930 - val_accuracy: 0.9048
Epoch 84/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3769 - accuracy: 0.5809 - val_loss: 0.8939 - val_accuracy: 0.9048
Epoch 85/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3761 - accuracy: 0.5809 - val_loss: 0.8947 - val_accuracy: 0.9048
Epoch 86/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3753 - accuracy: 0.5809 - val_loss: 0.8951 - val_accuracy: 0.9048
Epoch 87/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3745 - accuracy: 0.5809 - val_loss: 0.8958 - val_accuracy: 0.9048
Epoch 88/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3737 - accuracy: 0.5809 - val_loss: 0.8964 - val_accuracy: 0.9048
Epoch 89/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3729 - accuracy: 0.5809 - val_loss: 0.8968 - val_accuracy: 0.9048
Epoch 90/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3721 - accuracy: 0.5809 - val_loss: 0.8978 - val_accuracy: 0.9048
Epoch 91/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3713 - accuracy: 0.5809 - val_loss: 0.8981 - val_accuracy: 0.9048
Epoch 92/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3705 - accuracy: 0.5809 - val_loss: 0.8992 - val_accuracy: 0.9048
Epoch 93/100
587/587 [==============================] - 7s 13ms/step - loss: 1.3698 - accuracy: 0.5809 - val_loss: 0.8995 - val_accuracy: 0.9048
Epoch 94/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3690 - accuracy: 0.5809 - val_loss: 0.9002 - val_accuracy: 0.9048
Epoch 95/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3682 - accuracy: 0.5809 - val_loss: 0.9004 - val_accuracy: 0.9048
Epoch 96/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3675 - accuracy: 0.5809 - val_loss: 0.9013 - val_accuracy: 0.9048
Epoch 97/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3667 - accuracy: 0.5809 - val_loss: 0.9022 - val_accuracy: 0.9048
Epoch 98/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3660 - accuracy: 0.5809 - val_loss: 0.9024 - val_accuracy: 0.9048
Epoch 99/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3652 - accuracy: 0.5809 - val_loss: 0.9034 - val_accuracy: 0.9048
Epoch 100/100
587/587 [==============================] - 7s 12ms/step - loss: 1.3645 - accuracy: 0.5809 - val_loss: 0.9038 - val_accuracy: 0.9048
6/6 [==============================] - 0s 4ms/step
"""

loss_values = parse_text(input_text)

plot_and_save_loss(loss_values, "gotowe_lr=0,0000001_epochs100_batch=1", title="Loss per epochs with Adam optimizer (learning rate=0,0000001)")
