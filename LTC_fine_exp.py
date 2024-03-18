from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import tensorflow as tf


import ale_py
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from ncps.tf import LTC
import numpy as np
from ncps.datasets.tf import AtariCloningDatasetTF
import gym
from tensorflow.keras.callbacks import ModelCheckpoint


class ConvBlock(tf.keras.models.Sequential):
    def __init__(self):
        super(ConvBlock, self).__init__(
            [
                tf.keras.Input((84, 84, 4)),
                tf.keras.layers.Lambda(
                    lambda x: tf.cast(x, tf.float32) / 255.0
                ),  # normalize input
                tf.keras.layers.Conv2D(
                    64, 5, padding="same", activation="relu", strides=2
                ),
                tf.keras.layers.Conv2D(
                    128, 5, padding="same", activation="relu", strides=2
                ),
                tf.keras.layers.Conv2D(
                    128, 5, padding="same", activation="relu", strides=2
                ),
                tf.keras.layers.Conv2D(
                    256, 5, padding="same", activation="relu", strides=2
                ),
                tf.keras.layers.GlobalAveragePooling2D(),
            ]
        )


class ImpalaConvLayer(tf.keras.models.Sequential):
    def __init__(self, filters, kernel_size, strides, padding="valid", use_bias=False):
        super(ImpalaConvLayer, self).__init__(
            [
                tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    use_bias=use_bias,
                    kernel_initializer=tf.keras.initializers.VarianceScaling(
                        scale=2.0, mode="fan_out", distribution="truncated_normal"
                    ),
                ),
                tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001),
                tf.keras.layers.ReLU(),
            ]
        )


class ImpalaConvBlock(tf.keras.models.Sequential):
    def __init__(self):
        super(ImpalaConvBlock, self).__init__(
            [
                ImpalaConvLayer(filters=16, kernel_size=8, strides=4),
                ImpalaConvLayer(filters=32, kernel_size=4, strides=2),
                ImpalaConvLayer(filters=32, kernel_size=3, strides=1),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=256, activation="relu"),
            ]
        )


class ConvLTC(tf.keras.Model):
    def __init__(self, n_actions, units = 4, mixed_memory=False, go_backwards=False, stateful=False, ode_unfolds=6):
        super().__init__()
        self.conv_block = ImpalaConvBlock()
        self.td_conv = tf.keras.layers.TimeDistributed(self.conv_block)
        # EDIT : 1  
        self.rnn = LTC(units=units, mixed_memory=mixed_memory, go_backwards=go_backwards, stateful=stateful, ode_unfolds=ode_unfolds, return_sequences=True, return_state=True)
        self.linear = tf.keras.layers.Dense(n_actions)


    def get_initial_states(self, batch_size=1):
        return self.rnn.cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)

    def call(self, x, training=None, **kwargs):
        has_hx = isinstance(x, list) or isinstance(x, tuple)
        initial_state = None
        if has_hx:
            # additional inputs are passed as Copyright 2022 Mathias Lechner a tuple
            x, initial_state = x

        x = self.td_conv(x, training=training)
        x, next_state = self.rnn(x, initial_state=initial_state)
        x = self.linear(x)
        if has_hx:
            return (x, next_state)
        return x

def run_closed_loop(model, env, num_episodes=None):
    obs = env.reset()
    hx = model.get_initial_states()
    returns = []
    total_reward = 0
    while True:
        # add batch and time dimension (with a single element in each)
        obs = np.expand_dims(np.expand_dims(obs, 0), 0)
        pred, hx = model.predict((obs, hx), verbose=0)
        action = pred[0, 0].argmax()
        # remove time and batch dimension -> then argmax
        # obs, r, term, trunc, _ = env.step(action)
        # done = term or trunc
        obs, r, done, _ = env.step(action)
        total_reward += r
        if done:
            returns.append(total_reward)
            total_reward = 0
            obs = env.reset()
            hx = model.get_initial_states()
            # Reset RNN hidden states when episode is over
            if num_episodes is not None:
                # Count down the number of episodes
                num_episodes = num_episodes - 1
                if num_episodes == 0:
                    return returns
            if num_episodes is None:
                print(
                    f"Return {returns[-1]:0.2f} [{np.mean(returns):0.2f} +- {np.std(returns):0.2f}]"
                )



class EvalCSVCallback(tf.keras.callbacks.Callback):
    def __init__(self,model,valloader,inp_data,loss_fxn):
        super().__init__()
        self.model = model
        self.valloader = valloader
        self.inp_data = inp_data
        self.loss_fxn = loss_fxn

    def on_epoch_end(self,epoch,logs=None):
        all_pred_labels = []
        all_true_labels = []
        total_loss = 0
        for inputs, labels in self.valloader:
            outputs = self.model.predict(inputs,verbose=0)
            loss = self.loss_fxn(labels, outputs)

            # Store predictions and true labels
            pred_labels = tf.argmax(outputs, axis=-1).numpy()
            all_pred_labels.extend(pred_labels)
            all_true_labels.extend(labels.numpy())

            # Accumulate the total loss
            total_loss += loss.numpy()

        # Calculate metrics
        precision = precision_score(all_true_labels, all_pred_labels, average='weighted', labels=np.unique(all_pred_labels))
        recall = recall_score(all_true_labels, all_pred_labels, average='weighted', labels=np.unique(all_pred_labels))
        f1 = f1_score(all_true_labels, all_pred_labels, average='weighted', labels=np.unique(all_pred_labels))
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        average_loss = total_loss / len(self.valloader)

        register_metrics(self.inp_data["LR"],self.inp_data["batch_size"],self.inp_data["epochs"],epoch,self.inp_data["units"],average_loss,accuracy,precision,f1,recall)


def register_metrics(LR,batch_size,epochs,epoch,hidden_size,loss,accuracy,precision_score,f1,recall,file_path = "data_LTC_fine.csv"):
    import os,csv
    if not os.path.exists(file_path):
        # File does not exist, create it
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Learing rate","batch_size","epochs","current_epoch","hidden_size","loss","accuracy","precision_score","f1","recall"])
        print(f'CSV file "{file_path}" created.')

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write a new row
        writer.writerow([LR,batch_size,epochs,epoch,hidden_size,loss,accuracy,precision_score,f1,recall])

def run_test(LR = 0.0001,epochs = 10,units = 4, mixed_memory=False, go_backwards=False, stateful=False, ode_unfolds=6):
    batch_size = 32
    
    env = gym.make("ALE/Breakout-v5")
    # env = gymnasium.make("GymV26Environment-v0", env_id="ALE/Breakout-v5")
    env = wrap_deepmind(env)

    data = AtariCloningDatasetTF("breakout")
    # batch size 32
    trainloader = data.get_dataset(32, split="train")
    valloader = data.get_dataset(32, split="val")

    model = ConvLTC(env.action_space.n,units = units, mixed_memory=mixed_memory, go_backwards=go_backwards, stateful=stateful, ode_unfolds=ode_unfolds)
    loss_fxn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(
        loss=loss_fxn,
        optimizer=tf.keras.optimizers.Adam(0.0001),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.build((None, None, 84, 84, 4))


    inp_data = {
        'batch_size':batch_size,
        'epochs':epochs,
        'LR':LR,
        'units':units, 
        'mixed_memory':mixed_memory,
        'go_backwards':go_backwards,
        'stateful':stateful,
        'ode_unfolds':ode_unfolds
    }


    model.summary()
    model.fit(
        trainloader,
        epochs=epochs,
        validation_data=valloader,
        callbacks=[EvalCSVCallback(model,valloader=valloader,inp_data=inp_data,loss_fxn = loss_fxn)],
    )
    


def visualize(model):
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    env = wrap_deepmind(env)
    run_closed_loop(model, env)

import time
for mixed_memory in (False,True):
    for stateful in (False,True):
        for go_backwards in (False,True):
            for ode_unfolds in (5,6,7):
                t = time.time()
                run_test(epochs=5,mixed_memory=mixed_memory,stateful=stateful,go_backwards=go_backwards,ode_unfolds=ode_unfolds)
                print(f"\n\n\n\ntest time :-{time.time()-t}\n\n\n\n")