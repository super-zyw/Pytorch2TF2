from __future__ import print_function
from utils import *
from darknet import Darknet
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from build_model import YoloV2

np.random.seed(0)

pytorch_network = Darknet('cfg/yolo-pose.cfg')
pytorch_network.load_weights_until_last('backup/benchvise/model_backup.weights')
pytorch_network.eval()
keras_network = YoloV2(416, 416)

i = 0
for name, param in pytorch_network.state_dict().items():
    print('layer: {}, name: {}'.format(i, name))
    i = i + 1

for i in range(len(keras_network.layers)):
    print('layer: {}, name: {}'.format(i, keras_network.layers[i].name))

print('start to assign weights...')

bn = pytorch_network.state_dict()
for torch_name, torch_param in pytorch_network.state_dict().items():
    i = 0
    for j in range(len(keras_network.layers)):
        if torch_name.split('.')[2] == keras_network.layers[j].name:
            print(torch_name.split('.')[2] +  '=' + keras_network.layers[j].name)

            if torch_name.split('.')[2] != 'conv23':

                try:
                    keras_network.layers[j].set_weights([torch_param.permute(2, 3, 1, 0).numpy()])
                    print('assign convolutional layer')

                except:
                    gamma = torch_name.split('.')[:3]
                    gamma.append('weight')
                    gamma = '.'.join(gamma)

                    beta = torch_name.split('.')[:3]
                    beta.append('bias')
                    beta = '.'.join(beta)

                    mean = torch_name.split('.')[:3]
                    mean.append('running_mean')
                    mean = '.'.join(mean)

                    var = torch_name.split('.')[:3]
                    var.append('running_var')
                    var = '.'.join(var)
                    print('gamma: {}, beta: {}, mean: {}, var: {}'.format(gamma, beta, mean, var))
                    keras_network.layers[j].set_weights([bn[gamma], bn[beta], bn[mean], bn[var]])
                    print('assign bn layer')
            else:
                weight = bn['models.30.conv23.weight'].permute(2, 3, 1, 0).numpy()
                bias = bn['models.30.conv23.bias'].numpy()
                keras_network.layers[j].set_weights([weight, bias])
        i += 1
print('success')

#put random input and test if the results consistent
tensor = np.random.randint(0, 255, size = (1, 416, 416, 3))
keras_result = keras_network(tf.cast(tensor, tf.float32))

torch_result = pytorch_network(torch.Tensor(tensor).permute(0, 3, 1, 2))
print('min: {}, max: {}'.format(np.min(keras_result.numpy() - torch_result.permute(0, 2, 3, 1).data.numpy()), np.max(keras_result.numpy() - torch_result.permute(0, 2, 3, 1).data.numpy())))

keras_network.save_weights('exchange_weight')
keras_network.load_weights('exchange_weight')
print('succuessfully load...')
