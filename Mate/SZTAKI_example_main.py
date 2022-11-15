import SZTAKI_constants_ as consts
consts.net_type = "fully connected"      # "fully connected" or "convolutional"
consts.optim_type = "adam"             # "adam"

consts.NUM_OF_INPUT_DATA_TYPES_1st_dim = 10
consts.NUM_OF_INPUT_DATA_TYPES_2st_dim = 10
consts.NUM_OF_INPUT_DATA_TYPES = 100
consts.NUM_OF_OUTPUT_DATA_TYPES = 10
consts.EPOCHS = 1500
consts.SAVE_EVERY_EPOCH = 1500

consts.LearnRateDropFactor = 0.7
consts.LearnRateDropPeriod = 150

consts.learn_rate = 0.1
# weight_decay = 0.00005

consts.num_of_layers = 0
consts.width = 10

consts.FILE_PRE = 'Mate_'

import SZTAKI_generic_net_creation_and_train as net_train
trainClass = net_train.trainClass()

trainClass.setup()

my_nets1 = trainClass.train()
print(my_nets1[0])



consts.net_type = "convolutional"
consts.learn_rate = 0.01
consts.EPOCHS = 2000
trainClass.setup()

my_nets2 = trainClass.train()
print(my_nets2[0])
