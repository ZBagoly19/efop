import SZTAKI_generic_net_creation_and_train as net_train
trainClass = net_train.trainClass()


# megadott valtozok nelkul a default lesz
trainClass.setup()
my_nets0 = trainClass.train()


net_train.settings.net_type = "fully connected"      # "fully connected" or "convolutional"
net_train.settings.net_name = "fc2"                 # fc1, fc2, cnn1
net_train.settings.optim_type = "adam"             # "adam"
net_train.settings.weight_decay_ = 0.0
net_train.settings.NUM_OF_INPUT_DATA_TYPES_1st_dim = 10
net_train.settings.NUM_OF_INPUT_DATA_TYPES_2st_dim = 10
net_train.settings.NUM_OF_INPUT_DATA_TYPES = 100
net_train.settings.NUM_OF_OUTPUT_DATA_TYPES = 10
net_train.settings.train_vector = range(1, 100, 2)
net_train.settings.test_vector = range(2, 100, 2)
net_train.settings.EPOCHS = 100
net_train.settings.SAVE_EVERY_EPOCH = 100
net_train.settings.LearnRateDropFactor = 0.7
net_train.settings.LearnRateDropPeriod = 150
net_train.settings.learn_rate = 0.01
net_train.settings.weight_decay_ = 0.00005
net_train.settings.num_of_layers = 8
net_train.settings.width = 10
net_train.settings.FILE_PRE = 'Mate_'
net_train.settings.DATA_FILE = 'Mate_data_train_2d__.npy'


# setuppal amit beallitottunk
trainClass.setup()

my_nets1 = trainClass.train()


# mas valtozokkal
net_train.settings.net_type = "convolutional"
net_train.settings.net_name = "cnn1"
net_train.settings.learn_rate = 0.01
net_train.settings.EPOCHS = 200
net_train.settings.SAVE_EVERY_EPOCH = 200
trainClass.setup()

my_nets2 = trainClass.train()
