import SZTAKI_constants_ as consts
import SZTAKI_generic_net_creation_and_train as net_train
trainClass = net_train.trainClass()


consts.net_type = "convolutional"      # "fully connected" or "convolutional"
consts.net_name = "cnn1"

consts.optim_type = "adam"             # "adam"
consts.weight_decay_ = 0.0

consts.NUM_OF_INPUT_DATA_TYPES_1st_dim = 10
consts.NUM_OF_INPUT_DATA_TYPES_2st_dim = 10
consts.NUM_OF_INPUT_DATA_TYPES = 100
consts.NUM_OF_OUTPUT_DATA_TYPES = 10

consts.train_vector = [0, 1, 2]
consts.test_vector = range(3, 5, 2)

consts.EPOCHS = 1500
consts.SAVE_EVERY_EPOCH = 1500

consts.LearnRateDropFactor = 0.7
consts.LearnRateDropPeriod = 150

consts.learn_rate = 0.1
# weight_decay = 0.00005

consts.num_of_layers = 0
consts.width = 10

consts.FILE_PRE = 'Mate_'



# consts file nelkul
trainClass.setup()

my_nets0 = trainClass.train()


# consts file-al
trainClass.get_variables_from_constsfile()
trainClass.setup()

my_nets1 = trainClass.train()


# consts file-al, mas valtozokkal
consts.net_type = "convolutional"
consts.net_name = "cnn1"
consts.learn_rate = 0.01
consts.EPOCHS = 2000
consts.SAVE_EVERY_EPOCH = 2000
# trainClass.get_variables_from_constsfile()
trainClass.setup()

my_nets2 = trainClass.train()

