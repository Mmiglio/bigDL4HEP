from bigdl.nn.layer import *

def InclusiveModel():

    ## GRU branch
    recurrent = Recurrent()
    recurrent.add(
        GRU(
            input_size=19,
            hidden_size=50,
            activation='tanh'
            )
    )

    gruBranch = Sequential() \
                .add(Masking(0.0)) \
                .add(recurrent) \
                .add(Select(2,-1))

    ## HLF branch
    hlfBranch = Sequential() \
                .add(Dropout(0.0))

    ## Concatenate the branches
    branches = ParallelTable() \
            .add(gruBranch).add(hlfBranch)

    ## Create the model
    model = Sequential() \
            .add(branches) \
            .add(JoinTable(2,2)) \
            .add(BatchNormalization(64)) \
            .add(Linear(64,3)) \
            .add(SoftMax())
    return model

def GRUModel():
    recurrent = Recurrent()
    recurrent.add(
        GRU(
            input_size=19,
            hidden_size=50,
            activation='tanh'
            )
    )

    model = Sequential()
    model.add(Masking(0.0))
    model.add(recurrent)
    model.add(Select(2,-1))
    model.add(BatchNormalization(50))
    model.add(Linear(50,3))
    model.add(SoftMax())
    return model

def HLFmodel():
    model = Sequential()
    model.add(Linear(14,50))
    model.add(ReLU())
    model.add(Linear(50,20))
    model.add(ReLU())
    model.add(Linear(20,10))
    model.add(ReLU())
    model.add(Linear(10,3))
    model.add(SoftMax())
    return model
