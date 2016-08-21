--
-- Created by IntelliJ IDEA.
-- User: Jun LU
-- Date: 8/21/16
-- Time: 10:43 AM
-- To change this template use File | Settings | File Templates.
--

require 'nn'
require 'paths'
if (not paths.filep("./data/cifar10torchsmall.zip")) then
    print('Do not have data, loading...')
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
end

trainset = torch.load('./data/cifar10-train.t7')
testset = torch.load('./data/cifar10-test.t7')
classes = {'airplane', 'automobile', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse', 'ship', 'truck' }


-- DATA PREPROCESSING --
setmetatable(trainset,
    {__index = function(t, i)
        return {t.data[i], t.label[i]}
    end}
);
trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.

function trainset:size()
    return self.data:size(1)
end

-- picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
redChannel = trainset.data[{ {}, {1}, {}, {}  }]
print(#redChannel)
print(redChannel:size())

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction

    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

testset.data = testset.data:double()   -- convert from Byte tensor to Double tensor
for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end
print('Test set contains '..testset.data:size(1)..' samples')
-- END DATA PREPROCESSING

net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 6, 5, 5)) -- 3 input image channels, 6 output channels, 5x5 convolution kernel
net:add(nn.ReLU())                       -- non-linearity
net:add(nn.SpatialMaxPooling(2,2,2,2))     -- A max-pooling operation that looks at 2x2 windows and finds the max.
net:add(nn.SpatialConvolution(6, 16, 5, 5))
net:add(nn.ReLU())                       -- non-linearity
net:add(nn.SpatialMaxPooling(2,2,2,2))
net:add(nn.View(16*5*5))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
net:add(nn.Linear(16*5*5, 120))             -- fully connected layer (matrix multiplication between input and weights)
net:add(nn.ReLU())                       -- non-linearity
net:add(nn.Linear(120, 84))
net:add(nn.ReLU())                       -- non-linearity
net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
net:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classification problems

criterion = nn.ClassNLLCriterion()

trainer = nn.StochasticGradient(net, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 5 -- just do 5 epochs of training.

trainer:train(trainset)


-- TODO: print the mean and standard-deviation of example-100
print(classes[testset.label[100]])
horse = testset.data[100]
print(horse:mean(), horse:std())

predicted = net:forward(testset.data[100])

-- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x
print(predicted:exp())

for i=1,predicted:size(1) do
    print(classes[i], predicted[i])
end

-- Get accuracy for test set
correct = 0
for i=1,testset.data:size(1) do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
end

print(correct, 100*correct/testset.data:size(1) .. ' % ')

-- what are the classes that performed well, and the classes that did not perform well
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
for i=1,testset.data:size(1) do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
end

for i=1,#classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
end