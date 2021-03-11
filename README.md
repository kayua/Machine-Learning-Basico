# Introdução ao Machine Learning

Exemplo aplicação de aprendizado de máquinas a diversos problemas.

## How the system works?
The software consists of three recurrent convolutive neural networks, each of which plays a specific role in the classification of insects.

`1 - Neural network that identifies insect noises.(ultra light)`

`2 - Neural network for removing ambient noise.`

`3 - Neural network for species classification.`


## Commands:

    Optional Parameters
    
        --bands       |   Define output bands of mel scale.
        --frames      |   Define number of frames.
        --class       |   Define number of mosquito class.
        --epochs      |   Define number of training epochs.
        --steps       |   Define number of steps per epochs.
        --jumps       |   Define size jumps.
        --fft         |   Define dimension of transform fourie.
        --rate        |   Define the sample rate.
        --gain        |   Define audio gain.
        --logarithms  |   Define the logarithm scale.
        --splits      |   Define number splits.
        --dataset     |   Define dataset input directory.
        --samples     |   Define pre-processed samples input directory.
        --save        |   Define directory save models.
    
    Required Parameters
        
        Training         | This command allow training your model.
        Predict          | This command is reserved for developers.
        Help             | This command show this message.
    
    Tools Parameters
    
        GetPreprocessing | This command allow create dataset training pre-processed.
        GetSamples       | This command allow create samples to training and tests.   
        Evaluation       | This command allow evaluation your model.
        Scales           | This command allow visualizing sound scales with your dataset.
     

## Requirements:

`numpy 1.18.5`
`Keras 2.4.3`
`tqdm 4.48.2`
`tensorflow 2.3.0`

`h5py 2.10.0`
`librosa 0.8.0`
`scikit-learn 0.23.2`
`matplotlib 3.2.2`

`SoundFile 0.10.3.post1`
`scipy 1.4.1`
