from LSTM_models import *

if __name__ == '__main__':
    mode = 'use_keras'  # ['use_tf', 'use_keras', 'use_torch', 'self_implement']
    
    hidden_size = 500
    num_epochs = 50
    num_steps = 30
    batch_size = 20

    if mode == 'use_keras':
        train_data, valid_data, test_data, vocabulary, reversed_dictionary = keras_data()
        classifier = Keras_LSTM(vocabulary, hidden_size, num_steps)
        classifier.fit(train_data, valid_data, batch_size, num_epochs)
        classifier.evaluate(test_data, reversed_dictionary)
    elif mode == 'use_torch':
        trainloader, testloader = Torch_data()
        classifier = Torch_LSTM(num_classes)
        classifier.fit(trainloader, epochs=5, batch_size=10, learning_rate = learning_rate)
        classifier.evaluate(testloader)      
    elif mode == 'use_tf':
        classifier = TF_LSTM() # TODO
        classifier.fit(X_train, Y_train)
    elif mode == 'self_implement': # self-implement
        classifier = Skylark_LSTM(input_size, hidden_sizes, num_classes)
        classifier.fit(X_train, Y_train, epochs=100, batch_size=10, learning_rate = learning_rate)
    else:
        print('Attention: Wrong Mode!')