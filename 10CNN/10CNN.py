from CNN_models import *

if __name__ == '__main__':
    mode = 'use_torch'  # ['use_tf', 'use_keras', 'use_torch', 'self_implement']
    
    hidden_sizes = [12, 8]
    num_classes = 10
    learning_rate = 0.0001

    if mode == 'use_keras':
        X_train, Y_train, X_test, Y_test = keras_data()
        classifier = Keras_CNN(input_size, hidden_sizes, num_classes)
        classifier.fit(X_train, Y_train, epochs=100, batch_size=32, learning_rate = learning_rate)
        classifier.evaluate(X_test, Y_test)
    elif mode == 'use_torch':
        trainloader, testloader = Torch_data()
        classifier = Torch_CNN(num_classes)
        classifier.fit(trainloader, epochs=5, batch_size=10, learning_rate = learning_rate)
        classifier.evaluate(testloader)      
    elif mode == 'use_tf':
        classifier = TF_CNN() # TODO
        classifier.fit(X_train, Y_train)
    elif mode == 'self_implement': # self-implement
        classifier = Skylark_CNN(input_size, hidden_sizes, num_classes)
        classifier.fit(X_train, Y_train, epochs=100, batch_size=10, learning_rate = learning_rate)
    else:
        print('Attention: Wrong Mode!')


