import mnist
from CNN_models import *

if __name__ == '__main__':
    mode = 'self_implement'  # ['use_tf', 'use_keras', 'use_torch', 'self_implement']
    
    hidden_sizes = [12, 8]
    num_classes = 10
    learning_rate = 0.0001

    if mode == 'use_keras':
        X_train, Y_train, X_test, Y_test = keras_data(num_classes)
        input_size = X_train.shape[1:]
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
        # X_train = mnist.train_images()[:1000] ## mnist data
        # Y_train = mnist.train_labels()[:1000]
        # X_test = mnist.test_images()[:1000]
        # Y_test = mnist.test_labels()[:1000]

        X_train, Y_train, X_test, Y_test = keras_data(num_classes) ## CIFAR10 data
        X_train = rgb2gray(X_train)[:1000]          # (32, 32, 3) -> (32, 32)
        X_test = rgb2gray(X_test)
        Y_train = np.argmax(Y_train, axis=1)[:1000] # (n, 10) -> (n,)
        Y_test = np.argmax(Y_test, axis=1)
        
        classifier = Skylark_CNN(num_classes)
        classifier.fit(X_train, Y_train, epochs=1, batch_size=10, learning_rate = learning_rate)
        classifier.evaluate(X_test, Y_test)
    else:
        print('Attention: Wrong Mode!')
    
    


