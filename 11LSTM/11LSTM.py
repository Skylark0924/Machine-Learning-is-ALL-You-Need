from LSTM_models import *

if __name__ == '__main__':
    mode = 'self_implement'  # ['use_tf', 'use_keras', 'use_torch', 'self_implement']
    embed_size = 128
    hidden_size = 1024
    num_epochs = 50
    batch_size = 20
    learning_rate = 0.002
    num_layers = 1
    num_samples = 1000     # number of words to be sampled
    seq_length = 30

    if mode == 'use_keras': # For more info: https://adventuresinmachinelearning.com/keras-lstm-tutorial/
        train_data, valid_data, test_data, vocab_size, reversed_dictionary = keras_data()
        model = Keras_LSTM(vocab_size, hidden_size, seq_length)
        model.fit(train_data, valid_data, batch_size, num_epochs)
        model.evaluate(test_data, reversed_dictionary)
    elif mode == 'use_torch': # Sorry everyone! 
        ## I cannot find a way to package train and fit into class
        corpus = Corpus() # Two dictionaries of idx2word: {id: 'vocabulary'} and word2idx: {'vocabulary': id}
        ids = corpus.get_data('./dataset/PTB_data/ptb.train.txt', batch_size) # (bacth_size, 46479)
        vocab_size = len(corpus.dictionary)
        num_batches = ids.size(1) // seq_length
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = Torch_LSTM(vocab_size, embed_size, hidden_size, num_layers).to(device)
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            # 初始化隐状态和细胞状态
            init_state = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                            torch.zeros(num_layers, batch_size, hidden_size).to(device) )
            
            for i in range(0, ids.size(1) - seq_length, seq_length):
                # Get mini-batch inputs and targets
                inputs = ids[:, i:i+seq_length].to(device)          # (batch_size, seq_length)
                targets = ids[:, (i+1):(i+1)+seq_length].to(device) # (batch_size, seq_length)
                # Forward pass
                init_state = detach(init_state)
                outputs, states = model(inputs, init_state)
                loss = criterion(outputs, targets.reshape(-1))
                # Backward and optimize
                model.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), 0.5)
                optimizer.step()

                step = (i+1) // seq_length
                if step % 100 == 0:
                    print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                        .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

        # model.evaluate(testloader)      
    elif mode == 'use_tf':
        model = TF_LSTM() # TODO
        model.fit(X_train, Y_train)
    elif mode == 'self_implement': # self-implement

        model = Skylark_LSTM(len(train_data), hidden_size, seq_length)
        model.fit(train_data, batch_size, num_epochs, init_state)
    else:
        print('Attention: Wrong Mode!')