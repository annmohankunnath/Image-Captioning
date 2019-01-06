import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super(DecoderRNN, self).__init__()
        
        self.hidden_dim = hidden_size
        
        self.embed_dim = embed_size
        
        self.vocab_size = vocab_size
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        self.hidden = (torch.zeros(1,1,hidden_size), torch.zeros(1,1,hidden_size))
    
    def forward(self, features, captions):
        
        caption_embeddings = self.word_embeddings(captions[:,:-1])
        
        embeddings = torch.cat((features.unsqueeze(1),caption_embeddings),1)
        
        lstm_output, self.hidden = self.lstm(embeddings)
        
        output = self.linear(lstm_output)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sample = []
        
        hidden = (torch.randn(1, 1, self.hidden_dim).to(inputs.device),
                  torch.randn(1, 1, self.hidden_dim).to(inputs.device))
        
        for i in range(max_len):#new version
            
            
            lstm_output, hidden = self.lstm(inputs, hidden)
            
            outputs = self.linear(lstm_output.squeeze(1))
            
            index = outputs.max(1)[1]
            
            sample.append(index.item())
            
            target_index = outputs.max(1)[1]
            
            inputs = self.word_embeddings(target_index).unsqueeze(1)
           
        return sample
            
        