import torch
import torch.nn as nn
# import torchvision.models as models
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self, embeddings,lstm_hid_dim, num_classes=30, norm=True, scale=True):
        super(Net,self).__init__()
        self.extractor = Extractor(embeddings,lstm_hid_dim)
        self.embedding = Embedding()
        self.classifier = Classifier(num_classes)
        self.s = nn.Parameter(torch.FloatTensor([10]))
        self.norm = norm
        self.scale = scale

    def forward(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        if self.norm:
            x = self.l2_norm(x)
        if self.scale:
            x = self.s * x
        x = self.classifier(x)
        return x

    def extract(self, x):
        x = self.extractor(x)
        x = self.embedding(x)
        x = self.l2_norm(x)
        return x

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        # _output = torch.div(input, norm.view(0, 1).expand_as(input))
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def weight_norm(self):
        w = self.classifier.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.classifier.fc.weight.data = w.div(norm.expand_as(w))

class Extractor(nn.Module):
    def __init__(self,embeddings,lstm_hid_dim):
        super(Extractor,self).__init__()
        self.lstm_hid_dim=lstm_hid_dim
        self.embeddings = self._load_embeddings(embeddings)
        self.lstm = torch.nn.LSTM(300, hidden_size=lstm_hid_dim, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.linear_first = torch.nn.Linear(lstm_hid_dim * 2, 1)
        # basenet = models.resnet50(pretrained=True)
        # self.extractor = nn.Sequential(*list(basenet.children())[:-1])

    def forward(self, x):
        embeddings = self.embeddings(x)
        hidden_state = self.init_hidden()
        outputs, hidden_state = self.lstm(embeddings, hidden_state)
        # x = self.extractor(x)
        # selfatt =  self.linear_first(outputs)
        selfatt = self.linear_first(outputs)
        pro= F.softmax(selfatt, dim=1)
        pro=  pro.transpose(1, 2)
        out = torch.bmm(pro, outputs).squeeze()

        return out

    def _load_embeddings(self, embeddings):
        """Load the embeddings based on flag"""
        word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
        word_embeddings.weight = torch.nn.Parameter(embeddings)
        return word_embeddings

    def init_hidden(self):
        return (torch.randn(2,90,self.lstm_hid_dim).cuda(),torch.randn(2,90,self.lstm_hid_dim).cuda())

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding,self).__init__()
        self.fc = nn.Linear(300, 128)

    def forward(self, x):
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier,self).__init__()
        self.fc = nn.Linear(128, num_classes, bias=False)

    def forward(self, x):
        x1=self.fc(x)
        x = torch.sigmoid(x1)
        return x


class Transfer(nn.Module):
    def __init__(self):
        super(Transfer,self).__init__()
        self.transfor = nn.Linear(128, 128, bias=False)
        self.transfor.weight.data = self.transfor.weight.data.float()

    def forward(self, x):
        x=self.transfor(x)
        return x
