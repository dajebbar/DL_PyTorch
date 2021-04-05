import torch
import torch.nn
import torch.nn.functional as F 

class FMnistNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''


        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        layer_size = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1,h2) for h1,h2 in layer_size])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)
    

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''

        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = F.dropout(x)
        
        x = self.output(x)

        return F.log_softmax(x, dim=1)
    

def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0

    for images, labels in testloader:
        images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data.type(torch.FloatTensor) == ps.max(dim=1)[1])
        accuracy += equality.mean()

    return test_loss, accuracy


def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=40):
    steps = 0
    running_loss = 0

    for epoch in range(epochs):
        # Model in training mode, dropout is on
        model.train()

        for images, labels in trainloader:
            steps += 1

            images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)
                

                print(f"Epoch: {epoch+1}/{epochs}..",
                f"Training Loss: {running_loss/print_every}..",
                f"Test Loss: {test_loss/len(testloader): .3f}..",
                f"Test Accuracy: {accuracy/len(testloader): .3f}")

                running_loss = 0
                
                # Make sure dropout and grads are on for training
                model.train()

    



