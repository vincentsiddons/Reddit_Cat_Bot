from PIL import Image
import clip
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import urllib.request
import glob
import shutil

class Model:

    learning_rate = 0.001
    epochs = 10
    model = None
    preprocess = None
    preprocessor = None


    def __init__(self, new_learning_rate, new_epochs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.learning_rate = new_learning_rate
        self.epochs = new_epochs
        self.model, self.preprocess = clip.load("ViT-B/16", device=device, jit=False)
    
    #Returns an array of cat breeds
    def get_breeds(self):
        dataset = torchvision.datasets.ImageFolder(os.path.abspath(__file__)[:os.path.abspath(__file__).index("Model.py")] + "\\cats_dataset\\test")
        #Put each breed into an array
        classes = dataset.class_to_idx.keys()
        return np.array([key for key in classes])
    
    #Load pickle file. If not, train
    def get_model(self):
        try:
            with open(os.path.abspath(__file__)[:os.path.abspath(__file__).index("Model.py")] + "\\clip_model.pkl", 'rb') as f:
                model = pickle.load(f)
                return model
        except:
            model = self.train_model()
            return model

    #Returns the losses for plotting
    #I chose not to do epochs because for fine-tuning you usually only need a few
    def train_model(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr= self.learning_rate)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        #get path of where images are
        abs_path = os.path.abspath(__file__)
        rel_path = abs_path[:abs_path.index("Model.py")] 
        transform = transforms.Compose([self.preprocess])
        #initalize dataset
        dataset = torchvision.datasets.ImageFolder(rel_path + "\\cats_dataset\\train", transform= transform)
        #To get lables for images from dataset generated indicies
        curr_map = dataset.class_to_idx
        inver_map = {v: k for k, v in curr_map.items()}

        data_loader = torch.utils.data.DataLoader(dataset, batch_size= 8, shuffle= True)
        loss_list = []
        count_list = []
        count = 0
        for i in range(0, self.epochs):
            for images, text in data_loader:
                new_text = []
                #Getting lables for each image in batch
                for i in range(0, len(text)):
                    new_text.append(inver_map[text[i].item()])
                new_text = clip.tokenize(new_text)
                image_logits, text_logits = self.model(images, new_text)
                golden = torch.arange(len(images), dtype=torch.long, device=device)
                img_loss = torch.nn.CrossEntropyLoss()
                text_loss = torch.nn.CrossEntropyLoss()
                loss = (img_loss(image_logits, golden) + text_loss(text_logits, golden))/2
                print("Loss: " + str(loss.item()))
                optimizer.zero_grad()
                loss.backward(retain_graph= True)
                optimizer.step()
                loss_list.append(loss.item())
                count += 1
                count_list.append(count)
            valid_model()
        with open(rel_path + "\\clip_model.pkl", "wb") as pickle_file:
            pickle.dump((self.model), pickle_file)
        return count_list, loss_list

    #Returns counts and validation losses lists
    def valid_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.model.to(device)
        #get path of where images are
        abs_path = os.path.abspath(__file__)
        rel_path = abs_path[:abs_path.index("Model.py")] 
        transform = transforms.Compose([self.preprocess])
        #initalize dataset
        dataset = torchvision.datasets.ImageFolder(rel_path + "\\cats_dataset\\validation", transform= transform)
        #To get lables for images from dataset generated indicies
        curr_map = dataset.class_to_idx
        inver_map = {v: k for k, v in curr_map.items()}

        data_loader = torch.utils.data.DataLoader(dataset, batch_size= 8, shuffle= True)
        loss_list = []
        count_list = []
        count = 0
        for i in range(0, self.epochs):
            for images, text in data_loader:
                new_text = []
                #Getting lables for each image in batch
                for i in range(0, len(text)):
                    new_text.append(inver_map[text[i].item()])
                new_text = clip.tokenize(new_text)
                image_logits, text_logits = self.model(images, new_text)
                golden = torch.arange(len(images), dtype=torch.long, device=device)
                img_loss = torch.nn.CrossEntropyLoss()
                text_loss = torch.nn.CrossEntropyLoss()
                loss = (img_loss(image_logits, golden) + text_loss(text_logits, golden))/2
                print("Loss: " + str(loss.item()))
                loss_list.append(loss.item())
                count += 1
                count_list.append(count)
        return count_list, loss_list

    #Obtains accuracy from test set
    def test_model(self):
        #Load pickle file. If not, train
        model = self.get_model()

        #Transform the image so it is like training data
        transform = transforms.Compose([self.preprocess])
        dataset = torchvision.datasets.ImageFolder(os.path.abspath(__file__)[:os.path.abspath(__file__).index("Train.py")] + "\\cats_dataset\\test", transform= transform)
        #Put each breed into an array
        classes = self.get_breeds()
        #To get lables for image from dataset generated index
        curr_map = dataset.class_to_idx
        inver_map = {v: k for k, v in curr_map.items()}

        data_loader = torch.utils.data.DataLoader(dataset, batch_size= 1, shuffle= True)
        #Get most likely cat breed, according to model, and see if it matches true cat breed 
        count = 0
        tp = 0
        for images,text in data_loader:
            true_class = inver_map[text.item()]
            new_text = clip.tokenize(classes)
            image_logits, text_logits = model(images, new_text)
            new_image = image_logits.softmax(dim=-1).detach().numpy()
            predicted = classes[np.argmax(new_image)]
            if(true_class == predicted):
                tp += 1
            count += 1
        acc = tp/count
        return acc
    
    #Returns breed given image URL scraped from reddit
    def get_breed(self, image_url):
        abs_path = os.path.abspath(__file__)
        rel_path = abs_path[:abs_path.index("Model.py")]
        urllib.request.urlretrieve(image_url, rel_path + "local-filename.jpg")
        with Image.open(rel_path + "local-filename.jpg") as image:
            image = self.preprocess(image).unsqueeze(0)
            classes = self.get_breeds()
            text = clip.tokenize(classes)
            model = self.get_model()
            image_logits, text_logits = model(image, text)
            image_probs = image_logits.softmax(dim=-1).detach().numpy()
            label =  classes[np.argmax(image_probs)]
            os.remove(rel_path + "/local-filename.jpg")
            return label

    def __str__(self):
        return "The learning rate is: " + self.learning_rate + ". The number of epochs is: " + self.epochs + "."
