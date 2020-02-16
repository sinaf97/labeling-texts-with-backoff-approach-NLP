import collections
from prettytable import PrettyTable
from src.Document import *


class NLP:
    def __init__(self):
        self.classes = {}
        self.test_classes = {}
        self.data = collections.OrderedDict()

    def get_data(self, path):
        """ Get data from file """
        with open(path) as f:
            for line in f:
                temp = line.split('@@@@@@@@@@')
                if temp[0] not in self.classes:
                    self.classes[temp[0]] = DocumentClassification(temp[0])
                self.classes.get(temp[0]).documents.append(Document(temp[0], temp[1]))

    def train(self, path):
        """ Train data """
        self.get_data(path)                     # Fetch data
        self.calculating_probabilities()        # Creat word bag
        self.calculate_class_probability()      # Assign probability of each class

    def calculating_probabilities(self):
        """ Calculating unary, binary and backoff probabilities """
        for i in self.classes.values():
            i.calculate_unary_probabilities()
            i.calculate_binary_probabilities()
            i.binary_back_off()

    def calculate_class_probability(self):
        """ Calculating class probability """
        total_docs = sum([len(i.documents) for i in self.classes.values()])
        for i in self.classes.values():
            i.p = len(i.documents)/total_docs

    def assign_class(self, doc, mode):
        """ Assigning a class to the input document based on the mode(unary, binary, backoff model) requested"""
        result = {}
        if mode == 'U':
            for key, value in self.classes.items():
                result[key] = value.get_unary_probability(doc)
        elif mode == 'B':
            for key, value in self.classes.items():
                result[key] = value.get_binary_probability(doc)
        elif mode == 'S':
            for key, value in self.classes.items():
                result[key] = value.get_smoothed_probability(doc)
        else:
            raise Exception("Not a valid mode")
        m = max(result.values())
        for key, value in result.items():
            if value == m:
                return key

    def get_test_data(self, path):
        """ Get test data from file """
        self.test_classes = {}
        with open(path) as f:
            for line in f:
                temp = line.split('@@@@@@@@@@')
                if temp[0] not in self.test_classes:
                    self.test_classes[temp[0]] = DocumentClassification(temp[0])
                self.test_classes.get(temp[0]).documents.append(Document(temp[0], temp[1]))

    def predict(self, mode, path):
        """ Predicts all corpus classes of the test file """
        count = 0
        total_count = 0
        self.get_test_data(path)
        for i in self.test_classes.keys():
            self.data[i] = collections.OrderedDict()
            self.data[i] = {k: 0 for k in self.test_classes.keys()}
        for key, value in self.test_classes.items():
            for doc in value.documents:
                status = self.assign_class(doc, mode)
                self.data[key][status] = self.data.get(key).get(status) + 1
                if status == key:
                    count += 1
                total_count += 1
        print(count/total_count)

    def unary_predict(self, path):
        self.predict('U', path)

    def binary_predict(self, path):
        self.predict('B', path)

    def smoothed_predict(self, path):
        self.predict('S', path)

    def print_table(self):
        """ Printing the result as a table """
        t = PrettyTable([""] + [i for i in self.test_classes.keys()])

        for key, value in self.data.items():
            t.add_row([key, ] + [v for v in value.values()])
        print(t)


if __name__ == "__main__":
    model = NLP()                           # Initiating the model
    model.train('Train.txt')                # Training the model
    model.smoothed_predict('custom_test.txt')      # Predict test data
    model.print_table()                     # Print the result

