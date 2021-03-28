# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 23:29:58 2020

@author: Clive Gomes <cliveg@andrew.cmu.edu>
@title: 10-601 Intro to Machine Learning Homework #2 Program #2
"""
import sys
import csv
import numpy as np

# Node in Decision Tree
class Node:
    def __init__(self, depth, data):
        self.depth = depth
        self.data = data
        self.attrIdx = None
        self.leftLabel = None
        self.left = None
        self.right = None
        self.parent = None
        self.childType = None
        self.decisionLabel = None

def csv_reader_to_list(reader):
    readerList = []
    for row in reader:
        readerList.append(row)
    return [np.array(readerList[0]), np.array(readerList[1:])]

def div(A, B):
    return (A/B) if B else 0

def log2(A):
    return np.log2(A) if A else 1
    
def entropy(data, attrIdx):
    # H(X) = -P(X=A)log2(P(X=A)) - P(X=B)log2(P(X=B))
    
    # Get # of Entries
    N = len(data)
    
    # Count Labels
    labelA = data[0, attrIdx]
    A = len(data[data[:, attrIdx] == labelA])
    B = N - A
    
    # Compute Probabilities of Labels
    pA = div(A,N) # P(X=A)
    pB = div(B,N) # P(X=B)
    
    # Compute Entropy
    entropy = (pA*log2(pA)) + (pB*log2(pB))
    return (-1)*entropy

def spec_cond_entropy(data, attrIdx, classIdx, labelA):
    # H(Y|X=A) = -P(Y=a|X=A)log2(P(Y=a|X=A)) - P(Y=b|X=A)log2(P(Y=b|X=A))
    # P(Y=a|X=A) = P(Y=a & X=A)/P(X=A)
    
    # Get # of Entries
    N = len(data)
    
    # Get data where X = A
    dataXisA = data[data[:, attrIdx] == labelA]
    
    # Compute P(X=A)
    A = len(dataXisA)
    pA = div(A,N)
    
    # Compute P(Y=a & X=A) and P(Y=b & X=A)
    labela = data[0, classIdx]
    n = len(dataXisA)
    a = len(dataXisA[dataXisA[:,classIdx] == labela])
    b = n - a
    paA = div(a,N) # P(Y=a & X=A)
    pbA = div(b,N) # P(Y=b & X=A)
    
    # Compute P(Y=a|X=A) and P(Y=b|X=A)
    pYa_XA = div(paA,pA) # P(Y=a|X=A)
    pYb_XA = div(pbA,pA) # P(Y=b|X=A)
    
    # Compute Special Conditional Entropy
    sc_entropy = (pYa_XA*log2(pYa_XA)) + (pYb_XA*log2(pYb_XA))
    return (-1)*sc_entropy

def cond_entropy(data, attrIdx, classIdx):
    # H(Y|X) = P(X=A)H(Y|X=A) + P(X=B)H(Y|X=B)
    
    # Get # of Entries
    N = len(data)
    
    # Get Labels
    labelA = data[0, attrIdx]
    dataB = data[data[:, attrIdx] != labelA]
    if (len(dataB) == 0):
        labelB = None
    else:
        labelB = dataB[0, attrIdx]
    
    # Compute P(X=A) and P(X=B)
    A = len(data[data[:, attrIdx] == labelA])
    B = N - A
    pA = div(A,N) # P(X=A)
    pB = div(B,N) # P(X=B)
    
    # Compute Specific Conditional Entropies
    hY_XA = spec_cond_entropy(data, attrIdx, classIdx, labelA)
    if (labelB is None):
        hY_XB = 0
    else:
        hY_XB = spec_cond_entropy(data, attrIdx, classIdx, labelB)
    
    # Compute Conditional Entropy
    return (pA*hY_XA) + (pB*hY_XB)
    
def mutual_info(data, attrIdx, classIdx):
    # I(Y;X) = H(Y) - H(Y|X)
    return entropy(data, classIdx) - cond_entropy(data, attrIdx, classIdx)
    
# Return -1 if no attribute with positive mutual information
def pick_attr_to_split_on(data, classIdx):
    mutualInfos = []
    for i in range(0, len(data[0])):
        if (i == classIdx):
            mutualInfos.append(0)
        else:
            mutualInfo = mutual_info(data, i, classIdx)
            if (mutualInfo < 0):
                mutualInfos.append(0)
            else:
                mutualInfos.append(mutualInfo)
    
    maxMutualInfo = max(mutualInfos)
    if (maxMutualInfo == 0):
        return -1
    else:        
        return mutualInfos.index(maxMutualInfo)
        
def split_dataset(data, splitIdx):
    label1 = data[0, splitIdx]
    split1 = data[data[:,splitIdx] == label1]
    split2 = data[data[:,splitIdx] != label1]
    return [split1, split2, label1]

def build_tree(data, labels, classIdx, maxDepth):
    root = Node(0, data)
    expand_tree(root, classIdx, maxDepth)
    perform_majority_voting(root, labels, classIdx)
    return root
    
def expand_tree(node, classIdx, maxDepth):
    # Check if Further Split is Possible
    if (node.depth == maxDepth or len(node.data) == 0):
        return
    
    # Get Attribute to Split On
    attrIdx = pick_attr_to_split_on(node.data, classIdx)
    if (attrIdx == -1): # Can't Split
        return
    
    node.attrIdx = attrIdx
    
    # Split Dataset on Attribute
    [data1, data2, label1] = split_dataset(node.data, node.attrIdx)
    node.leftLabel = label1
    
    # Increase Depth by 1
    leftNode = Node(node.depth+1, data1)
    rightNode = Node(node.depth+1, data2)
    
    leftNode.parent = node
    rightNode.parent = node
    
    leftNode.childType = 'l'
    rightNode.childType = 'r'
    
    node.left = leftNode
    node.right = rightNode
    
    expand_tree(leftNode, classIdx, maxDepth)
    expand_tree(rightNode, classIdx, maxDepth)
    
    return

def get_labels(data):
    labels = []
    
    for i in range(0, len(data[0])):
        labelA = data[0, i]
        dataB = data[data[:,i] != labelA]
        labelB = dataB[0, i] if (len(dataB) != 0) else None
        labels.append([labelA, labelB])
        
    return labels
    
def count_classes(data, classIdx, labelA, labelB):
    countA = len(data[data[:,classIdx] == labelA])
    countB = len(data[data[:,classIdx] == labelB])
    return [countA, countB]

def print_class_count(data, labels, classIdx):
    lA = labels[classIdx][0]
    lB = labels[classIdx][1]
    [A, B] = count_classes(data, classIdx, lA, lB)
    print("[" + str(A) + " " + lA + "/" + str(B) + " " + lB + "]")
    
def print_depth(node):
    for i in range(node.depth):
        print("|", end=" ")
    
def pretty_print(node, attrs, labels, classIdx):
    if (node is None):
        return
    
    # Print Node Details
    print_depth(node)
    
    if (node.parent is not None):
        label = node.parent.leftLabel
        if (node.childType == 'r'):
            attrLabels = labels[node.parent.attrIdx]
            label = attrLabels[attrLabels != label]
        attr = attrs[node.parent.attrIdx]
        print(attr + " = " + label + ":", end=" ")

    print_class_count(node.data, labels, classIdx)
    
    # Go to Left & Right Nodes
    pretty_print(node.left, attrs, labels, classIdx)
    pretty_print(node.right, attrs, labels, classIdx)
    
    return
       
def majority_labels(data, labels, classIdx):
    lA = labels[classIdx][0]
    lB = labels[classIdx][1]
    [A, B] = count_classes(data, classIdx, lA, lB)
   
    return pick_label(lA, lB, A, B)

def pick_label(labelA, labelB, labelA_count, labelB_count):
    if (labelA_count > labelB_count):
        label = labelA
    elif (labelA_count < labelB_count):
        label = labelB
    else:
        # Reverse-Lexical Order
        labels = [labelA, labelB]
        labels.sort(reverse=True)
        label = labels[0]
    return label

def perform_majority_voting(node, labels, classIdx):
    if (node.left is None and node.right is None):
        node.decisionLabel = majority_labels(node.data, labels, classIdx)
        return
    
    perform_majority_voting(node.left, labels, classIdx)
    perform_majority_voting(node.right, labels, classIdx)
    return
    
def get_leaf_node(entry, node):
    if (node.attrIdx is None):
        return node # Leaf Node
    
    if (entry[node.attrIdx] == node.leftLabel):
        return get_leaf_node(entry, node.left)
    else:
        return get_leaf_node(entry, node.right)
    
def predict_labels(data, root):
    preds = []
    for idx in range(0, len(data)):
        node = get_leaf_node(data[idx], root)
        preds.append(node.decisionLabel)
    return np.array(preds)

def evaluate_predictions(data, preds, classIdx):
    total_preds = len(preds)
    erroneous_preds = len(preds[data[:,classIdx] != preds])
    
    return (erroneous_preds / total_preds)

# Main Routine
if __name__ == '__main__':
    # Read Command Line Arguments
    train_input = sys.argv[1] 
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out_path = sys.argv[4]
    test_out_path = sys.argv[5]
    metrics_out_path = sys.argv[6]
  
    # Open Dataset Files
    train_file = open(train_input)
    test_file = open(test_input)
    
    # Read Datasets
    [train_attrs, train_data] = csv_reader_to_list(csv.reader(train_file, delimiter='\t')) 
    [test_attrs, test_data] = csv_reader_to_list(csv.reader(test_file, delimiter='\t'))

    # Get Dataset Properties
    classIdx = len(train_data[0]) - 1
    labels = get_labels(train_data)
    
    # -- Training Phase --
    
    # Build Decision Tree w/ Majority Voting at Leaf Nodes
    root = build_tree(train_data, labels, classIdx, max_depth)
    
    # Print Tree
    pretty_print(root, train_attrs, labels, classIdx)

    
    # -- Testing Phase --

    # Predict on Training Data
    train_preds = predict_labels(train_data, root)
    
    # Evaluate Predictions for Training Data
    train_eval = evaluate_predictions(train_data, train_preds, classIdx)

    # Predict on Testing Data
    test_preds = predict_labels(test_data, root)
    
    # Evaluate Predictions for Testing Data
    test_eval = evaluate_predictions(test_data, test_preds, classIdx)
    
    
    # -- Saving Results --

    # Open Output Files
    train_out = open(train_out_path, "w")
    test_out = open(test_out_path, "w")
    metrics_out = open(metrics_out_path, "w")

    # Write Training Predictions
    for idx in range(0, len(train_preds)):
        train_out.write(train_preds[idx] + '\n')
    
    # Write Testing Predictions
    for idx in range(0, len(test_preds)):
        test_out.write(test_preds[idx] + '\n')

    # Write Prediction Metrics
    metrics_out.write('error(train): ' + str(train_eval) + '\n')
    metrics_out.write('error(test): ' + str(test_eval) + '\n')
    
    # Close Files
    train_file.close()
    test_file.close()
    train_out.close()
    test_out.close()
    metrics_out.close()
    