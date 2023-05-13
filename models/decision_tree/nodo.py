import numpy as np
import math

class Nodo:
  def __init__(self, X, Y, classes, max_depth, depth):
    self.max_depth = max_depth
    self.depth     = depth
    # If terminal set as terminal
    if Nodo.IsTerminal(Y):
      self.label     = Nodo.GetLabel(Y)
      self.prob      = np.array([1 if cls == self.label else 0 for cls in classes])
      self.index     = None
      self.lt_child  = None
      self.ge_child  = None
    # Max depth
    elif max_depth is not None and depth>=max_depth:
      # Count which label is most common and set as label
      self.label, self.prob = Nodo.MostCommon(Y, classes)
    # If not terminal, resplit
    else:
      self.label                   = None
      self.index, self.boundary    = Nodo.BestSplit(X, Y)
      if self.boundary is None:
        # If undividable, count which label is most common and set as label
        self.label, self.prob = Nodo.MostCommon(Y, classes)
      else:
        # If dividable, recursive call
        self.lt_child, self.ge_child = Nodo.SplitByIndex(X, Y, self.index, self.boundary, classes, self.max_depth, self.depth)

  def IsTerminal(Y):
    # return true if this node have the sames labeles in Y
    label = Y[0]
    for y in Y:
      if y != label:
        return False
    return True

  def GetLabel(Y):
    # return label if terminal
    return Y[0]
  
  def MostCommon(Y, classes):
    # return most common label
    values, counts = np.unique(Y, return_counts=True)
    total = counts.sum()
    cls_count = dict()
    for v,c in zip(values, counts):
      cls_count[v] = c
    probs = [cls_count[cls]/total if cls in cls_count else 0 for cls in classes]
    ind = np.argmax(counts)
    return values[ind], probs

  def BestSplit(X, Y):
    # Evaluate (n-1) splits
    max_ent       = 0
    best_index    = None
    best_boundary = None

    # Iterate over all indexes
    for ind in range(len(X[0])):
       # Get boundary options
       options = { x[ind] for x in X }
       options = list(options)
       options = sorted(options)
       # Evaluate Entropy for each boundary. If better, save
       for b in options:
         ent = Nodo.Entropy(X, Y, ind, b)
         if ent >= max_ent:
          max_ent = ent
          best_index = ind
          best_boundary = b

    if max_ent==0:
      # Undividable
      return None, None

    # Return best options
    return best_index, best_boundary


  def SplitByIndex(X, Y, index, boundary, classes, max_depth, depth):
    # Init children X,Y
    lt_x = []
    lt_y = []
    ge_x = []
    ge_y = []
    # Split X,Y by boundary
    for x,y in zip(X,Y):
      if x[index] < boundary:
        lt_x.append(x)
        lt_y.append(y)
      else:
        ge_x.append(x)
        ge_y.append(y)
    # Create children nodes
    lt_child = Nodo(lt_x, lt_y, classes, max_depth, depth+1)
    ge_child = Nodo(ge_x, ge_y, classes, max_depth, depth+1)
    # Return reference to node children
    return lt_child, ge_child

  def Entropy(X, Y, index, boundary):
    # ENTROPY GAIN

    # Create dictionaries counting instances of each class in
    # the left and right node defined by index & boundary
    label_count_lt = dict()
    count_lt = 0
    label_count_ge = dict()
    count_ge = 0
    label_count_all = dict()
    for x, y in zip(X, Y):
      if y in label_count_all:
        label_count_all[y] += 1
      else:
        label_count_all[y] = 1
      if x[index] >= boundary:
        count_ge += 1
        if y in label_count_ge:
          label_count_ge[y] += 1
        else:
          label_count_ge[y] = 1
      else:
        count_lt += 1
        if y in label_count_lt:
          label_count_lt[y] += 1
        else:
          label_count_lt[y] = 1

    # Calc entropy in nodes
    # Left Node
    ent_lt = 0
    for label in label_count_lt:
      prob_label = label_count_lt[label]/count_lt
      ent_lt += (-prob_label * math.log(prob_label, 2))
    # Right node
    ent_ge = 0
    for label in label_count_ge:
      prob_label = label_count_ge[label]/count_ge
      ent_ge += (-prob_label * math.log(prob_label, 2))
    # Weighted nodes entropy
    total_count = count_ge + count_lt
    weighted_ent_sum = ent_ge * (count_ge/total_count) + ent_lt * (count_lt/total_count)

    # Calc total entropy
    ent_all = 0
    for label in label_count_all:
      prob_label = label_count_all[label]/total_count
      ent_all += (-prob_label * math.log(prob_label, 2))
    
    # Calc gain
    return ent_all - weighted_ent_sum
