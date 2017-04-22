
root_path = '/home/jyh/github/Teaism/datasets/mnist/';

with open('train.txt') as f:
  lines = f.readlines()

f = open('train_abs.txt', 'w');
for line in lines:
  tokens = line.split()
  filename = tokens[0]
  label = tokens[1]
  f.write(root_path + filename + ' ' + label + '\n')
