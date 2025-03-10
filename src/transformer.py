import torch # Pytorch for encoding data set and storing in .Tensor
             # https://pytorch.org/get-started/locally/

from bigram_language_model import BigramLanguageModel as BLM

#seeding for reproducibility
torch.manual_seed(1337)
# Consntant variables / Hyperparameters
BATCH_SIZE = 32 # indepents seq will proccess in parallel
BLOCK_SIZE = 8 # max length for predictions

# Getting dataset from input.txt in the resource folder
with open('resource\input.txt','r',encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text))) #get unique characters
vocabSize = len(chars)
# Mapping Char->Int (vice versa)
strToInt = {ch:i for i,ch in enumerate(chars)}
intToStr = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [strToInt[c] for c in s] # take the string and output list of ints
decode = lambda l: ''.join([intToStr[i] for i in l]) # take the list of intergers convert it to a string

data = torch.tensor(encode(text),dtype=torch.long) # encrypts data set

# Data Training / Validation
n = int(.9 * len(data)) # 90% of data is for training, rest validation
# Lets look into what this means ^^^ -SW
train_data = data[:n] 
val_data = data[n:]
x = train_data[:BLOCK_SIZE]
y = train_data[1:BLOCK_SIZE+1]

# Functions
def getBatch(split):
      data = train_data
      if split != 'train':
           data = val_data

      ix = torch.randint(len(data) - BLOCK_SIZE,(BATCH_SIZE,))
      x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix])
      y = torch.stack([data[i+1:i+(BLOCK_SIZE+1)] for i in ix])
      return x,y


xb, yb = getBatch('train')
m = BLM(vocabSize)
logits,loss = m(xb,yb)
"""
    Testing Purposes:
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(BATCH_SIZE): # batch dimension
    for t in range(BLOCK_SIZE): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")

print(logits.shape)
print(loss)
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))"
"""

optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)
"""
for steps in range(100): # increase number of steps for good results...

    # sample a batch of data
    xb, yb = getBatch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))"
"""
print("heyyoooo")